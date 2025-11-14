import os
import uuid
import json
import logging
import time
import io  # For reading bytes as an image
import hashlib  # For creating the cache key

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from kafka import KafkaProducer
from prometheus_client import start_http_server, Counter, Histogram

# --- ML Imports ---
import numpy as np
from PIL import Image

# --- Configuration ---
KAFKA_BROKER = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_PREPROCESSED = os.environ.get("TOPIC_PREPROCESSED", "TOPIC_PREPROCESSED")
TOPIC_STATUS_UPDATES = os.environ.get("TOPIC_STATUS_UPDATES", "TOPIC_STATUS_UPDATES")
PROVENANCE_API_URL = os.environ.get("PROVENANCE_API_URL", "http://localhost:8001")
# This path is set in the Dockerfile
MEAN_STD_PATH = "/app/config/mean_std.json"

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

kafka_producer = None
mean_np = None
std_np = None

# --- Prometheus Metrics ---
REQUESTS_TOTAL = Counter("preprocessor_requests_total", "Total requests received")
REQUEST_LATENCY = Histogram("preprocessor_request_latency_seconds", "Request latency")
MESSAGES_SENT_TOTAL = Counter("preprocessor_kafka_messages_sent_total", "Total Kafka messages sent", ["topic"])

# --- Lifespan Events (Connect to Kafka, Load Models) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global kafka_producer, mean_np, std_np
    logger.info("Application startup...")
    
    # Start Prometheus metrics server
    start_http_server(9100)
    logger.info("Started Prometheus metrics server on port 9100")

    # Connect to Kafka
    try:
        kafka_producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        logger.info(f"Connected to Kafka at {KAFKA_BROKER}")
    except Exception as e:
        logger.error(f"Failed to connect to Kafka: {e}")
        # Note: The app will fail API calls if this is not set

    # Load mean/std and store as numpy arrays
    try:
        logger.info(f"Loading mean/std from {MEAN_STD_PATH}")
        with open(MEAN_STD_PATH, "r") as f:
            mean_std = json.load(f)
        
        # Reshape for broadcasting (C, 1, 1)
        mean_np = np.array(mean_std["mean"], dtype=np.float32).reshape(3, 1, 1)
        std_np = np.array(mean_std["std"], dtype=np.float32).reshape(3, 1, 1)
        
        logger.info("Mean/Std arrays created successfully.")
    except Exception as e:
        logger.error(f"Failed to load ML transforms: {e}")
        # Note: The app will fail API calls if this is not set
        
    yield
    
    # Shutdown
    if kafka_producer:
        kafka_producer.close()
    logger.info("Application shutdown.")

app = FastAPI(lifespan=lifespan)

# --- Helper Function ---
def send_to_kafka(topic, value):
    try:
        kafka_producer.send(topic, value=value)
        MESSAGES_SENT_TOTAL.labels(topic=topic).inc()
    except Exception as e:
        logger.error(f"Failed to send to Kafka topic {topic}: {e}")

# --- Preprocessing Logic (Re-written with Numpy/Pillow) ---
def preprocess_image(image_bytes: bytes) -> list:
    """
    Load and preprocess image bytes for inference using Pillow and Numpy.
    This replicates:
    - transforms.Resize((128, 128))
    - transforms.ToTensor()
    - transforms.Normalize(mean=mean, std=std)
    - .unsqueeze(0)
    """
    global mean_np, std_np
    if mean_np is None or std_np is None:
        raise RuntimeError("Mean/Std arrays are not loaded")
        
    # 1. Load image from in-memory bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 2. Resize
    image = image.resize((128, 128), Image.Resampling.LANCZOS)
    
    # 3. Replicate transforms.ToTensor()
    # Convert to numpy array
    image_np = np.array(image, dtype=np.float32)
    # Scale from [0, 255] to [0.0, 1.0]
    image_np = image_np / 255.0
    # Change from (H, W, C) to (C, H, W)
    image_np = np.transpose(image_np, (2, 0, 1))

    # 4. Replicate transforms.Normalize()
    image_np = (image_np - mean_np) / std_np
    
    # 5. Replicate .unsqueeze(0) to add batch dimension (B, C, H, W)
    image_np = np.expand_dims(image_np, axis=0)
    
    # 6. Convert to list for JSON serialization
    return image_np.tolist()

# --- API Endpoint ---
@app.post("/v1/infer")
async def infer(request: Request, file: UploadFile = File(...)):
    start_time = time.time()
    REQUESTS_TOTAL.inc()

    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "File must be an image"})

    # Updated check
    if not kafka_producer or mean_np is None:
        return JSONResponse(status_code=503, content={"error": "Service not ready, dependencies not loaded"})

    try:
        image_bytes = await file.read()
        
        # 1. Generate Request ID
        request_id = str(uuid.uuid4())
        
        # 2. Generate cache key from *original* image bytes
        cache_key = hashlib.md5(image_bytes).hexdigest()
        
        # 3. Run preprocessing
        processed_image_tensor_list = preprocess_image(image_bytes)

        # 4. Create Kafka messages
        status_message = {
            "request_id": request_id,
            "status": "PENDING",
            "submitted_at": time.time(),
        }
        
        image_message = {
            "request_id": request_id,
            "cache_key": cache_key, # For the feature extractor to use
            "image_tensor_list": processed_image_tensor_list # The preprocessed data
        }

        # 5. Send to Kafka
        send_to_kafka(TOPIC_STATUS_UPDATES, status_message)
        send_to_kafka(TOPIC_PREPROCESSED, image_message)
        kafka_producer.flush() # Ensure messages are sent before responding
        
        logger.info(f"Submitted request {request_id} to Kafka topics")

        # 6. Return 202 Accepted response
        status_url = f"{PROVENANCE_API_URL}/v1/status/{request_id}"
        
        REQUEST_LATENCY.observe(time.time() - start_time)
        
        return JSONResponse(
            status_code=202,
            content={
                "request_id": request_id,
                "status_url": status_url
            }
        )
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
