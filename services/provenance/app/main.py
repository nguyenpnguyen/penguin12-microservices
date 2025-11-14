import os
import json
import logging
import time
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import redis
from kafka import KafkaConsumer
from prometheus_client import start_http_server, Counter, Histogram

# --- Configuration ---
KAFKA_BROKER = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
TOPIC_STATUS_UPDATES = os.environ.get("TOPIC_STATUS_UPDATES", "TOPIC_STATUS_UPDATES")
CONSUMER_GROUP = "provenance-service"

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

redis_client = None

# --- Prometheus Metrics ---
REQUESTS_STATUS_TOTAL = Counter("provenance_status_requests_total", "Total status poll requests")
REQUEST_LATENCY_REST = Histogram("provenance_rest_latency_seconds", "REST API latency")
STATUS_UPDATES_TOTAL = Counter("provenance_status_updates_total", "Total status updates received", ["status"])

# --- Kafka Consumer Logic (runs in a background thread) ---
def start_kafka_consumer():
    logger.info("Starting Kafka consumer thread...")
    
    consumer = None
    while consumer is None:
        try:
            consumer = KafkaConsumer(
                TOPIC_STATUS_UPDATES,
                bootstrap_servers=KAFKA_BROKER,
                group_id=CONSUMER_GROUP,
                value_deserializer=lambda v: json.loads(v.decode('utf-8'))
            )
            logger.info("Kafka consumer connected.")
        except Exception as e:
            logger.error(f"Failed to connect Kafka consumer, retrying in 5s: {e}")
            time.sleep(5)
            
    logger.info("Provenance consumer started. Waiting for status updates...")
    
    for message in consumer:
        try:
            data = message.value
            request_id = data.get("request_id")
            if not request_id:
                logger.warning("Status update received without request_id, skipping.")
                continue

            status = data.get("status", "UNKNOWN")
            STATUS_UPDATES_TOTAL.labels(status=status).inc()
            logger.info(f"Updating status for {request_id} to {status}")

            key = f"status:{request_id}"
            
            # --- Get existing data ---
            existing_data_json = redis_client.get(key)
            if existing_data_json:
                state = json.loads(existing_data_json)
            else:
                state = {"request_id": request_id}
            
            # --- Merge new data into state ---
            # This allows us to add "result" or "error" keys
            state.update(data)
            state["last_updated"] = time.time()

            # --- Save back to Redis ---
            redis_client.set(key, json.dumps(state), ex=3600) # 1-hour expiry
            
        except Exception as e:
            logger.error(f"Error processing status message: {e}")

# --- Lifespan Events (Start Kafka Consumer, Connect to Redis) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    logger.info("Application startup...")

    # Start Prometheus metrics server
    start_http_server(9100)
    logger.info("Started Prometheus metrics server on port 9100")

    # Connect to Redis with retry
    while redis_client is None:
        try:
            redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
            redis_client.ping()
            logger.info(f"Connected to Redis at {REDIS_HOST}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis, retrying in 5s: {e}")
            time.sleep(5)
    
    # Start the Kafka consumer in a daemon thread
    # Daemon=True means the thread will exit when the main app exits
    consumer_thread = threading.Thread(target=start_kafka_consumer, daemon=True)
    consumer_thread.start()
    
    yield
    
    # Shutdown
    if redis_client:
        redis_client.close()
    logger.info("Application shutdown.")

app = FastAPI(lifespan=lifespan)

# --- REST API Endpoint (for Client Polling) ---
@app.get("/v1/status/{request_id}", name="get_status")
async def get_status(request_id: str):
    start_time = time.time()
    REQUESTS_STATUS_TOTAL.inc()
    
    if not redis_client:
        return JSONResponse(status_code=503, content={"error": "Service not ready, cannot connect to Redis"})

    try:
        data = redis_client.get(f"status:{request_id}")
        if not data:
            return JSONResponse(status_code=404, content={"error": "Request ID not found"})
        
        REQUEST_LATENCY_REST.observe(time.time() - start_time)
        # Return the raw JSON string from Redis
        return JSONResponse(status_code=200, content=json.loads(data))
    
    except Exception as e:
        logger.error(f"Error fetching status for {request_id}: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    # This will be started by the 'CMD' in the Dockerfile
    uvicorn.run(app, host="0.0.0.0", port=8001)
