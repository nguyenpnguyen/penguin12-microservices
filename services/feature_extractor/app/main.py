import os
import json
import logging
import time

from penguin12_cnn import Penguin12CNN
from kafka import KafkaConsumer, KafkaProducer
import redis
from prometheus_client import start_http_server, Counter, Histogram


# --- ML Imports ---
import torch

# Model import

# --- Configuration ---
KAFKA_BROKER = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
TOPIC_PREPROCESSED = os.environ.get("TOPIC_PREPROCESSED", "TOPIC_PREPROCESSED")
TOPIC_FEATURES = os.environ.get("TOPIC_FEATURES", "TOPIC_FEATURES")
TOPIC_STATUS_UPDATES = os.environ.get("TOPIC_STATUS_UPDATES", "TOPIC_STATUS_UPDATES")
CONSUMER_GROUP = "feature-extractors"
MODEL_PATH = "models/backbone_weights.pt"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Prometheus Metrics ---
MESSAGES_RECEIVED = Counter("feature_extractor_messages_received_total", "Total messages received")
MESSAGES_SENT_TOTAL = Counter("feature_extractor_messages_sent_total", "Total Kafka messages sent", ["topic"])
CACHE_HITS = Counter("feature_extractor_cache_hits_total", "Total cache hits")
CACHE_MISSES = Counter("feature_extractor_cache_misses_total", "Total cache misses")
PROCESSING_LATENCY = Histogram("feature_extractor_processing_latency_seconds", "Processing latency")

# --- Model Loading ---
def load_model():
    logger.info(f"Loading feature extractor model from {MODEL_PATH}")
    logger.info(f"Using device: {DEVICE}")

    model = Penguin12CNN()
    
    # Load the state dictionary (weights)
    # Use map_location to ensure it loads correctly onto the selected device
    state_dict =torch.load(MODEL_PATH, map_location=DEVICE)

    # load the weights directly into the 'features' submodule
    try:
        model.features.load_state_dict(state_dict)
    except RuntimeError as e:
        logger.error(f"Error loading state_dict into model.features: {e}")
        logger.error("This likely means the keys in your .pt file don't match the 'features' nn.Sequential block.")
        raise e
    
    # Move model to the device (CPU or GPU)
    model.to(DEVICE)
    
    model.eval()
    
    logger.info("Model loaded and set to eval mode.")
    return model

# --- Main Worker Loop ---
def run_feature_extractor():
    logger.info("Starting Feature Extractor service...")
    start_http_server(9100)
    logger.info("Started Prometheus metrics server on port 9100")

    model = load_model()

    # --- Setup connections with retry logic ---
    consumer = None
    producer = None
    redis_cache = None

    while consumer is None:
        try:
            consumer = KafkaConsumer(
                TOPIC_PREPROCESSED,
                bootstrap_servers=KAFKA_BROKER,
                group_id=CONSUMER_GROUP,
                value_deserializer=lambda v: json.loads(v.decode('utf-8'))
            )
            logger.info("Kafka consumer connected.")
        except Exception as e:
            logger.error(f"Failed to connect Kafka consumer, retrying in 5s: {e}")
            time.sleep(5)

    while producer is None:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            logger.info("Kafka producer connected.")
        except Exception as e:
            logger.error(f"Failed to connect Kafka producer, retrying in 5s: {e}")
            time.sleep(5)

    while redis_cache is None:
        try:
            redis_cache = redis.Redis(host=REDIS_HOST, port=6379, db=1)
            redis_cache.ping() # Test connection
            logger.info(f"Connected to Redis at {REDIS_HOST}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis, retrying in 5s: {e}")
            time.sleep(5)

    logger.info("Feature Extractor service started. Waiting for messages...")

    for message in consumer:
        start_time = time.time()
        MESSAGES_RECEIVED.inc()
        
        data = message.value
        request_id = data.get("request_id")
        
        if not request_id:
            logger.warning("Message received without request_id")
            continue
            
        logger.info(f"Processing request {request_id}")

        try:
            # 1. Update status to PROCESSING
            producer.send(TOPIC_STATUS_UPDATES, value={
                "request_id": request_id,
                "status": "PROCESSING_FEATURES"
            })
            MESSAGES_SENT_TOTAL.labels(topic=TOPIC_STATUS_UPDATES).inc()

            # 2. Check cache
            cache_key = data.get("cache_key")
            redis_key = f"cache:{cache_key}"
            embedding_json = redis_cache.get(redis_key)
            
            if embedding_json:
                embedding = json.loads(embedding_json)
                CACHE_HITS.inc()
                logger.info(f"Cache hit for request {request_id}")
            else:
                CACHE_MISSES.inc()
                logger.info(f"Cache miss for request {request_id}, running inference...")
                
                # 3. Get preprocessed data and run model
                image_tensor_list = data["image_tensor_list"]
                
                # --- Convert list back to tensor ---
                image_tensor = torch.tensor(image_tensor_list).to(DEVICE)
                
                # --- Run Inference ---
                with torch.no_grad(): # Disable gradient calculation
                    embedding_tensor = model(image_tensor)
                
                # --- Convert result back to list ---
                embedding = embedding_tensor.cpu().numpy().flatten().tolist()
                
                # 4. Store in cache (if we have a key)
                if cache_key:
                    redis_cache.set(redis_key, json.dumps(embedding), ex=86400) # 24-hour expiry

            # 5. Publish to next topic
            output_message = {
                "request_id": request_id,
                "embedding": embedding
            }
            producer.send(TOPIC_FEATURES, value=output_message)
            MESSAGES_SENT_TOTAL.labels(topic=TOPIC_FEATURES).inc()
            
            producer.flush()
            PROCESSING_LATENCY.observe(time.time() - start_time)
            logger.info(f"Successfully processed request {request_id}")
            
        except Exception as e:
            logger.error(f"Failed to process request {request_id}: {e}")
            # Send FAILED status
            producer.send(TOPIC_STATUS_UPDATES, value={
                "request_id": request_id,
                "status": "FAILED",
                "error": str(e)
            })
            MESSAGES_SENT_TOTAL.labels(topic=TOPIC_STATUS_UPDATES).inc()
            producer.flush()

if __name__ == "__main__":
    run_feature_extractor()
