import os
import json
import logging
import time

from kafka import KafkaConsumer, KafkaProducer
from prometheus_client import start_http_server, Counter, Histogram

# --- ML Imports ---
import torch
from penguin12_classifier import Penguin12Classifier

# --- Configuration ---
KAFKA_BROKER = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_FEATURES = os.environ.get("TOPIC_FEATURES", "TOPIC_FEATURES")
TOPIC_STATUS_UPDATES = os.environ.get("TOPIC_STATUS_UPDATES", "TOPIC_STATUS_UPDATES")
CONSUMER_GROUP = "classifiers"
MODEL_PATH = "models/classifier_weights.pt"
DEVICE = torch.device("cpu") # The head is small, CPU is fine

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Prometheus Metrics ---
MESSAGES_RECEIVED = Counter("classifier_messages_received_total", "Total messages received")
MESSAGES_SENT_TOTAL = Counter("classifier_messages_sent_total", "Total Kafka messages sent", ["topic"])
PREDICTIONS_TOTAL = Counter("classifier_predictions_total", "Total predictions made", ["prediction"])
PROCESSING_LATENCY = Histogram("classifier_processing_latency_seconds", "Processing latency")


# --- Model Loading ---
def load_model():
    logger.info(f"Loading classifier head model from {MODEL_PATH}")
    logger.info(f"Using device: {DEVICE}")

    model = Penguin12Classifier()
    
    # Load the state dictionary
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # load the weights directly into the 'classifier' submodule
    try:
        model.classifier.load_state_dict(state_dict)
    except RuntimeError as e:
        logger.error(f"Error loading state_dict into model.features: {e}")
        logger.error("This likely means the keys in your .pt file don't match the 'features' nn.Sequential block.")
        raise e
    # Move model to the device
    model.to(DEVICE)
    
    # --- CRITICAL: Set model to evaluation mode ---
    model.eval()
    
    logger.info("Classifier head model loaded and set to eval mode.")
    return model

# --- Main Worker Loop ---
def run_classifier():
    logger.info("Starting Classifier service...")
    start_http_server(9100)
    logger.info("Started Prometheus metrics server on port 9100")

    model = load_model()

    # --- Setup connections with retry logic ---
    consumer = None
    producer = None

    while consumer is None:
        try:
            consumer = KafkaConsumer(
                TOPIC_FEATURES,
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

    logger.info("Classifier service started. Waiting for messages...")

    for message in consumer:
        start_time = time.time()
        MESSAGES_RECEIVED.inc()
        
        data = message.value
        request_id = data.get("request_id")
        
        if not request_id:
            logger.warning("Message received without request_id")
            continue
            
        logger.info(f"Classifying request {request_id}")

        try:
            # 1. Update status to CLASSIFYING
            producer.send(TOPIC_STATUS_UPDATES, value={
                "request_id": request_id, 
                "status": "CLASSIFYING"
            })
            MESSAGES_SENT_TOTAL.labels(topic=TOPIC_STATUS_UPDATES).inc()

            # 2. Get embedding and run model
            embedding_list = data["embedding"]
            
            # Convert list back to tensor, add batch dimension (1, 512)
            embedding_tensor = torch.tensor(embedding_list).unsqueeze(0).to(DEVICE)
            
            # --- Run Inference ---
            with torch.no_grad():
                # Get the raw logit (e.g., -1.23)
                output_logit = model(embedding_tensor)
            
            # 3. Format result
            # Convert logit to probability (0.0 to 1.0)
            prob = torch.sigmoid(output_logit).item()
            
            prediction = "FAKE" if prob > 0.5 else "REAL"
            # Calculate confidence
            confidence = prob if prediction == "FAKE" else 1.0 - prob
            
            result = {
                "prediction": prediction,
                "confidence": round(confidence, 4), # Round to 4 decimal places
            }
            PREDICTIONS_TOTAL.labels(prediction=prediction).inc()

            # 4. Publish final status
            producer.send(TOPIC_STATUS_UPDATES, value={
                "request_id": request_id, 
                "status": "COMPLETED",
                "result": result
            })
            MESSAGES_SENT_TOTAL.labels(topic=TOPIC_STATUS_UPDATES).inc()
            
            producer.flush()
            PROCESSING_LATENCY.observe(time.time() - start_time)
            logger.info(f"Successfully classified request {request_id}")
            
        except Exception as e:
            logger.error(f"Failed to classify request {request_id}: {e}")
            # Send FAILED status
            producer.send(TOPIC_STATUS_UPDATES, value={
                "request_id": request_id, 
                "status": "FAILED",
                "error": str(e)
            })
            MESSAGES_SENT_TOTAL.labels(topic=TOPIC_STATUS_UPDATES).inc()
            producer.flush()

if __name__ == "__main__":
    run_classifier()
