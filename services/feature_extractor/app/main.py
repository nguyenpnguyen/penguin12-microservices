import os
import time
import json
import asyncio
from typing import Tuple

import numpy as np
import grpc
from fastapi import FastAPI
import uvicorn
from kafka import KafkaProducer

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.grpc import GrpcInstrumentorServer

# Proto imports: ensure you've generated these from protos/embedding.proto
# See README.md for generation command.
try:
    from protos import embedding_pb2, embedding_pb2_grpc
except Exception:
    # Fallback import error message - generated stubs must exist
    embedding_pb2 = None
    embedding_pb2_grpc = None

# OTEL setup (simple)
JAEGER_HOST = os.environ.get("JAEGER_HOST", "jaeger")
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "redpanda:9092")
MODEL_VERSION = os.environ.get("FE_MODEL_VERSION", "fe_v0")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))

# minimal OpenTelemetry initialization (also in otel.py if you prefer)
provider = TracerProvider()
otlp_exporter = OTLPSpanExporter(endpoint=f"http://{JAEGER_HOST}:4317", insecure=True)
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# Kafka producer (JSON for now). For protobuf, serialize to bytes.
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

app = FastAPI(title="feature-extractor")
FastAPIInstrumentor.instrument_app(app)
# Instrument gRPC server later (in Grpc server startup)
grpc_instrumentor = GrpcInstrumentorServer()
grpc_instrumentor.instrument()


def make_traceparent_from_current_span() -> Tuple[str, bytes]:
    """
    Build an RFC W3C traceparent header string from current span context.
    Return tuple (header_str, header_bytes) - Kafka headers need bytes.
    Format: "00-{traceid}-{spanid}-01"
    """
    span = trace.get_current_span()
    ctx = span.get_span_context()
    # If there's no valid span, return empty
    if not ctx or not ctx.trace_id:
        return "", b""
    trace_id = format(ctx.trace_id, "032x")
    span_id = format(ctx.span_id, "016x")
    traceparent = f"00-{trace_id}-{span_id}-01"
    return traceparent, traceparent.encode("utf-8")


# --- Embedding generation logic (placeholder) ---
def compute_embedding(image_bytes: bytes) -> list:
    """Placeholder: compute random embedding. Replace with real model."""
    # deterministic pseudo-random for reproducibility if needed
    rng = np.random.default_rng(int(time.time() * 1000) % (2 ** 32))
    emb = (rng.standard_normal(512)).astype(float).tolist()
    return emb


# --- gRPC servicer implementation ---
if embedding_pb2 is not None and embedding_pb2_grpc is not None:
    class FeatureExtractorServicer(embedding_pb2_grpc.FeatureExtractorServiceServicer):
        async def Extract(self, request, context):
            """
            gRPC handler for Extract(ImageRequest) -> EmbeddingResponse
            request: embedding_pb2.ImageRequest
            """
            request_id = getattr(request, "request_id", "req-none")
            # start traced span
            with tracer.start_as_current_span("feature_extractor.extract") as span:
                span.set_attribute("request.id", request_id)
                span.set_attribute("model.version", MODEL_VERSION)

                # In real code, handle request.image_bytes or reference
                image_bytes = request.image_bytes if hasattr(request, "image_bytes") else b""
                embedding = compute_embedding(image_bytes)
                embed_id = f"embed-{int(time.time()*1000)}"

                event = {
                    "request_id": request_id,
                    "image_id": getattr(request, "image_id", ""),
                    "embedding": embedding,
                    "model_version": MODEL_VERSION,
                    "timestamp": int(time.time() * 1000),
                }

                # inject traceparent header into kafka message headers
                traceparent_str, traceparent_bytes = make_traceparent_from_current_span()
                headers = []
                if traceparent_str:
                    headers.append(("traceparent", traceparent_bytes))

                # publish to Kafka 'embeddings' topic
                producer.send("embeddings", value=event, headers=headers)
                producer.flush()

                # build response proto
                resp = embedding_pb2.EmbeddingResponse(
                    request_id=request_id, image_id=event["image_id"], embed_id=embed_id
                )
                return resp
else:
    FeatureExtractorServicer = None


# --- gRPC server runner ---
async def serve_grpc(host="0.0.0.0", port=50052):
    if FeatureExtractorServicer is None:
        print("Proto stubs not found. gRPC server will not start. Generate protos first.")
        return

    server = grpc.aio.server()
    embedding_pb2_grpc.add_FeatureExtractorServiceServicer_to_server(FeatureExtractorServicer(), server)
    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    print(f"[gRPC] starting server on {listen_addr}")
    await server.start()
    await server.wait_for_termination()


# --- FastAPI endpoints (health / debug) ---
@app.get("/health")
async def health():
    return {"status": "ok", "service": "feature_extractor", "model_version": MODEL_VERSION}


@app.post("/admin/generate_test_embedding")
async def admin_gen(request: dict):
    """
    Synchronous test endpoint to generate embedding and publish to kafka.
    Body: { "request_id": "...", "image_id": "..." }
    """
    request_id = request.get("request_id", f"req-{int(time.time()*1000)}")
    image_id = request.get("image_id", f"img-{int(time.time()*1000)}")
    with tracer.start_as_current_span("feature_extractor.admin_generate"):
        emb = compute_embedding(b"")
        event = {
            "request_id": request_id,
            "image_id": image_id,
            "embedding": emb,
            "model_version": MODEL_VERSION,
            "timestamp": int(time.time() * 1000),
        }
        traceparent_str, traceparent_bytes = make_traceparent_from_current_span()
        headers = []
        if traceparent_str:
            headers.append(("traceparent", traceparent_bytes))
        producer.send("embeddings", value=event, headers=headers)
        producer.flush()
        return {"published": True, "request_id": request_id, "image_id": image_id}


# --- Entrypoint: run FastAPI and gRPC concurrently ---
def start_uvicorn():
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")


def main():
    loop = asyncio.get_event_loop()
    # Run gRPC server in background task
    if FeatureExtractorServicer is not None:
        grpc_task = loop.create_task(serve_grpc(host="0.0.0.0", port=50052))
    else:
        grpc_task = None

    # Run uvicorn in a thread to keep things simple
    # uvicorn.run blocks, so run it in executor to keep asyncio loop alive
    from concurrent.futures import ThreadPoolExecutor

    executor = ThreadPoolExecutor(max_workers=1)
    loop.run_in_executor(executor, start_uvicorn)

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        if grpc_task:
            grpc_task.cancel()
        producer.close()


if __name__ == "__main__":
    main()
