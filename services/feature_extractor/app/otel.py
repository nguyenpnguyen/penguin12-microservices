from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
import os

def init_tracer(service_name: str = "feature_extractor"):
    jaeger_host = os.environ.get("JAEGER_HOST", "jaeger")
    provider = TracerProvider()
    otlp_exporter = OTLPSpanExporter(endpoint=f"http://{jaeger_host}:4317", insecure=True)
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer(service_name)
    return tracer
