FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Download the model config into a specific config directory
# This runs as part of the image build
RUN hf download nguyenpn/Penguin-12 mean_std.json --local-dir /app/config

COPY ./app .

EXPOSE 8000 9100
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
