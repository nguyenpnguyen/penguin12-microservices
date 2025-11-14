FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

RUN hf download nguyenpn/Penguin-12 backbone_weights.pt --local-dir /app/models

COPY ./app .

EXPOSE 9100
CMD ["python", "main.py"]
