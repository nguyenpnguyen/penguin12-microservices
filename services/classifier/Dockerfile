FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN hf download nguyenpn/Penguin-12 classifier_weights.pt --local-dir /app/models

COPY ./app .

EXPOSE 9100
CMD ["python", "main.py"]
