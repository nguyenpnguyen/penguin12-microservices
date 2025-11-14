FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY ./app .

EXPOSE 8001 9100
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]