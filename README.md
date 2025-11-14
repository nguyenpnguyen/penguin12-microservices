# Penguin12: Asynchronous Deepfake Detection Microservice

This project is a scalable, event-driven system for deepfake detection, based on [Penguin12](https://huggingface.co/nguyenpn/Penguin-12) model trained by Team Penguin for our Capstone Project at RMIT University Vietnam. It's built with a microservice architecture using Python, FastAPI, PyTorch, and Kafka.

The system accepts an image, schedules it for analysis, and provides a non-blocking API for clients to poll for the final detection results.

## Table of Contents

* [Features](#-features)
* [Architecture](#-architecture)
  * [Data Flow Diagram](#data-flow-diagram)
  * [Service Directory](#service-directory)
* [Technology Stack](#-technology-stack)
* [How to Run the Project](#-how-to-run-the-project)
  * [Prerequisites](#prerequisites)
  * [Configuration](#configuration)
  * [Running the System](#running-the-system)
* [How to Use the API](#-how-to-use-the-api)
  * [Step 1: Submit an Image for Analysis](#step-1-submit-an-image-for-analysis)
  * [Step 2: Poll for Results](#step-2-poll-for-results)
* [Connecting Clients (Web Apps & Extensions)](#-connecting-clients-web-apps--extensions)
* [Observability & Monitoring](#-observability--monitoring)

---

## Features

* **Asynchronous API:** The client-facing API is non-blocking. It accepts a job and immediately returns a "ticket," allowing clients to remain responsive.
* **Event-Driven & Decoupled:** Services communicate via a Kafka message bus. This means the `preprocessor` can accept 10,00s of requests per second, even if the `feature_extractor` is slow, without crashing the system.
* **Resilient & Fault-Tolerant:** If a service (like the `classifier`) crashes, the unprocessed message remains in the Kafka queue. The service will process it upon restarting, ensuring no data is lost.
* **Scalable:** The `feature_extractor` (the slowest, most expensive part) can be scaled horizontally by simply running more containers (`docker-compose up --scale feature_extractor=3`).
* **Centralized State:** A dedicated `provenance` service tracks the status of every request in a Redis database.
* **Intelligent Caching:** The `feature_extractor` caches all computed embeddings in Redis. If the same image is submitted twice, the expensive CNN inference is skipped.
* **Fully Observable:** The system is pre-configured with Prometheus for metrics, Grafana for dashboards, and Jaeger for distributed tracing.

---

## Architecture

This project uses an event-driven microservice architecture. Services are split by function and communicate asynchronously using Kafka.

### Data Flow

There are three main data flows:

1. **Job Submission (REST):** A client `POST`s an image to the `preprocessor`. The `preprocessor` publishes jobs to Kafka and immediately returns a `202 Accepted` response with a `status_url`.
2. **Processing Pipeline (Kafka):**
    * `preprocessor` -> `TOPIC_PREPROCESSED`
    * `feature_extractor` consumes from this topic, runs the CNN, and produces to: -> `TOPIC_FEATURES`
    * `classifier` consumes from this topic, runs the head, and produces to: -> `TOPIC_STATUS_UPDATES`
3. **Status Tracking (Kafka & REST):**
    * All services send status updates (`PENDING`, `FAILED`, `COMPLETED`, etc.) to the `TOPIC_STATUS_UPDATES`.
    * The `provenance` service is the *only* consumer of this topic, updating its Redis DB.
    * The client `GET`s the `status_url`, which hits the `provenance` API.

### Diagram

graph TD
    subgraph User
        Client
    end

    subgraph "Application Services (The Pipeline)"
        direction LR
        Pre[Preprocessor (Port 8000)]
        FE[Feature Extractor (Worker)]
        Class[Classifier (Worker)]
        Prov[Provenance (Port 8001)]
    end

    subgraph "Infrastructure (Messaging & Storage)"
        direction LR
        Zoo[(Zookeeper)]
        Kafka[(Kafka Message Bus)]
        Redis[(Redis: DB0 Status, DB1 Cache)]
    end

    subgraph "Observability Suite"
        direction LR
        Prom[Prometheus (Port 9090)]
        Graf[Grafana (Port 3000)]
        Jaeger[Jaeger (Port 16686)]
    end
    
    %% 1. Main Data Flow
    Client -- 1. POST /v1/infer (Image) --> Pre
    Pre -- 2. Publishes (Preprocessed Tensor) --> Kafka
    Kafka -- 3. Consumes (Tensor) --> FE
    FE -- 4. Publishes (Embedding) --> Kafka
    Kafka -- 5. Consumes (Embedding) --> Class

    %% 2. Status & Polling Flow
    Pre -- Publishes (PENDING) --> Kafka
    FE -- Publishes (PROCESSING) --> Kafka
    Class -- Publishes (COMPLETED / FAILED) --> Kafka
    Kafka -- 6. Consumes ALL Statuses --> Prov
    Client -- 7. GET /v1/status (Polling) --> Prov
    
    %% 3. Database & Cache
    Prov -- Reads/Writes Status --> Redis
    FE -- Reads/Writes Cache --> Redis
    
    %% 4. Infrastructure Management
    Kafka -- Managed by --> Zoo
    
    %% 5. Observability Flow
    Prom -- Scrapes /metrics --> Pre
    Prom -- Scrapes /metrics --> FE
    Prom -- Scrapes /metrics --> Class
    Prom -- Scrapes /metrics --> Prov
    Graf -- Queries --> Prom
    
    %% (All services also push traces to Jaeger)

### Service Directory

| Service | Technology | Port(s) | Description |
| :--- | :--- | :--- | :--- |
| **`preprocessor`** | FastAPI, Numpy | `8000` | Exposes the public `POST /v1/infer` REST API. Preprocesses images and publishes jobs to Kafka. |
| **`feature_extractor`** | Python, PyTorch | - | A worker that consumes images, runs the CNN backbone, caches embeddings in Redis, and publishes features. |
| **`classifier`** | Python, PyTorch | - | A worker that consumes features, runs the classifier head, and publishes the final prediction. |
| **`provenance`** | FastAPI, Kafka | `8001` | Consumes all status updates and saves them to Redis. Exposes the public `GET /v1/status/...` API for polling. |
| **`kafka`** | Kafka | `9092` | The central message bus. Decouples all services. |
| **`zookeeper`** | Zookeeper | `2181` | Manages the Kafka cluster (required for Kafka v6.x). |
| **`redis`** | Redis | `6379` | Provides two databases: **DB 0** (State Store for `provenance`) and **DB 1** (Cache for `feature_extractor`). |
| **`prometheus`** | Prometheus | `9090` | Scrapes the `/metrics` endpoint on all services to collect time-series data. |
| **`grafana`** | Grafana | `3000` | Visualizes metrics from Prometheus. (Default login: `admin`/`admin`). |
| **`jaeger`** | Jaeger | `16686` | Collects and visualizes distributed traces for debugging bottlenecks. |

---

## Technology Stack

* **Application:** Python 3.12+, FastAPI, PyTorch, Numpy, Pillow
* **Infrastructure:** Docker, Docker Compose
* **Messaging:** Kafka, Zookeeper
* **Database / Cache:** Redis
* **Observability:** Prometheus, Grafana, Jaeger
* **Package Management:** `uv` (for local development)

---

## How to Run

### Prerequisites

* Docker
* Docker Compose

### Configuration

1. **Models:** This project expects your model files to be available. The `Dockerfile` for `feature_extractor` and `classifier` are set up to download them from Hugging Face Hub.
    * Update `services/feature_extractor/Dockerfile` to point to your `backbone_weights.pt` file.
    * Update `services/classifier/Dockerfile` to point to your `classifier_head.pt` file.
    * Ensure the model definitions in `model.py` for each service match the saved weights.

2. **Dependencies:** Dependencies for each service are managed in their respective `pyproject.toml` files.

3. **Ports:** All ports are defined in `docker-compose.yml`. Ensure these ports are free.

### Running the System

1. **Start all services:**
    This command will build all service images (using `uv` for speed) and start them in detached mode.

    ```bash
    docker-compose up --build -d
    ```

2. **Watch the logs:**
    To see all services starting up and connecting, you can stream the logs.

    ```bash
    docker-compose logs -f
    ```

    You should see all workers successfully connect to Kafka and Redis (after a few retries, which is normal).

3. **Stop the system:**
    This command stops and removes all containers, networks, and volumes.

    ```bash
    docker-compose down -v
    ```

---

## How to Use the API

The API is asynchronous. You must make two separate calls to get a result.

### Step 1: Submit an Image for Analysis

You send your image file to the `preprocessor`.

* **Endpoint:** `POST /v1/infer`
* **Port:** `8000`
* **Body:** `multipart/form-data` with a key named `file`.

**Example (`curl`):**

```bash
curl -X POST "[http://127.0.0.1:8000/v1/infer](http://127.0.0.1:8000/v1/infer)" \
     -F "file=@/path/to/your/image.jpg"

```

**Success Response** (`202 Accepted`): The server accepts your job and gives you a URL to check for the status.

```json
{
  "request_id": "14f90e53-c54b-45e2-8177-0da3fed6c646",
  "status_url": "http://provenance:8001/v1/status/14f90e53-c54b-45e2-8177-0da3fed6c646"
}
```

**Note**: The status_url returned uses the internal Docker hostname (provenance). Clients (like a browser) must replace this with 127.0.0.1:8001.

### Step 2: Poll for Results

Periodically call the `status_url` given in Step 1 until the `status` changes from `PENDING` to `COMPLETED` or `FAILED`.

* **Endpoint**: `GET /v1/status/{request_id}`
* **Port**: `8001`

**Example** (`curl`):

```bash
curl "[http://127.0.0.1:8001/v1/status/14f90e53-c54b-45e2-8177-0da3fed6c646](http://127.0.0.1:8001/v1/status/14f90e53-c54b-45e2-8177-0da3fed6c646)"
```

**Pending Response** (`200 OK`): This is what you will see while the job is in the queue or being processed.

```json
{
  "request_id": "14f90e53-c54b-45e2-8177-0da3fed6c646",
  "status": "PROCESSING_FEATURES",
  "submitted_at": 1731600000.123,
  "last_updated": 1731600001.456
}
```

**Completed Response** (`200 OK`): This is the final result.

```json
{
  "request_id": "14f90e53-c54b-45e2-8177-0da3fed6c646",
  "status": "COMPLETED",
  "submitted_at": 1731600000.123,
  "last_updated": 1731600003.789,
  "result": {
    "prediction": "FAKE",
    "confidence": 0.9876
  }
}
```

**Failed Response** (`200 OK`): If something goes wrong (e.g., in the classifier), the error is reported.

```json
{
  "request_id": "14f90e53-c54b-45e2-8177-0da3fed6c646",
  "status": "FAILED",
  "submitted_at": 1731600000.123,
  "last_updated": 1731600002.345,
  "error": "Incorrect label names"
}
```

---

## Observability & Monitoring

The system exposes several dashboards for monitoring:

* **Prometheus**: `http://localhost:9090`
  * (View raw metrics and target status)
* **Grafana**: `http://localhost:3000`
  * (Build dashboards. Login: `admin`/`admin`. Add Prometheus as a data source: `http://prometheus:9090`)
* **Jaeger**: `http://localhost:16686`
  * (View distributed traces)
