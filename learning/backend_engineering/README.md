# Phase 8: Backend & Database Engineering

This module covers backend API engineering for high-throughput drone video systems using FastAPI, Pydantic v2, and SQLAlchemy 2.0.

---

## Modules Overview

| Script | Title | Key Concepts & Formulations |
| :--- | :--- | :--- |
| `01_fastapi_async_routes.py` | FastAPI Routing & Contracts | Async request cycles, Pydantic v2 validation contracts, HTTP status semantics (201 Created, 202 Accepted, 400 Bad Request, 404 Not Found, 422 Unprocessable Entity). |
| `02_sqlalchemy_orm_and_persistence.py` | SQLAlchemy 2.0 Relational ORM | Declarative Base, One-to-Many foreign key cascades (Jobs $\to$ Violations), modern `select()` syntax, session transaction commits/rollbacks. |
| `03_background_task_processing.py` | Async Background Workers | Solving the synchronous video processing hazard (preventing 504 Gateway Timeouts), Job State Machine (`QUEUED` $\to$ `PROCESSING` $\to$ `COMPLETED`), non-blocking 202 responses. |
| `04_rest_api_testing_and_contracts.py` | API Testing & File Uploads | Multipart/form-data upload handling, file extension whitelisting, CORS preflight verification, FastAPI `TestClient` contract testing. |

---

## Running the Exercises

Activate the primary `.venv` environment and run:

```powershell
.\.venv\Scripts\Activate.ps1

python learning/backend_engineering/01_fastapi_async_routes.py
python learning/backend_engineering/02_sqlalchemy_orm_and_persistence.py
python learning/backend_engineering/03_background_task_processing.py
python learning/backend_engineering/04_rest_api_testing_and_contracts.py
```

---

## Backend & AI Systems Engineer Interview Questions

1. **Why must long-running AI/CV video pipelines NEVER be executed directly inside a synchronous FastAPI route handler?**
   *Answer*: Standard HTTP request handlers operate with reverse proxy timeout limits (e.g., NGINX, AWS ALB default is 30-60 seconds). A high-resolution drone video requires 15-60 seconds of deep learning inference. If executed inside the route handler, the client connection hangs, reverse proxies abort with `504 Gateway Timeout`, and the server's thread pool is starved of capacity to handle concurrent users. The standard production pattern is the **Asynchronous Job Pattern**: the upload route immediately persists the raw file, saves a job record with state `QUEUED`, kicks off a background task or message queue message (Celery, Redis, FastAPI `BackgroundTasks`), and returns `202 Accepted` with a `job_id` in $< 20\text{ ms}$. The client then polls `GET /jobs/{id}` for live progress.

2. **How does Pydantic v2 achieve up to 15x faster serialization than Pydantic v1?**
   *Answer*: Pydantic v2 completely rewrote its core validation and serialization engine in Rust (`pydantic-core`). Rather than traversing Python ASTs and invoking Python-level validator functions for every dictionary field, validation is performed in compiled Rust native code with zero-copy memory parsing, significantly reducing latency on heavy JSON video detection responses.

3. **What is the difference between `add_task` in FastAPI `BackgroundTasks` and a distributed task queue like Celery or RQ?**
   *Answer*: FastAPI `BackgroundTasks` runs tasks within the same Python process worker thread pool after sending the HTTP response. It is ideal for lightweight background jobs that do not require multi-node scaling or process isolation. However, if the server process crashes or restarts, all in-flight `BackgroundTasks` are lost. Distributed task queues (Celery, RQ, temporal.io) store job definitions in an external broker (Redis, RabbitMQ), providing persistence, automatic retries across machine crashes, and horizontal worker autoscaling across multiple GPU/CPU nodes.
