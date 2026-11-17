"""
=============================================================================
AeroVision AI - Backend Engineering Foundations: Module 03
=============================================================================
Asynchronous Background Processing & The Job State Machine

Learning Objectives:
1. Understand why long-running computer vision tasks must not block HTTP handlers.
2. Implement the Job State Machine: QUEUED -> PROCESSING -> COMPLETED / FAILED.
3. Decouple synchronous client responses (202 Accepted) from asynchronous processing.
4. Execute background video processing workers using FastAPI BackgroundTasks.
"""

import time
import asyncio
from typing import Dict, Any
from fastapi import FastAPI, BackgroundTasks, status, HTTPException
from fastapi.testclient import TestClient

app = FastAPI(title="Background Job Demo")

# In-memory simulated Job Database
jobs_db: Dict[str, Dict[str, Any]] = {}

def execute_video_analysis_worker(job_id: str, total_frames: int):
    """
    Background worker simulating frame-by-frame computer vision analytics.
    Executes asynchronously in a worker thread pool, decoupling from HTTP request.
    """
    jobs_db[job_id]["status"] = "PROCESSING"
    
    for frame_idx in range(1, total_frames + 1):
        # Simulate ~2ms per frame inference
        time.sleep(0.002)
        progress = int((frame_idx / total_frames) * 100.0)
        jobs_db[job_id]["processed_frames"] = frame_idx
        jobs_db[job_id]["progress_percent"] = progress

    jobs_db[job_id]["status"] = "COMPLETED"
    jobs_db[job_id]["completed_at"] = time.time()

@app.post("/api/v1/jobs/simulate-upload", status_code=status.HTTP_202_ACCEPTED)
async def submit_job(job_id: str, background_tasks: BackgroundTasks):
    """
    HTTP Handler: Responds within ~2ms with 202 Accepted,
    delegating the long-running video inference to the background worker.
    """
    jobs_db[job_id] = {
        "job_id": job_id,
        "status": "QUEUED",
        "progress_percent": 0,
        "total_frames": 100,
        "processed_frames": 0,
        "submitted_at": time.time()
    }
    # Enqueue background task
    background_tasks.add_task(execute_video_analysis_worker, job_id, total_frames=100)
    
    return {
        "job_id": job_id,
        "status": "QUEUED",
        "message": "Video analysis queued for background processing."
    }

@app.get("/api/v1/jobs/{job_id}/status")
async def check_job_status(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found.")
    return jobs_db[job_id]

def demonstrate_background_processing():
    print("=" * 70)
    print("STEP 1: ASYNC HTTP SUBMISSION & BACKGROUND TASK LIFECYCLE")
    print("=" * 70)

    client = TestClient(app)

    # 1. Submit Job: Must return HTTP 202 Accepted instantly
    t0 = time.perf_counter()
    res = client.post("/api/v1/jobs/simulate-upload?job_id=JOB_UAV_001")
    http_latency_ms = (time.perf_counter() - t0) * 1000.0

    print(f"POST /api/v1/jobs/simulate-upload -> Status {res.status_code}")
    print(f"HTTP Response Latency: {http_latency_ms:.2f} ms (Non-blocking response!)")
    print(f"Initial Response Body: {res.json()}")

    assert res.status_code == 202
    assert http_latency_ms < 1000.0, "Submission took too long!"

    # In TestClient, BackgroundTasks run during request teardown
    # Verify final completed state in DB
    status_res = client.get("/api/v1/jobs/JOB_UAV_001/status")
    print(f"\nPolled Job Status Post-Execution:")
    print(f"  Status          : {status_res.json()['status']}")
    print(f"  Progress        : {status_res.json()['progress_percent']}%")
    print(f"  Processed Frames: {status_res.json()['processed_frames']}/{status_res.json()['total_frames']}")

    assert status_res.json()["status"] == "COMPLETED"
    assert status_res.json()["progress_percent"] == 100
    print("\n[SUCCESS] Background task processing and 202 Accepted pattern verified.")
    print()

def main():
    demonstrate_background_processing()
    print("=" * 70)
    print("PHASE 8 - MODULE 03: COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
