"""
=============================================================================
AeroVision AI - Video Processing Jobs REST API Routes
=============================================================================
Endpoints for video uploads, asynchronous processing dispatch, and status polling.
"""

import os
import uuid
import shutil
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import select, desc
from pydantic import BaseModel

from backend.app.db.database import get_db
from backend.app.db.models import JobRecord

router = APIRouter(prefix="/jobs", tags=["jobs"])

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class JobResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    progress_percent: int
    total_frames: int
    processed_frames: int
    annotated_video_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str

    class Config:
        from_attributes = True

@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_drone_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Accepts video file upload, persists it to disk, creates a JobRecord in the DB,
    and enqueues the CV analysis background worker.
    """
    filename = video.filename or "uploaded_video.mp4"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in [".mp4", ".avi", ".mov", ".mkv"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video extension: '{ext}'. Allowed: ['.mp4', '.avi', '.mov', '.mkv']"
        )

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    saved_filename = f"{job_id}_{filename}"
    saved_path = os.path.join(UPLOAD_DIR, saved_filename)

    # Stream file to disk
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    if os.path.getsize(saved_path) == 0:
        os.remove(saved_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )

    # Insert into database
    job_record = JobRecord(
        job_id=job_id,
        filename=filename,
        file_path=saved_path,
        status="QUEUED",
        progress_percent=0
    )
    db.add(job_record)
    db.commit()

    # Enqueue background task
    from backend.app.services.worker import process_video_job
    background_tasks.add_task(process_video_job, job_id)

    return {
        "job_id": job_id,
        "filename": filename,
        "status": "QUEUED",
        "message": "Video successfully uploaded and scheduled for background CV analysis."
    }

@router.get("/{job_id}", response_model=JobResponse)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Retrieves current processing status and progress for a job."""
    job = db.query(JobRecord).filter(JobRecord.job_id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found for ID: {job_id}"
        )
    return JobResponse(
        job_id=job.job_id,
        filename=job.filename,
        status=job.status,
        progress_percent=job.progress_percent,
        total_frames=job.total_frames,
        processed_frames=job.processed_frames,
        annotated_video_path=job.annotated_video_path,
        error_message=job.error_message,
        created_at=job.created_at.isoformat()
    )

@router.get("", response_model=List[JobResponse])
def list_jobs(limit: int = 50, db: Session = Depends(get_db)):
    """Lists all historical processing jobs ordered by creation date."""
    jobs = db.query(JobRecord).order_by(desc(JobRecord.created_at)).limit(limit).all()
    return [
        JobResponse(
            job_id=j.job_id,
            filename=j.filename,
            status=j.status,
            progress_percent=j.progress_percent,
            total_frames=j.total_frames,
            processed_frames=j.processed_frames,
            annotated_video_path=j.annotated_video_path,
            error_message=j.error_message,
            created_at=j.created_at.isoformat()
        )
        for j in jobs
    ]
