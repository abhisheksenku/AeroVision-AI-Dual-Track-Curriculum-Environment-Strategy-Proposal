"""
=============================================================================
AeroVision AI - Mission Analytics REST API Routes
=============================================================================
Endpoints for querying structured mission summaries, speed metrics, and violation alerts.
"""

import json
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.db.models import JobRecord, MissionAnalyticsRecord, ViolationEventRecord

router = APIRouter(prefix="/analytics", tags=["analytics"])

@router.get("/{job_id}")
def get_mission_analytics(job_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Retrieves full structured mission analytics for a completed job."""
    analytics = db.query(MissionAnalyticsRecord).filter(MissionAnalyticsRecord.job_id == job_id).first()
    if not analytics:
        job = db.query(JobRecord).filter(JobRecord.job_id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        if job.status in ["QUEUED", "PROCESSING"]:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Job is still {job.status} (progress: {job.progress_percent}%). Analytics not yet ready."
            )
        raise HTTPException(status_code=404, detail="Analytics not found for this job.")

    # Retrieve violations
    violations = db.query(ViolationEventRecord).filter(ViolationEventRecord.job_id == job_id).all()
    violations_log = [
        {
            "event_id": v.event_id,
            "rule_type": v.rule_type,
            "track_id": v.track_id,
            "class_name": v.class_name,
            "timestamp_sec": round(v.timestamp_sec, 2),
            "severity": v.severity,
            "details": v.details,
            "location": (round(v.location_x, 1), round(v.location_y, 1))
        }
        for v in violations
    ]

    return {
        "mission_id": analytics.mission_id,
        "job_id": analytics.job_id,
        "duration_sec": round(analytics.duration_sec, 2),
        "total_unique_objects": analytics.total_unique_objects,
        "counts_by_class": json.loads(analytics.counts_by_class_json or "{}"),
        "class_statistics": json.loads(analytics.class_stats_json or "{}"),
        "line_crossings": json.loads(analytics.line_crossings_json or "{}"),
        "safety_audit": {
            "total_violations": analytics.total_violations,
            "compliance_status": analytics.compliance_status,
            "violations_log": violations_log
        }
    }

@router.get("/{job_id}/violations")
def get_job_violations(
    job_id: str,
    severity: Optional[str] = Query(None, description="Filter by severity: LOW, MEDIUM, HIGH, CRITICAL"),
    db: Session = Depends(get_db)
):
    """Retrieves security violation alerts for a job with optional severity filtering."""
    query = db.query(ViolationEventRecord).filter(ViolationEventRecord.job_id == job_id)
    if severity:
        query = query.filter(ViolationEventRecord.severity == severity.upper())

    violations = query.all()
    return [
        {
            "event_id": v.event_id,
            "rule_type": v.rule_type,
            "track_id": v.track_id,
            "class_name": v.class_name,
            "timestamp_sec": round(v.timestamp_sec, 2),
            "severity": v.severity,
            "details": v.details,
            "location": (round(v.location_x, 1), round(v.location_y, 1))
        }
        for v in violations
    ]
