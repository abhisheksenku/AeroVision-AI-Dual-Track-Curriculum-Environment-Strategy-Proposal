"""
=============================================================================
AeroVision AI - LLM Mission Reports REST API Routes
=============================================================================
Endpoints for generating and retrieving grounded Gemini compliance briefings.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.db.models import JobRecord, MissionAnalyticsRecord
from backend.app.schemas.report import MissionComplianceReport, GenerateReportRequest
from backend.app.services.llm_reporter import generate_mission_compliance_report

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post("/{job_id}/generate", response_model=MissionComplianceReport, status_code=status.HTTP_200_OK)
def create_mission_ai_report(
    job_id: str,
    request: Optional[GenerateReportRequest] = None,
    db: Session = Depends(get_db)
):
    """
    Generates a structured LLM mission compliance report using Gemini API and
    cross-verifies findings against spatial CV telemetry for hallucination prevention.
    """
    job = db.query(JobRecord).filter(JobRecord.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job.status != "COMPLETED":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot generate AI report while job is {job.status}. Must be COMPLETED."
        )

    model_name = request.model_name if request else "gemini-2.5-flash"
    temperature = request.temperature if request else 0.2

    try:
        report = generate_mission_compliance_report(
            job_id=job_id,
            db=db,
            model_name=model_name or "gemini-2.5-flash",
            temperature=temperature or 0.2
        )
        return report
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.get("/{job_id}", response_model=MissionComplianceReport, status_code=status.HTTP_200_OK)
def get_mission_ai_report(job_id: str, db: Session = Depends(get_db)):
    """
    Retrieves previously generated and cached AI compliance report for a mission.
    """
    analytics = db.query(MissionAnalyticsRecord).filter(MissionAnalyticsRecord.job_id == job_id).first()
    if not analytics:
        raise HTTPException(status_code=404, detail=f"Analytics not found for job: {job_id}")

    if not analytics.ai_report_json:
        raise HTTPException(
            status_code=404,
            detail=f"AI report has not yet been generated for job: {job_id}. Use POST /reports/{job_id}/generate"
        )

    try:
        report = MissionComplianceReport.model_validate_json(analytics.ai_report_json)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse stored AI report: {str(e)}")
