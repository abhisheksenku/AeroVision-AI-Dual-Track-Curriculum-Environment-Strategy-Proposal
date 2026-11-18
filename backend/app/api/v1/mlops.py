"""
=============================================================================
AeroVision AI - MLOps & Production Acceleration REST API
=============================================================================
Endpoints for querying ONNX Runtime status, triggering model serialization,
and computing live Population Stability Index (PSI) drift telemetry.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.mlops import (
    ModelExportRequest,
    ModelExportResponse,
    DriftAuditRequest,
    DriftAuditResponse,
    MLOpsSystemStatus
)
from backend.app.services.mlops_service import AeroVisionMLOpsService

router = APIRouter(prefix="/mlops", tags=["mlops"])


@router.get("/status", response_model=MLOpsSystemStatus, status_code=status.HTTP_200_OK)
def get_mlops_deployment_status():
    """
    Returns deployment health, active ONNX execution providers,
    exported model registries, and Docker/CI/CD readiness.
    """
    try:
        return AeroVisionMLOpsService.get_system_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MLOps status check failed: {str(e)}")


@router.post("/export", response_model=ModelExportResponse, status_code=status.HTTP_200_OK)
def trigger_model_export(request: Optional[ModelExportRequest] = None):
    """
    Triggers graph compilation and ONNX model export for edge deployment.
    """
    model_name = request.model_name if request else "drone_backbone"
    try:
        response = AeroVisionMLOpsService.export_model_to_onnx(model_name=model_name or "drone_backbone")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model export failed: {str(e)}")


@router.post("/drift-audit", response_model=DriftAuditResponse, status_code=status.HTTP_200_OK)
def audit_batch_drift(request: DriftAuditRequest):
    """
    Evaluates Population Stability Index (PSI) on observed telemetry velocities
    to detect covariate distribution drift and trigger retraining mandates.
    """
    if not request.batch_speeds or len(request.batch_speeds) < 2:
        raise HTTPException(status_code=400, detail="Batch speeds must contain at least 2 data points.")

    try:
        audit = AeroVisionMLOpsService.audit_telemetry_drift(
            batch_speeds=request.batch_speeds,
            num_bins=request.num_bins or 10
        )
        return audit
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift audit failed: {str(e)}")
