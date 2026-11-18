"""
=============================================================================
AeroVision AI - MLOps & Production Deployment Schemas
=============================================================================
Pydantic contracts for model export, ONNX runtime acceleration,
covariate data drift auditing (PSI), and container health metrics.
"""

from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field


class ModelExportRequest(BaseModel):
    model_name: Optional[str] = "drone_backbone"
    target_format: Literal["ONNX", "INT8", "TORCHSCRIPT"] = "ONNX"
    dynamic_batch: bool = True


class ModelExportResponse(BaseModel):
    status: Literal["SUCCESS", "FAILED"]
    format: str
    export_path: str
    file_size_kb: float
    exported_at: str


class DriftAuditRequest(BaseModel):
    batch_speeds: List[float] = Field(description="Observed velocity points from live telemetry batch")
    num_bins: Optional[int] = Field(default=10, description="Quantile bins for PSI evaluation")


class DriftAuditResponse(BaseModel):
    psi_score: float
    verdict: Literal["NO_DRIFT", "MODERATE_DRIFT", "CRITICAL_DRIFT"]
    recommendation: str
    baseline_mean_speed_kmh: float
    observed_mean_speed_kmh: float
    num_samples_evaluated: int


class MLOpsSystemStatus(BaseModel):
    status: Literal["OPERATIONAL", "DEGRADED"]
    onnx_runtime_available: bool
    active_execution_provider: str
    quantization_supported: bool
    docker_container_ready: bool
    exported_models: List[str]
    cicd_gates_verified: bool
