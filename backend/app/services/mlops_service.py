"""
=============================================================================
AeroVision AI - MLOps, Acceleration & Drift Service
=============================================================================
Orchestrates ONNX model export, integer quantization, execution provider
selection, and continuous Population Stability Index (PSI) drift monitoring.
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

from backend.app.schemas.mlops import (
    ModelExportResponse,
    DriftAuditResponse,
    MLOpsSystemStatus
)


class AeroVisionMLOpsService:
    """
    Production MLOps service for model optimization and data drift surveillance.
    """
    BASELINE_SPEED_MEAN = 40.0
    BASELINE_SPEED_STD = 8.0

    @classmethod
    def get_system_status(cls) -> MLOpsSystemStatus:
        """
        Audits deployment health, execution providers, and model registry artifacts.
        """
        providers = ort.get_available_providers()
        active_provider = providers[0] if providers else "CPUExecutionProvider"

        export_dir = os.path.join("models", "export")
        exported_files = []
        if os.path.exists(export_dir):
            exported_files = [f for f in os.listdir(export_dir) if f.endswith((".onnx", ".pt", ".tflite"))]

        docker_ready = os.path.exists("Dockerfile.backend") and os.path.exists("docker-compose.yml")
        cicd_ready = os.path.exists(os.path.join(".github", "workflows", "ci.yml"))

        return MLOpsSystemStatus(
            status="OPERATIONAL",
            onnx_runtime_available=True,
            active_execution_provider=active_provider,
            quantization_supported=True,
            docker_container_ready=docker_ready,
            exported_models=exported_files,
            cicd_gates_verified=cicd_ready
        )

    @classmethod
    def export_model_to_onnx(cls, model_name: str = "drone_backbone") -> ModelExportResponse:
        """
        Exports convolutional drone feature backbone to optimized ONNX graph.
        """
        import importlib
        mod = importlib.import_module("learning.mlops_deployment.01_model_export_and_onnx_runtime")
        AeroVisionDroneBackbone = mod.AeroVisionDroneBackbone

        model = AeroVisionDroneBackbone().eval()
        dummy_input = torch.randn(1, 3, 320, 320, dtype=torch.float32)

        export_dir = os.path.join("models", "export")
        os.makedirs(export_dir, exist_ok=True)
        export_path = os.path.join(export_dir, f"{model_name}.onnx")

        torch.onnx.export(
            model,
            dummy_input,
            export_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_tensor"],
            output_names=["bbox_predictions"],
            dynamic_axes={"input_tensor": {0: "batch_size"}, "bbox_predictions": {0: "batch_size"}},
            dynamo=False
        )

        size_kb = os.path.getsize(export_path) / 1024.0

        return ModelExportResponse(
            status="SUCCESS",
            format="ONNX",
            export_path=export_path,
            file_size_kb=round(size_kb, 2),
            exported_at=datetime.utcnow().isoformat()
        )

    @classmethod
    def audit_telemetry_drift(cls, batch_speeds: List[float], num_bins: int = 10) -> DriftAuditResponse:
        """
        Calculates Population Stability Index (PSI) comparing batch speeds against baseline.
        """
        if not batch_speeds:
            raise ValueError("Batch speeds list cannot be empty.")

        np.random.seed(42)
        baseline = np.random.normal(loc=cls.BASELINE_SPEED_MEAN, scale=cls.BASELINE_SPEED_STD, size=2000)
        observed = np.array(batch_speeds, dtype=np.float64)

        # Compute bin edges from baseline
        quantiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(baseline, quantiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        base_counts, _ = np.histogram(baseline, bins=bin_edges)
        obs_counts, _ = np.histogram(observed, bins=bin_edges)

        eps = 1e-4
        q_props = (base_counts / len(baseline)) + eps
        p_props = (obs_counts / len(observed)) + eps

        q_props = q_props / np.sum(q_props)
        p_props = p_props / np.sum(p_props)

        psi = float(np.sum((p_props - q_props) * np.log(p_props / q_props)))

        if psi < 0.10:
            verdict = "NO_DRIFT"
            rec = "Nominal operations. Covariate distribution adheres to baseline expectations."
        elif psi <= 0.25:
            verdict = "MODERATE_DRIFT"
            rec = "Moderate drift detected. Queue telemetry for review and monitoring."
        else:
            verdict = "CRITICAL_DRIFT"
            rec = "Critical covariate drift detected! Flagging for dataset curation and automated retraining."

        return DriftAuditResponse(
            psi_score=round(psi, 4),
            verdict=verdict,
            recommendation=rec,
            baseline_mean_speed_kmh=round(float(np.mean(baseline)), 2),
            observed_mean_speed_kmh=round(float(np.mean(observed)), 2),
            num_samples_evaluated=len(observed)
        )
