"""
=============================================================================
AeroVision AI - MLOps, Acceleration & Deployment Test Suite
=============================================================================
Validates ONNX model export, ONNX Runtime CPU inference, mathematical
Population Stability Index (PSI) drift calculation, and RESTful API endpoints.
"""

import os
import sys
import numpy as np
import onnxruntime as ort
from fastapi import HTTPException

# Ensure workspace root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.app.services.mlops_service import AeroVisionMLOpsService
from backend.app.schemas.mlops import DriftAuditRequest, ModelExportRequest
from backend.app.api.v1.mlops import (
    get_mlops_deployment_status,
    trigger_model_export,
    audit_batch_drift
)


def test_mlops_service_and_endpoints():
    print("=" * 70, flush=True)
    print("TEST: MLOps Service, ONNX Runtime & Drift Monitoring", flush=True)
    print("=" * 70, flush=True)

    # 1. Test System Status Service
    status = AeroVisionMLOpsService.get_system_status()
    print(f"Deployment Status         : {status.status}", flush=True)
    print(f"Active Execution Provider : {status.active_execution_provider}", flush=True)
    print(f"Docker Artifacts Verified : {status.docker_container_ready}", flush=True)
    print(f"CI/CD Workflow Verified   : {status.cicd_gates_verified}", flush=True)

    assert status.status == "OPERATIONAL"
    assert status.onnx_runtime_available is True
    assert status.docker_container_ready is True
    assert status.cicd_gates_verified is True

    # 2. Test Model Export to ONNX
    export_res = AeroVisionMLOpsService.export_model_to_onnx("test_drone_backbone")
    print(f"\nModel Export Result       : {export_res.status}", flush=True)
    print(f"Export Format             : {export_res.format}", flush=True)
    print(f"Exported Path             : {export_res.export_path}", flush=True)
    print(f"Serialized File Size      : {export_res.file_size_kb} KB", flush=True)

    assert export_res.status == "SUCCESS"
    assert os.path.exists(export_res.export_path)
    assert export_res.file_size_kb > 50.0

    # 3. Test ONNX Runtime Inference Session Execution
    session = ort.InferenceSession(
        export_res.export_path,
        providers=["CPUExecutionProvider"]
    )
    dummy_input = np.random.randn(1, 3, 320, 320).astype(np.float32)
    ort_out = session.run(["bbox_predictions"], {"input_tensor": dummy_input})[0]

    print(f"ONNX Inference Output     : Shape {list(ort_out.shape)}", flush=True)
    assert ort_out.shape == (1, 4)
    assert not np.isnan(ort_out).any()
    print("✓ ONNX Runtime CPU inference executed with valid bounding box coordinates.", flush=True)

    # 4. Test Data Drift: Population Stability Index (PSI)
    # 4A. In-distribution telemetry (Nominal)
    nominal_speeds = list(np.random.normal(loc=40.0, scale=7.5, size=300))
    drift_nom = AeroVisionMLOpsService.audit_telemetry_drift(nominal_speeds)
    print(f"\nNominal Telemetry PSI     : {drift_nom.psi_score} -> Verdict: {drift_nom.verdict}", flush=True)
    assert drift_nom.psi_score < 0.10
    assert drift_nom.verdict == "NO_DRIFT"

    # 4B. Drifted telemetry (High-speed anomaly batch)
    drifted_speeds = list(np.random.normal(loc=65.0, scale=10.0, size=300))
    drift_crit = AeroVisionMLOpsService.audit_telemetry_drift(drifted_speeds)
    print(f"Drifted Telemetry PSI     : {drift_crit.psi_score} -> Verdict: {drift_crit.verdict}", flush=True)
    assert drift_crit.psi_score > 0.25
    assert drift_crit.verdict == "CRITICAL_DRIFT"

    # 5. Test REST Route Handlers
    status_handler = get_mlops_deployment_status()
    assert status_handler.status == "OPERATIONAL"
    print("✓ Route Handler: GET /api/v1/mlops/status verified.", flush=True)

    export_handler = trigger_model_export(ModelExportRequest(model_name="api_test_backbone"))
    assert export_handler.status == "SUCCESS"
    print("✓ Route Handler: POST /api/v1/mlops/export verified.", flush=True)

    audit_handler = audit_batch_drift(DriftAuditRequest(batch_speeds=nominal_speeds))
    assert audit_handler.verdict == "NO_DRIFT"
    print("✓ Route Handler: POST /api/v1/mlops/drift-audit verified.", flush=True)

    # 6. Test Error Handling: 400 on empty batch
    try:
        audit_batch_drift(DriftAuditRequest(batch_speeds=[]))
        assert False, "Expected HTTPException 400"
    except HTTPException as e:
        assert e.status_code == 400
        print("✓ Route Handler: 400 handled cleanly for empty batch speeds.", flush=True)

    print("\n[SUCCESS] All MLOps & Production Deployment tests passed successfully!", flush=True)


if __name__ == "__main__":
    test_mlops_service_and_endpoints()
