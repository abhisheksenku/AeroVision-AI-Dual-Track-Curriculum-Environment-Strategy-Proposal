"""
=============================================================================
AeroVision AI - REST API Endpoints Integration Test Suite
=============================================================================
Comprehensive automated integration tests validating:
1. Healthcheck diagnostics endpoint (DB, model weights, CPU/RAM metrics).
2. Video upload endpoint (multipart/form-data, validation, 202 Accepted).
3. Background task CV pipeline execution and database persistence.
4. Job status polling (QUEUED -> PROCESSING -> COMPLETED).
5. Analytics report query endpoint and safety violations filtering.
6. Error handling (400 Bad Request on invalid files, 404 Not Found).
"""

import os
import sys
import io
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.db.database import Base, engine

SAMPLE_VIDEO_PATH = "data/samples/drone_flight_sample.mp4"

def setup_module():
    """Initializes tables before running tests."""
    Base.metadata.create_all(bind=engine)

def test_root_and_health_endpoints():
    """Validates root welcome and healthcheck diagnostics."""
    print("\n--- TEST: Health & Root Endpoints ---")
    client = TestClient(app)

    # Root
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert res_root.json()["platform"] == "AeroVision AI"
    assert res_root.json()["status"] == "OPERATIONAL"

    # Healthcheck
    res_health = client.get("/api/v1/health")
    print(f"Health Response: {res_health.json()}")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "HEALTHY"
    assert res_health.json()["database_connected"] is True
    assert res_health.json()["model_weights_available"] is True
    assert "cpu_usage_percent" in res_health.json()["system_metrics"]
    print("✓ Healthcheck diagnostics verified.")

def test_job_upload_and_analytics_lifecycle():
    """
    Validates the complete end-to-end API lifecycle:
    Upload Video (202 Accepted) -> Background CV Pipeline -> Status (COMPLETED) -> Query Analytics
    """
    print("\n--- TEST: Full Video Job & Analytics API Lifecycle ---")
    assert os.path.exists(SAMPLE_VIDEO_PATH), f"Sample video missing: {SAMPLE_VIDEO_PATH}"
    client = TestClient(app)

    # 1. Upload Video
    with open(SAMPLE_VIDEO_PATH, "rb") as video_file:
        files = {"video": ("drone_flight_sample.mp4", video_file, "video/mp4")}
        t0 = time.perf_counter()
        upload_res = client.post("/api/v1/jobs/upload", files=files)
        upload_time_s = time.perf_counter() - t0

    print(f"POST /api/v1/jobs/upload -> Status {upload_res.status_code}")
    assert upload_res.status_code == 202
    job_data = upload_res.json()
    job_id = job_data["job_id"]
    print(f"Dispatched Job ID: {job_id} (Status: {job_data['status']})")
    print(f"Execution Duration in TestClient: {upload_time_s:.2f} s")

    # 2. Check Job Status Post-Execution
    status_res = client.get(f"/api/v1/jobs/{job_id}")
    print(f"GET /api/v1/jobs/{job_id} -> Status {status_res.status_code}")
    assert status_res.status_code == 200
    status_data = status_res.json()
    print(f"Job State: {status_data['status']} ({status_data['processed_frames']}/{status_data['total_frames']} frames)")

    assert status_data["status"] == "COMPLETED"
    assert status_data["progress_percent"] == 100
    assert status_data["processed_frames"] == 120
    assert status_data["annotated_video_path"] is not None
    assert os.path.exists(status_data["annotated_video_path"])

    # 3. Query Analytics Summary
    analytics_res = client.get(f"/api/v1/analytics/{job_id}")
    print(f"GET /api/v1/analytics/{job_id} -> Status {analytics_res.status_code}")
    assert analytics_res.status_code == 200
    analytics_data = analytics_res.json()

    print(f"Total Unique Objects : {analytics_data['total_unique_objects']}")
    print(f"Class Inventory      : {analytics_data['counts_by_class']}")
    print(f"Safety Compliance    : {analytics_data['safety_audit']['compliance_status']}")
    print(f"Total Violations     : {analytics_data['safety_audit']['total_violations']}")

    assert analytics_data["job_id"] == job_id
    assert analytics_data["total_unique_objects"] > 0
    assert "class_statistics" in analytics_data

    # 4. Query Violations Filtered
    violations_res = client.get(f"/api/v1/analytics/{job_id}/violations")
    assert violations_res.status_code == 200
    all_violations = violations_res.json()
    print(f"Retrieved {len(all_violations)} violation records from database.")
    assert len(all_violations) == analytics_data["safety_audit"]["total_violations"]

    # 5. List Jobs
    list_res = client.get("/api/v1/jobs")
    assert list_res.status_code == 200
    assert len(list_res.json()) >= 1
    print("✓ Full video job and analytics lifecycle passed 100%!")

def test_api_error_handling():
    """Validates 400 and 404 error responses."""
    print("\n--- TEST: API Error Responses ---")
    client = TestClient(app)

    # Invalid extension
    bad_file = {"video": ("data.txt", io.BytesIO(b"Hello world"), "text/plain")}
    res_bad = client.post("/api/v1/jobs/upload", files=bad_file)
    assert res_bad.status_code == 400
    assert "Unsupported video extension" in res_bad.json()["detail"]

    # Not found job
    res_404 = client.get("/api/v1/jobs/non_existent_job_id")
    assert res_404.status_code == 404

    # Analytics for non-existent job
    res_analytics_404 = client.get("/api/v1/analytics/non_existent_job_id")
    assert res_analytics_404.status_code == 404
    print("✓ Error handling verified.")

if __name__ == "__main__":
    setup_module()
    test_root_and_health_endpoints()
    test_job_upload_and_analytics_lifecycle()
    test_api_error_handling()
    print("\n" + "=" * 70)
    print("ALL PHASE 8 TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
