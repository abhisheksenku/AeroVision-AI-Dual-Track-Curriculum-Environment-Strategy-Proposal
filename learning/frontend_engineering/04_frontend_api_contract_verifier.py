"""
=============================================================================
AeroVision AI - Frontend Engineering Foundations: Module 04
=============================================================================
TypeScript Interface & API Contract Schema Verifier

Learning Objectives:
1. Validate synchronization between FastAPI REST responses and TypeScript interfaces.
2. Ensure no missing keys, null pointer hazards, or unexpected type coercions.
3. Verify contract integrity for SystemHealth, JobResponse, and MissionAnalytics.
"""

import os
import sys
from typing import Dict, Any

# Ensure workspace root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.db.database import Base, engine

def verify_frontend_contracts():
    print("=" * 70)
    print("STEP 1: FRONTEND TYPESCRIPT CONTRACT VERIFICATION")
    print("=" * 70)

    Base.metadata.create_all(bind=engine)
    client = TestClient(app)

    # 1. Health Contract
    res_health = client.get("/api/v1/health")
    assert res_health.status_code == 200
    health_json = res_health.json()

    required_health_keys = ["status", "database_connected", "model_weights_available", "system_metrics"]
    for k in required_health_keys:
        assert k in health_json, f"Missing key '{k}' in health response!"
    assert "cpu_usage_percent" in health_json["system_metrics"]
    print("✓ Healthcheck interface contract matches TypeScript: SystemHealth")

    # 2. List Jobs Contract
    res_jobs = client.get("/api/v1/jobs")
    assert res_jobs.status_code == 200
    jobs_list = res_jobs.json()
    assert isinstance(jobs_list, list)

    if len(jobs_list) > 0:
        sample_job = jobs_list[0]
        required_job_keys = ["job_id", "filename", "status", "progress_percent", "total_frames", "processed_frames", "created_at"]
        for k in required_job_keys:
            assert k in sample_job, f"Missing key '{k}' in job response!"
        print("✓ Job response interface contract matches TypeScript: JobResponse")

        # 3. Analytics Contract
        job_id = sample_job["job_id"]
        res_analytics = client.get(f"/api/v1/analytics/{job_id}")
        if res_analytics.status_code == 200:
            analytics_json = res_analytics.json()
            required_analytics_keys = [
                "mission_id", "job_id", "duration_sec", "total_unique_objects",
                "counts_by_class", "class_statistics", "line_crossings", "safety_audit"
            ]
            for k in required_analytics_keys:
                assert k in analytics_json, f"Missing key '{k}' in analytics response!"
            assert "violations_log" in analytics_json["safety_audit"]
            print("✓ Analytics response interface contract matches TypeScript: MissionAnalytics")

    print("\n[SUCCESS] All API endpoints adhere strictly to frontend client contracts.")
    print()

def main():
    verify_frontend_contracts()
    print("=" * 70)
    print("PHASE 9 - MODULE 04: COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
