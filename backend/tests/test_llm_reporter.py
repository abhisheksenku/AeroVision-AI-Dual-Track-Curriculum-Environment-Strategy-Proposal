"""
=============================================================================
AeroVision AI - LLM Reporter & Grounding Integration Tests
=============================================================================
Verifies structured compliance report generation, Gemini fallbacks,
mathematical grounding score evaluation, and RESTful API endpoints.
"""

import os
import sys
import json

# Ensure workspace root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sqlalchemy import text
from fastapi import HTTPException
from backend.app.main import app
from backend.app.db.database import Base, engine, SessionLocal
from backend.app.db.models import JobRecord, MissionAnalyticsRecord, ViolationEventRecord
from backend.app.services.llm_reporter import generate_mission_compliance_report
from backend.app.api.v1.reports import create_mission_ai_report, get_mission_ai_report
from backend.app.schemas.report import GenerateReportRequest


def setup_module():
    Base.metadata.create_all(bind=engine)


def test_llm_report_generation_and_api():
    print("=" * 70)
    print("TEST: LLM Mission Compliance Reporter & API Endpoints")
    print("=" * 70)

    db = SessionLocal()

    # 1. Create a dummy completed job with analytics and violations in SQLite
    test_job_id = "job_test_llm_01"
    existing = db.query(JobRecord).filter(JobRecord.job_id == test_job_id).first()
    if existing:
        db.delete(existing)
        db.commit()

    job = JobRecord(
        job_id=test_job_id,
        filename="recon_test.mp4",
        file_path="data/uploads/recon_test.mp4",
        status="COMPLETED",
        progress_percent=100,
        total_frames=120,
        processed_frames=120
    )
    db.add(job)
    db.commit()

    analytics = MissionAnalyticsRecord(
        job_id=test_job_id,
        mission_id="MS-TEST-001",
        duration_sec=4.0,
        total_unique_objects=5,
        counts_by_class_json=json.dumps({"car": 3, "truck": 2}),
        class_stats_json=json.dumps({
            "car": {"count": 3, "mean_speed_kmh": 45.2, "max_speed_kmh": 62.1},
            "truck": {"count": 2, "mean_speed_kmh": 32.0, "max_speed_kmh": 38.5}
        }),
        line_crossings_json=json.dumps({"incoming": 2, "outgoing": 1}),
        total_violations=1,
        compliance_status="AUDIT_FLAGGED"
    )
    db.add(analytics)
    db.commit()

    violation = ViolationEventRecord(
        job_id=test_job_id,
        event_id=1,
        rule_type="SPEED_LIMIT",
        track_id=3,
        class_name="car",
        timestamp_sec=2.4,
        severity="CRITICAL",
        details="Velocity 62.1 km/h exceeded speed limit 50.0 km/h",
        location_x=250.0,
        location_y=320.0
    )
    db.add(violation)
    db.commit()

    # 2. Test Direct Service Call
    report = generate_mission_compliance_report(test_job_id, db=db)
    print("✓ Report generated successfully via service.")
    print("  Mission ID       :", report.mission_id)
    print("  Threat Level     :", report.threat_level)
    print("  Peak Speed       :", report.peak_speed_kmh, "km/h")
    print("  Grounding Score  :", report.grounding_audit.grounding_score)
    print("  Audit Verdict    :", report.grounding_audit.verdict)
    print("  Recommendations  :", report.operational_recommendations)

    assert report.mission_id == "MS-TEST-001"
    assert report.job_id == test_job_id
    assert report.threat_level == "CRITICAL"
    assert report.peak_speed_kmh == 62.1
    assert report.grounding_audit.grounding_score >= 0.90
    assert report.grounding_audit.verdict == "VERIFIED_GROUNDED"
    assert len(report.incidents) == 1
    assert report.incidents[0].track_id == 3

    # 3. Test REST Route Handler: GET Cached Report
    cached_report = get_mission_ai_report(job_id=test_job_id, db=db)
    assert cached_report.job_id == test_job_id
    assert cached_report.grounding_audit.verdict == "VERIFIED_GROUNDED"
    print("✓ Route Handler: GET /api/v1/reports/{job_id} returned cached report.")

    # 4. Test REST Route Handler: POST Generate Report
    gen_req = GenerateReportRequest(model_name="gemini-2.5-flash")
    post_report = create_mission_ai_report(job_id=test_job_id, request=gen_req, db=db)
    assert post_report.mission_id == "MS-TEST-001"
    print("✓ Route Handler: POST /api/v1/reports/{job_id}/generate returned valid report.")

    # 5. Test Error Handling: 404 for nonexistent job
    try:
        get_mission_ai_report(job_id="job_nonexistent_xyz", db=db)
        assert False, "Expected HTTPException 404"
    except HTTPException as e:
        assert e.status_code == 404
        print("✓ Route Handler: 404 handled cleanly for non-existent job.")

    db.close()
    print("\n[SUCCESS] All LLM Reporter tests passed!")


if __name__ == "__main__":
    setup_module()
    test_llm_report_generation_and_api()
