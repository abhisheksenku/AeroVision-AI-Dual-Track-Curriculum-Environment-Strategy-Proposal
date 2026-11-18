"""
=============================================================================
AeroVision AI - Autonomous Agent & LangGraph Integration Tests
=============================================================================
Validates multi-agent incident response StateGraph execution, supervisor routing,
tool calling, reflection auditing, and RESTful API endpoints.
"""

import os
import sys
import json
from fastapi import HTTPException

# Ensure workspace root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import backend.app.utils.qdrant_bootstrap
from backend.app.db.database import Base, engine, SessionLocal
from backend.app.db.models import JobRecord, MissionAnalyticsRecord, ViolationEventRecord
from backend.app.services.agent_service import AeroVisionAgentService
from backend.app.schemas.agent import AgentChatRequest
from backend.app.api.v1.agents import (
    trigger_incident_response_graph,
    operator_agent_dialogue
)


def setup_module():
    Base.metadata.create_all(bind=engine)


def test_autonomous_agent_graph_and_api():
    print("=" * 70, flush=True)
    print("TEST: Autonomous Agent Graph & Operator Telemetry Dialogue", flush=True)
    print("=" * 70, flush=True)

    db = SessionLocal()

    # 1. Create a dummy completed job with analytics and violations in SQLite
    test_job_id = "job_test_agent_01"
    existing = db.query(JobRecord).filter(JobRecord.job_id == test_job_id).first()
    if existing:
        db.delete(existing)
        db.commit()

    job = JobRecord(
        job_id=test_job_id,
        filename="perimeter_incident.mp4",
        file_path="data/uploads/perimeter_incident.mp4",
        status="COMPLETED",
        progress_percent=100,
        total_frames=150,
        processed_frames=150
    )
    db.add(job)
    db.commit()

    analytics = MissionAnalyticsRecord(
        job_id=test_job_id,
        mission_id="MS-INCIDENT-001",
        duration_sec=5.0,
        total_unique_objects=6,
        counts_by_class_json=json.dumps({"car": 4, "truck": 2}),
        class_stats_json=json.dumps({
            "car": {"count": 4, "mean_speed_kmh": 58.2, "max_speed_kmh": 72.5},
            "truck": {"count": 2, "mean_speed_kmh": 34.0, "max_speed_kmh": 40.0}
        }),
        line_crossings_json=json.dumps({"incoming": 3, "outgoing": 2}),
        total_violations=2,
        compliance_status="AUDIT_FLAGGED"
    )
    db.add(analytics)
    db.commit()

    v1 = ViolationEventRecord(
        job_id=test_job_id,
        event_id=1,
        rule_type="SPEED_LIMIT",
        track_id=3,
        class_name="car",
        timestamp_sec=2.2,
        severity="CRITICAL",
        details="Vehicle reached peak speed 72.5 km/h exceeding 50.0 km/h limit",
        location_x=220.0,
        location_y=380.0
    )
    v2 = ViolationEventRecord(
        job_id=test_job_id,
        event_id=2,
        rule_type="RESTRICTED_ZONE_INTRUSION",
        track_id=3,
        class_name="car",
        timestamp_sec=3.4,
        severity="CRITICAL",
        details="Vehicle penetrated active runway gate without clearance",
        location_x=410.0,
        location_y=510.0
    )
    db.add_all([v1, v2])
    db.commit()

    # 2. Test Direct Service Call: Multi-Agent Incident Response Graph
    result = AeroVisionAgentService.execute_incident_response(job_id=test_job_id, db=db)
    print("\n✓ Multi-Agent Graph executed successfully via Service.", flush=True)
    print(f"  Mission ID         : {result.action_plan.mission_id}", flush=True)
    print(f"  Threat Level       : {result.action_plan.threat_level}", flush=True)
    print(f"  Containment Radius : {result.action_plan.containment_radius_m}m", flush=True)
    print(f"  Grounding Verdict  : {result.action_plan.grounding_verdict}", flush=True)
    print(f"  Total Directives   : {len(result.action_plan.tactical_directives)}", flush=True)
    print(f"  Trace Steps Run    : {len(result.execution_trace)}", flush=True)

    assert result.status == "SUCCESS"
    assert result.action_plan.mission_id == "MS-INCIDENT-001"
    assert result.action_plan.threat_level == "CRITICAL"
    assert result.action_plan.containment_radius_m > 30.0
    assert result.action_plan.grounding_score == 1.0
    assert result.action_plan.grounding_verdict == "VERIFIED_GROUNDED"
    assert len(result.action_plan.tactical_directives) == 4
    assert len(result.execution_trace) >= 4

    # 3. Test Route Handler: POST /api/v1/agents/incident-response/{job_id}
    route_res = trigger_incident_response_graph(job_id=test_job_id, db=db)
    assert route_res.status == "SUCCESS"
    assert route_res.action_plan.containment_radius_m == result.action_plan.containment_radius_m
    print("✓ Route Handler: POST /api/v1/agents/incident-response/{job_id} passed.", flush=True)

    # 4. Test Operator Dialogue with Autonomous Tool Calling
    # 4A. Telemetry Inspection Tool
    chat_telem = operator_agent_dialogue(
        request=AgentChatRequest(message="What is the speed and tracking status of objects?", job_id=test_job_id),
        db=db
    )
    assert "inspect_mission_telemetry" in chat_telem.tools_invoked
    assert "telemetry" in chat_telem.data_grounding
    print("✓ Operator Chat: Successfully invoked 'inspect_mission_telemetry' tool.", flush=True)

    # 4B. Statutory Regulation Tool
    chat_legal = operator_agent_dialogue(
        request=AgentChatRequest(message="What are the FAA and OSHA safety regulations regarding this incident?", job_id=test_job_id),
        db=db
    )
    assert "query_statutory_regulations" in chat_legal.tools_invoked
    assert "citations" in chat_legal.data_grounding
    print("✓ Operator Chat: Successfully invoked 'query_statutory_regulations' tool.", flush=True)

    # 4C. Kinematic Radius Tool
    chat_radius = operator_agent_dialogue(
        request=AgentChatRequest(message="Compute the emergency containment stopping radius.", job_id=test_job_id),
        db=db
    )
    assert "calculate_containment_radius" in chat_radius.tools_invoked
    assert chat_radius.data_grounding["containment_radius_m"] > 0
    print("✓ Operator Chat: Successfully invoked 'calculate_containment_radius' tool.", flush=True)

    # 5. Test Error Handling: 404 on nonexistent job, 400 on empty chat
    try:
        trigger_incident_response_graph(job_id="job_nonexistent_xyz", db=db)
        assert False, "Expected HTTPException 404"
    except HTTPException as e:
        assert e.status_code == 404
        print("✓ Route Handler: 404 handled cleanly for non-existent job.", flush=True)

    try:
        operator_agent_dialogue(request=AgentChatRequest(message=""), db=db)
        assert False, "Expected HTTPException 400"
    except HTTPException as e:
        assert e.status_code == 400
        print("✓ Route Handler: 400 handled cleanly for empty message.", flush=True)

    db.close()
    print("\n[SUCCESS] All Autonomous Agent & LangGraph tests passed!", flush=True)


if __name__ == "__main__":
    setup_module()
    test_autonomous_agent_graph_and_api()
