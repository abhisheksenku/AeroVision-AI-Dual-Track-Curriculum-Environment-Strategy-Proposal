"""
=============================================================================
AeroVision AI - LLM Autonomous Mission Reporter Service
=============================================================================
Orchestrates Gemini Generative AI reporting, structured JSON extraction,
and deterministic multimodal grounding audits against CV telemetry.
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session

from backend.app.db.models import JobRecord, MissionAnalyticsRecord, ViolationEventRecord
from backend.app.schemas.report import (
    MissionComplianceReport,
    IncidentDetail,
    GroundingAuditResult
)

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


def audit_report_grounding(report_data: Dict[str, Any], ground_truth: Dict[str, Any]) -> GroundingAuditResult:
    """
    Computes mathematical Grounding Fidelity score cross-verifying LLM outputs
    against OpenCV/ByteTrack ground-truth kinematics.
    """
    gt_classes = set(ground_truth.get("counts_by_class", {}).keys())
    gt_violations = ground_truth.get("violations", [])
    gt_violation_tracks = {v.track_id for v in gt_violations}
    gt_rule_types = {v.rule_type for v in gt_violations}
    gt_peak_speed = ground_truth.get("peak_speed_kmh", 0.0)

    # 1. Entity consistency
    reported_incidents = report_data.get("incidents", [])
    reported_classes = {inc.get("class_name", "").lower() for inc in reported_incidents}
    
    if reported_classes:
        hallucinated = reported_classes - gt_classes
        entity_score = 1.0 - (len(hallucinated) / len(reported_classes))
    else:
        hallucinated = set()
        entity_score = 1.0

    # 2. Numerical speed score
    reported_speed = report_data.get("peak_speed_kmh", 0.0)
    speed_delta = abs(reported_speed - gt_peak_speed)
    if speed_delta <= 1.5:
        speed_score = 1.0
    else:
        speed_score = max(0.0, 1.0 - (speed_delta / max(gt_peak_speed, 1.0)))

    # 3. Incident match rate
    if reported_incidents:
        valid_count = sum(
            1 for inc in reported_incidents
            if inc.get("track_id") in gt_violation_tracks and inc.get("rule_type") in gt_rule_types
        )
        incident_score = valid_count / len(reported_incidents)
    else:
        incident_score = 1.0 if len(gt_violations) == 0 else 0.5

    composite_score = round(
        (0.35 * entity_score) + (0.35 * speed_score) + (0.30 * incident_score),
        3
    )

    if composite_score >= 0.90:
        verdict = "VERIFIED_GROUNDED"
    elif composite_score >= 0.70:
        verdict = "MINOR_DEVIATION"
    else:
        verdict = "UNRELIABLE_HALLUCINATION"

    return GroundingAuditResult(
        grounding_score=composite_score,
        verdict=verdict,
        entity_score=round(entity_score, 2),
        speed_score=round(speed_score, 2),
        incident_score=round(incident_score, 2),
        hallucinated_classes=list(hallucinated),
        speed_delta_kmh=round(speed_delta, 2)
    )


def generate_mission_compliance_report(
    job_id: str,
    db: Session,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.2
) -> MissionComplianceReport:
    """
    Generates a structured compliance report using Gemini API with deterministic grounding.
    """
    analytics = db.query(MissionAnalyticsRecord).filter(MissionAnalyticsRecord.job_id == job_id).first()
    if not analytics:
        raise ValueError(f"No analytics record found for job {job_id}")

    violations = db.query(ViolationEventRecord).filter(ViolationEventRecord.job_id == job_id).all()
    counts_by_class = json.loads(analytics.counts_by_class_json or "{}")
    class_stats = json.loads(analytics.class_stats_json or "{}")

    # Compute actual peak speed across all vehicles
    peak_speed = 0.0
    for stat in class_stats.values():
        if stat.get("max_speed_kmh", 0) > peak_speed:
            peak_speed = stat.get("max_speed_kmh")

    ground_truth = {
        "mission_id": analytics.mission_id,
        "counts_by_class": counts_by_class,
        "total_unique_objects": analytics.total_unique_objects,
        "duration_sec": analytics.duration_sec,
        "compliance_status": analytics.compliance_status,
        "peak_speed_kmh": round(peak_speed, 1),
        "violations": violations
    }

    api_key = os.environ.get("GEMINI_API_KEY")
    report_dict: Optional[Dict[str, Any]] = None
    used_model = model_name

    # Check if Gemini live generation is possible
    if api_key and GENAI_AVAILABLE:
        try:
            client = genai.Client(api_key=api_key)
            system_instruction = (
                "You are the AeroVision AI Chief Flight Safety Officer. "
                "Analyze the provided drone surveillance telemetry and generate a strictly grounded, "
                "high-precision compliance report in JSON. Do not hallucinate classes, speeds, or incidents."
            )
            violations_text = "\n".join([
                f"- Track #{v.track_id} ({v.class_name}) at {v.timestamp_sec:.1f}s: {v.rule_type} [{v.severity}]. {v.details}"
                for v in violations
            ]) or "None"

            prompt = (
                f"Flight Telemetry:\n"
                f"- Mission ID: {analytics.mission_id}\n"
                f"- Duration: {analytics.duration_sec:.1f}s\n"
                f"- Tracked Entities: {analytics.total_unique_objects} ({counts_by_class})\n"
                f"- Peak Observed Speed: {peak_speed:.1f} km/h\n"
                f"- Compliance Status: {analytics.compliance_status}\n\n"
                f"Registered Infractions:\n{violations_text}\n"
            )

            config = types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
                system_instruction=system_instruction
            )
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config
            )
            if resp.text:
                report_dict = json.loads(resp.text)
        except Exception as e:
            print(f"[WARN] Gemini live generation failed, falling back to deterministic reporter: {e}")
            report_dict = None

    # Deterministic Grounded Generator (fallback when offline or key missing)
    if not report_dict:
        used_model = f"{model_name} (Deterministic Engine)"
        threat_level = "CRITICAL" if any(v.severity == "CRITICAL" for v in violations) else (
            "ELEVATED" if len(violations) > 0 else "LOW"
        )
        
        incidents = [
            {
                "track_id": v.track_id,
                "class_name": v.class_name,
                "rule_type": v.rule_type,
                "severity": v.severity,
                "narrative": f"Vehicle #{v.track_id} ({v.class_name}) triggered {v.rule_type} at t={v.timestamp_sec:.1f}s: {v.details}"
            }
            for v in violations
        ]

        recommendations = []
        if any(v.rule_type == "SPEED_LIMIT" for v in violations):
            recommendations.append("Deploy dynamic radar speed displays along high-velocity flight corridors.")
        if any(v.rule_type == "RESTRICTED_ZONE_INTRUSION" for v in violations):
            recommendations.append("Reinforce physical perimeter barriers and review automated geofence buffer boundaries.")
        if not recommendations:
            recommendations.append("Continue standard routine aerial monitoring. All sectors nominal.")

        executive_summary = (
            f"Autonomous flight analysis for Mission {analytics.mission_id} tracked {analytics.total_unique_objects} "
            f"entities over {analytics.duration_sec:.1f} seconds. Safety status: {analytics.compliance_status} "
            f"with {len(violations)} registered infractions and a peak velocity of {peak_speed:.1f} km/h."
        )

        report_dict = {
            "mission_id": analytics.mission_id,
            "job_id": job_id,
            "executive_summary": executive_summary,
            "threat_level": threat_level,
            "compliance_status": analytics.compliance_status,
            "total_tracked_objects": analytics.total_unique_objects,
            "peak_speed_kmh": round(peak_speed, 1),
            "incidents": incidents,
            "operational_recommendations": recommendations
        }

    # Audit grounding against actual SQLite data
    grounding_result = audit_report_grounding(report_dict, ground_truth)

    incident_objects = [
        IncidentDetail(
            track_id=inc["track_id"],
            class_name=inc["class_name"],
            rule_type=inc["rule_type"],
            severity=inc["severity"],
            narrative=inc["narrative"]
        )
        for inc in report_dict.get("incidents", [])
    ]

    final_report = MissionComplianceReport(
        mission_id=analytics.mission_id,
        job_id=job_id,
        executive_summary=report_dict["executive_summary"],
        threat_level=report_dict.get("threat_level", "LOW"),
        compliance_status=report_dict.get("compliance_status", analytics.compliance_status),
        total_tracked_objects=analytics.total_unique_objects,
        peak_speed_kmh=round(peak_speed, 1),
        incidents=incident_objects,
        operational_recommendations=report_dict.get("operational_recommendations", []),
        grounding_audit=grounding_result,
        model_used=used_model,
        generated_at=datetime.now(timezone.utc).isoformat()
    )

    # Persist report to database
    analytics.ai_report_json = final_report.model_dump_json()
    db.commit()

    return final_report
