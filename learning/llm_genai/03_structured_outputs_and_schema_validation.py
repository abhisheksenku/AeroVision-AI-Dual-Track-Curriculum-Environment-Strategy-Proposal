"""
=============================================================================
AeroVision AI - LLM & Generative AI Foundations: Module 03
=============================================================================
Structured Outputs & Pydantic Schema Enforcement with Gemini API

Learning Objectives:
1. Define strict type-safe Pydantic models for automated AI flight auditing.
2. Utilize Gemini's structured output capability (response_schema and response_mime_type).
3. Validate and parse JSON responses directly into Pydantic models without brittle regex.
4. Implement resilience mechanisms to handle partial JSON or schema validation exceptions.
"""

import json
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ValidationError


class IncidentItem(BaseModel):
    track_id: int = Field(description="Unique tracking ID of the vehicle or pedestrian")
    class_name: str = Field(description="Detected object classification")
    rule_type: str = Field(description="Regulatory safety rule violated")
    severity: Literal["WARNING", "CRITICAL"] = Field(description="Assessed severity level")
    narrative: str = Field(description="Concise description of the observed event and risk")


class MissionComplianceReport(BaseModel):
    mission_id: str = Field(description="Unique flight mission identifier")
    executive_summary: str = Field(description="High-level operational briefing of the flight")
    threat_level: Literal["LOW", "ELEVATED", "CRITICAL"] = Field(description="Overall mission risk classification")
    compliance_status: Literal["COMPLIANT", "AUDIT_FLAGGED"] = Field(description="Official safety audit verdict")
    detected_entities_count: int = Field(description="Total number of tracked objects")
    peak_speed_kmh: float = Field(description="Highest velocity observed in the mission")
    incident_breakdown: List[IncidentItem] = Field(default_factory=list, description="List of discrete safety incidents")
    operational_recommendations: List[str] = Field(default_factory=list, description="Actionable countermeasures")


def parse_and_validate_report(raw_json_str: str) -> MissionComplianceReport:
    """
    Parses and validates raw JSON string into MissionComplianceReport Pydantic model.
    """
    try:
        report = MissionComplianceReport.model_validate_json(raw_json_str)
        return report
    except ValidationError as e:
        print(f"[ERROR] Schema validation error: {e}")
        raise


def test_structured_outputs():
    print("=" * 70)
    print("STEP 1: DEFINING PYDANTIC SCHEMA SPECIFICATION")
    print("=" * 70)

    schema = MissionComplianceReport.model_json_schema()
    print("Generated JSON Schema Title:", schema.get("title"))
    print("Required Properties        :", schema.get("required"))

    print("\n" + "=" * 70)
    print("STEP 2: SIMULATING GEMINI STRUCTURED JSON RESPONSE")
    print("=" * 70)

    simulated_gemini_json = json.dumps({
        "mission_id": "MS-992",
        "executive_summary": "Surveillance flight over Industrial Sector 4 revealed stable traffic with one critical zone intrusion.",
        "threat_level": "ELEVATED",
        "compliance_status": "AUDIT_FLAGGED",
        "detected_entities_count": 8,
        "peak_speed_kmh": 48.7,
        "incident_breakdown": [
            {
                "track_id": 4,
                "class_name": "truck",
                "rule_type": "RESTRICTED_ZONE_INTRUSION",
                "severity": "CRITICAL",
                "narrative": "Commercial heavy vehicle penetrated Active Loading Runway B without clearance."
            }
        ],
        "operational_recommendations": [
            "Dispatch perimeter ground patrol to escort Track #4 out of Runway B.",
            "Inspect physical electronic barrier arm at Gate 3."
        ]
    })

    print("Raw Response Length:", len(simulated_gemini_json), "chars")
    
    # Validation step
    report = parse_and_validate_report(simulated_gemini_json)

    print("\n" + "=" * 70)
    print("STEP 3: VERIFYING PARSED TYPE-SAFE OBJECT")
    print("=" * 70)
    print(f"Mission ID        : {report.mission_id}")
    print(f"Threat Level      : {report.threat_level}")
    print(f"Compliance Status : {report.compliance_status}")
    print(f"Tracked Entities  : {report.detected_entities_count}")
    print(f"Peak Speed        : {report.peak_speed_kmh} km/h")
    print(f"Incidents Logged  : {len(report.incident_breakdown)}")
    print(f"Recommendations   : {len(report.operational_recommendations)}")

    assert report.mission_id == "MS-992"
    assert report.threat_level == "ELEVATED"
    assert report.compliance_status == "AUDIT_FLAGGED"
    assert len(report.incident_breakdown) == 1
    assert report.incident_breakdown[0].track_id == 4
    assert len(report.operational_recommendations) == 2

    # Test error handling on malformed JSON
    malformed_json = '{"mission_id": "MS-ERR", "threat_level": "INVALID_LEVEL"}'
    try:
        parse_and_validate_report(malformed_json)
        assert False, "Should have raised ValidationError!"
    except ValidationError:
        print("✓ Handled invalid schema with graceful ValidationError as expected.")

    print("\n[SUCCESS] Module 03 Structured Outputs verified successfully!")


if __name__ == "__main__":
    test_structured_outputs()
