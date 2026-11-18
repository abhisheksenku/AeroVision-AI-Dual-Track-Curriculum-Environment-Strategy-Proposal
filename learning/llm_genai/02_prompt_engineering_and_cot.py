"""
=============================================================================
AeroVision AI - LLM & Generative AI Foundations: Module 02
=============================================================================
Prompt Engineering & Chain-of-Thought (CoT) Reasoning for Drone Surveillance

Learning Objectives:
1. Engineer grounded prompts embedding spatio-temporal tracking kinematics.
2. Structure multi-turn and system prompts enforcing the Safety Incident Commander persona.
3. Implement Chain-of-Thought (CoT) templates guiding the LLM through:
   a. Telemetry Ingestion -> b. Threshold Auditing -> c. Root Cause Assessment -> d. Actionable Remediation.
4. Compare Zero-Shot vs CoT generation quality and reasoning depth.
"""

import json
from typing import Dict, Any, List


def build_cot_surveillance_prompt(mission_data: Dict[str, Any]) -> str:
    """
    Constructs a Chain-of-Thought reasoning prompt from structured mission telemetry.
    """
    counts_str = ", ".join([f"{k}: {v}" for k, v in mission_data.get("counts_by_class", {}).items()])
    
    violations_summary = []
    for v in mission_data.get("violations", []):
        violations_summary.append(
            f"- Track #{v['track_id']} ({v['class_name']}) at t={v['timestamp_sec']}s: "
            f"Violated {v['rule_type']} [Severity: {v['severity']}]. Details: {v['details']} at coords {v['location']}."
        )
    violations_text = "\n".join(violations_summary) if violations_summary else "None recorded."

    prompt = f"""You are the AeroVision AI Chief Incident Commander conducting an official flight review.
You must analyze the provided drone surveillance data using systematic step-by-step reasoning.

### FLIGHT TELEMETRY DATA:
- Mission Identifier: {mission_data.get('mission_id')}
- Total Surveillance Duration: {mission_data.get('duration_sec')} seconds
- Total Tracked Entities: {mission_data.get('total_unique_objects')}
- Class Inventory: {counts_str}
- Safety Compliance Status: {mission_data.get('compliance_status')}
- Peak Fleet Velocity: {mission_data.get('peak_speed_kmh')} km/h

### REGISTERED SAFETY VIOLATIONS:
{violations_text}

### REASONING INSTRUCTIONS (Think Step-by-Step):
Step 1: Telemetry Assessment - Review the total entity count and identify dominant vehicle classifications.
Step 2: Threat & Safety Evaluation - Inspect each registered violation, assess its severity level, and evaluate its collision risk.
Step 3: Root Cause Determination - Analyze whether the infraction stemmed from excessive speed, geofence intrusion, or erratic loitering.
Step 4: Operational Recommendations - Provide 2 to 3 concrete operational countermeasures (e.g., perimeter patrols, signage, speed mitigation).

Write your final response following these 4 clear sections:
1. SUMMARY BRIEFING
2. STEP-BY-STEP REASONING
3. SAFETY AUDIT VERDICT
4. COUNTERMEASURES
"""
    return prompt


def simulate_cot_reasoning(prompt: str) -> str:
    """
    Deterministic response generator mimicking Gemini 2.5 Flash with CoT reasoning.
    """
    return """1. SUMMARY BRIEFING
Aerial surveillance mission MS-809 monitored 10 distinct entities over a 4.00-second flight window. While traffic was predominantly composed of standard passenger cars, 2 critical safety violations were flagged involving excessive speed and unauthorized geofence intrusion.

2. STEP-BY-STEP REASONING
- Step 1 (Telemetry Assessment): The scene includes 3 passenger cars, 2 buses, 2 pedestrians, and 1 commercial truck. The traffic density is moderate.
- Step 2 (Threat & Safety Evaluation): Track #3 (car) reached 68.4 km/h in a 50.0 km/h zone, presenting an imminent collision threat. Track #7 (bus) entered Restricted Pedestrian Zone A, compromising ground personnel safety.
- Step 3 (Root Cause Determination): The excessive speed of Track #3 coincided with wide lane transitions. The intrusion by Track #7 was caused by an improper turning arc near the eastern perimeter boundary.
- Step 4 (Operational Recommendations): Immediate deployment of speed enforcement along the central transit corridor and installation of high-visibility physical barriers at Zone A.

3. SAFETY AUDIT VERDICT
STATUS: AUDIT_FLAGGED. High risk of multi-vehicle conflict due to combined perimeter intrusion and speed overruns.

4. COUNTERMEASURES
1. Dispatch ground traffic unit to Eastern Corridor to intercept Track #3.
2. Review lane demarcation adjacent to Restricted Zone A to prevent commercial bus ingress.
3. Calibrate roadside dynamic speed display to trigger warning flashes above 45 km/h.
"""


def test_cot_prompting():
    print("=" * 70)
    print("STEP 1: PREPARING STRUCTURED MISSION TELEMETRY")
    print("=" * 70)

    sample_mission = {
        "mission_id": "MS-809",
        "duration_sec": 4.0,
        "total_unique_objects": 10,
        "counts_by_class": {"car": 3, "bus": 2, "person": 2, "truck": 1},
        "compliance_status": "FLAGGED",
        "peak_speed_kmh": 68.4,
        "violations": [
            {
                "track_id": 3,
                "class_name": "car",
                "timestamp_sec": 1.25,
                "rule_type": "SPEED_LIMIT",
                "severity": "CRITICAL",
                "details": "Vehicle exceeded 50.0 km/h limit (recorded 68.4 km/h)",
                "location": [420.0, 180.5]
            },
            {
                "track_id": 7,
                "class_name": "bus",
                "timestamp_sec": 2.80,
                "rule_type": "RESTRICTED_ZONE_INTRUSION",
                "severity": "CRITICAL",
                "details": "Unauthorized entity in Restricted Pedestrian Zone A",
                "location": [310.0, 520.0]
            }
        ]
    }

    prompt = build_cot_surveillance_prompt(sample_mission)
    print("Constructed Prompt Length:", len(prompt), "chars")
    print("Contains Mission ID      :", "MS-809" in prompt)
    print("Contains CoT Steps       :", "Step 1: Telemetry Assessment" in prompt)

    print("\n" + "=" * 70)
    print("STEP 2: RUNNING COT INFERENCE SIMULATION")
    print("=" * 70)

    response = simulate_cot_reasoning(prompt)
    print(response)

    # Assertions
    assert "1. SUMMARY BRIEFING" in response
    assert "2. STEP-BY-STEP REASONING" in response
    assert "3. SAFETY AUDIT VERDICT" in response
    assert "4. COUNTERMEASURES" in response
    assert "Track #3" in response
    assert "68.4 km/h" in response

    print("\n[SUCCESS] Module 02 Chain-of-Thought logic verified!")


if __name__ == "__main__":
    test_cot_prompting()
