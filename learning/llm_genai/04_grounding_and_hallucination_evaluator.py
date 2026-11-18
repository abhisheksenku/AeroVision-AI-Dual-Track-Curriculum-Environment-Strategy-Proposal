"""
=============================================================================
AeroVision AI - LLM & Generative AI Foundations: Module 04
=============================================================================
Multimodal Grounding & Hallucination Verification Engine

Learning Objectives:
1. Formulate deterministic verification metrics to cross-audit LLM reports against CV telemetry.
2. Quantify Entity Grounding, Numerical Fidelity, and Incident Consistency.
3. Compute composite Grounding Fidelity Score and issue automated audit verdicts.
4. Protect production safety pipelines against LLM hallucinations.
"""

from typing import Dict, Any, List, Tuple


class GroundingAuditor:
    """
    Cross-checks an LLM compliance report against ground-truth CV telemetry.
    """
    def __init__(self, ground_truth: Dict[str, Any]):
        self.gt_entities = set(ground_truth.get("counts_by_class", {}).keys())
        self.gt_total_objects = ground_truth.get("total_unique_objects", 0)
        self.gt_peak_speed = ground_truth.get("peak_speed_kmh", 0.0)
        self.gt_violations = ground_truth.get("violations", [])
        self.gt_violation_tracks = {v["track_id"] for v in self.gt_violations}
        self.gt_rule_types = {v["rule_type"] for v in self.gt_violations}

    def evaluate_report(self, report_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates fidelity metrics and returns a structured audit verification result.
        """
        # 1. Entity Consistency
        reported_incidents = report_dict.get("incident_breakdown", [])
        reported_classes = {inc.get("class_name", "").lower() for inc in reported_incidents}
        
        if reported_classes:
            hallucinated_classes = reported_classes - self.gt_entities
            entity_score = 1.0 - (len(hallucinated_classes) / len(reported_classes))
        else:
            hallucinated_classes = set()
            entity_score = 1.0

        # 2. Numerical Speed Fidelity
        reported_peak_speed = report_dict.get("peak_speed_kmh", 0.0)
        speed_delta = abs(reported_peak_speed - self.gt_peak_speed)
        # Tolerance: up to 1.5 km/h discrepancy allowed due to rounding
        if speed_delta <= 1.5:
            speed_score = 1.0
        else:
            speed_score = max(0.0, 1.0 - (speed_delta / max(self.gt_peak_speed, 1.0)))

        # 3. Incident Match Rate
        if reported_incidents:
            valid_incidents = 0
            for inc in reported_incidents:
                t_id = inc.get("track_id")
                rule = inc.get("rule_type")
                if t_id in self.gt_violation_tracks and rule in self.gt_rule_types:
                    valid_incidents += 1
            incident_score = valid_incidents / len(reported_incidents)
        else:
            incident_score = 1.0 if len(self.gt_violations) == 0 else 0.5

        # 4. Composite Fidelity Score: Weighted combination
        composite_score = round(
            (0.35 * entity_score) + (0.35 * speed_score) + (0.30 * incident_score),
            3
        )

        # 5. Verdict Classification
        if composite_score >= 0.90:
            verdict = "VERIFIED_GROUNDED"
        elif composite_score >= 0.70:
            verdict = "MINOR_DEVIATION"
        else:
            verdict = "UNRELIABLE_HALLUCINATION"

        return {
            "grounding_score": composite_score,
            "verdict": verdict,
            "entity_score": round(entity_score, 2),
            "speed_score": round(speed_score, 2),
            "incident_score": round(incident_score, 2),
            "hallucinated_classes": list(hallucinated_classes),
            "speed_delta_kmh": round(speed_delta, 2)
        }


def test_grounding_evaluator():
    print("=" * 70)
    print("STEP 1: INITIALIZING GROUND TRUTH CV TELEMETRY")
    print("=" * 70)

    ground_truth = {
        "mission_id": "MS-GT-100",
        "total_unique_objects": 10,
        "counts_by_class": {"car": 4, "truck": 2, "bus": 1},
        "peak_speed_kmh": 54.2,
        "violations": [
            {"track_id": 2, "class_name": "car", "rule_type": "SPEED_LIMIT"},
            {"track_id": 5, "class_name": "truck", "rule_type": "RESTRICTED_ZONE_INTRUSION"}
        ]
    }
    auditor = GroundingAuditor(ground_truth)

    print("\n" + "=" * 70)
    print("STEP 2: AUDITING FULLY GROUNDED AI REPORT")
    print("=" * 70)

    grounded_report = {
        "mission_id": "MS-GT-100",
        "peak_speed_kmh": 54.2,
        "incident_breakdown": [
            {"track_id": 2, "class_name": "car", "rule_type": "SPEED_LIMIT"},
            {"track_id": 5, "class_name": "truck", "rule_type": "RESTRICTED_ZONE_INTRUSION"}
        ]
    }
    audit_res_1 = auditor.evaluate_report(grounded_report)
    print("Audit Result (Grounded):", audit_res_1)
    assert audit_res_1["verdict"] == "VERIFIED_GROUNDED"
    assert audit_res_1["grounding_score"] == 1.0

    print("\n" + "=" * 70)
    print("STEP 3: AUDITING HALLUCINATED AI REPORT")
    print("=" * 70)

    hallucinated_report = {
        "mission_id": "MS-GT-100",
        "peak_speed_kmh": 95.0,  # Hallucinated speed! Actual is 54.2
        "incident_breakdown": [
            {"track_id": 99, "class_name": "airplane", "rule_type": "SPEED_LIMIT"}  # Hallucinated class & track ID!
        ]
    }
    audit_res_2 = auditor.evaluate_report(hallucinated_report)
    print("Audit Result (Hallucinated):", audit_res_2)
    assert audit_res_2["verdict"] == "UNRELIABLE_HALLUCINATION"
    assert audit_res_2["grounding_score"] < 0.70
    assert "airplane" in audit_res_2["hallucinated_classes"]
    print(f"✓ Detected hallucinated entity: {audit_res_2['hallucinated_classes']}")
    print(f"✓ Detected speed delta of {audit_res_2['speed_delta_kmh']} km/h")

    print("\n[SUCCESS] Module 04 Grounding Evaluator verified successfully!")


if __name__ == "__main__":
    test_grounding_evaluator()
