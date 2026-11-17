"""
=============================================================================
AeroVision AI - Analytics Engine Foundations: Module 04
=============================================================================
Statistical Aggregations, Percentiles, and Structured Analytics JSON Export

Learning Objectives:
1. Compute traffic statistics: Mean, Median, P95 speed, and IQR outlier boundaries.
2. Calculate temporal traffic throughput and hourly flow rates.
3. Structure and serialize mission analytics into compliant JSON schemas.
4. Prepare aggregated metrics for FastAPI REST consumption and LLM RAG pipelines.
"""

import json
import numpy as np
from typing import Dict, List, Any

def compute_traffic_statistics(speeds: List[float]) -> Dict[str, float]:
    """Computes robust statistical metrics on vehicle speeds."""
    if not speeds:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "iqr": 0.0}

    arr = np.array(speeds, dtype=np.float64)
    mean_spd = float(np.mean(arr))
    median_spd = float(np.median(arr))
    p95_spd = float(np.percentile(arr, 95))
    
    q75, q25 = np.percentile(arr, [75, 25])
    iqr_spd = float(q75 - q25)

    return {
        "mean_speed_kmh": round(mean_spd, 2),
        "median_speed_kmh": round(median_spd, 2),
        "p95_speed_kmh": round(p95_spd, 2),
        "iqr_speed_kmh": round(iqr_spd, 2),
        "sample_count": len(speeds)
    }

def build_mission_analytics_summary(
    mission_id: str,
    duration_sec: float,
    unique_objects_by_class: Dict[str, int],
    speeds_by_class: Dict[str, List[float]],
    total_violations: int,
    tripwire_crossings: Dict[str, int]
) -> Dict[str, Any]:
    """Constructs a production-grade structured analytics JSON report."""
    total_unique = sum(unique_objects_by_class.values())
    flow_rate_per_hour = (total_unique / duration_sec * 3600.0) if duration_sec > 0 else 0.0

    class_stats = {}
    for cls_name, spds in speeds_by_class.items():
        class_stats[cls_name] = compute_traffic_statistics(spds)

    summary = {
        "mission_metadata": {
            "mission_id": mission_id,
            "duration_sec": round(duration_sec, 2),
            "estimated_flow_rate_per_hour": round(flow_rate_per_hour, 1)
        },
        "inventory_summary": {
            "total_unique_objects": total_unique,
            "counts_by_class": unique_objects_by_class
        },
        "traffic_speed_metrics": class_stats,
        "perimeter_flow_crossings": tripwire_crossings,
        "safety_audit": {
            "total_violations_recorded": total_violations,
            "compliance_status": "FLAGGED" if total_violations > 0 else "COMPLIANT"
        }
    }
    return summary

def demonstrate_statistical_export():
    print("=" * 70)
    print("STEP 1: TRAFFIC SPEED PERCENTILES & IQR CALCULATION")
    print("=" * 70)

    # Simulated car speeds (km/h) with two extreme speeder outliers
    np.random.seed(42)
    normal_speeds = np.random.normal(loc=38.0, scale=4.0, size=80).tolist()
    speeders = [62.0, 71.5]
    all_speeds = normal_speeds + speeders

    stats = compute_traffic_statistics(all_speeds)
    print(f"Total Speed Observations : {stats['sample_count']}")
    print(f"Mean Speed (km/h)        : {stats['mean_speed_kmh']}")
    print(f"Median Speed (km/h)      : {stats['median_speed_kmh']}")
    print(f"P95 Speed (km/h)         : {stats['p95_speed_kmh']} (Excludes standard distribution tails)")
    print(f"Interquartile Range (IQR): {stats['iqr_speed_kmh']}")

    assert stats["p95_speed_kmh"] > stats["median_speed_kmh"]

    print("\n" + "=" * 70)
    print("STEP 2: STRUCTURED MISSION ANALYTICS JSON SERIALIZATION")
    print("=" * 70)

    report = build_mission_analytics_summary(
        mission_id="UAV_FLIGHT_TEST_01",
        duration_sec=4.0,
        unique_objects_by_class={"car": 4, "bus": 1, "truck": 1, "person": 2},
        speeds_by_class={
            "car": all_speeds,
            "bus": [28.5, 30.0, 31.2],
            "truck": [24.0, 25.1],
            "person": [3.8, 4.2]
        },
        total_violations=3,
        tripwire_crossings={"highway_center_screen": 2}
    )

    json_str = json.dumps(report, indent=2)
    print(json_str)

    # Verify JSON roundtrip
    parsed = json.loads(json_str)
    assert parsed["inventory_summary"]["total_unique_objects"] == 8
    assert parsed["safety_audit"]["compliance_status"] == "FLAGGED"
    print("\n[SUCCESS] Statistical aggregations and JSON schema serialization verified.")

def main():
    demonstrate_statistical_export()
    print("=" * 70)
    print("PHASE 7 - MODULE 04: COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
