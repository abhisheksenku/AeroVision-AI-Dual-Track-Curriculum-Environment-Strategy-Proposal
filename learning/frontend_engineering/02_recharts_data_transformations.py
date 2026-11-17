"""
=============================================================================
AeroVision AI - Frontend Engineering Foundations: Module 02
=============================================================================
Data Transformations for Recharts Visualizations & Dashboard Analytics

Learning Objectives:
1. Transform backend relational JSON summaries into Recharts-compliant array payloads.
2. Build data pipelines for Class Distribution, Speed Percentiles, and Severity Donut charts.
3. Map categorical classes to distinct Tailwind/Hex visual color tokens.
"""

from typing import Dict, List, Any

CLASS_COLORS = {
    "car": "#38bdf8",           # Sky 400
    "bus": "#eab308",           # Yellow 500
    "truck": "#22c55e",         # Green 500
    "person": "#f97316",        # Orange 500
    "motorcycle": "#ec4899",    # Pink 500
    "bicycle": "#a855f7",       # Purple 500
    "default": "#94a3b8"        # Slate 400
}

SEVERITY_COLORS = {
    "CRITICAL": "#ef4444",      # Red 500
    "HIGH": "#f97316",          # Orange 500
    "MEDIUM": "#eab308",        # Yellow 500
    "LOW": "#3b82f6"            # Blue 500
}

def transform_class_distribution_for_barchart(counts_by_class: Dict[str, int]) -> List[Dict[str, Any]]:
    """
    Transforms {'car': 3, 'bus': 2, ...} ->
    [{'name': 'Car', 'count': 3, 'fill': '#38bdf8'}, ...]
    """
    return [
        {
            "name": cls.capitalize(),
            "count": count,
            "fill": CLASS_COLORS.get(cls.lower(), CLASS_COLORS["default"])
        }
        for cls, count in counts_by_class.items()
    ]

def transform_speed_stats_for_chart(class_statistics: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transforms backend speed stats into multi-bar chart payload comparing Mean vs P95.
    """
    data = []
    for cls, stats in class_statistics.items():
        data.append({
            "name": cls.capitalize(),
            "mean_speed": stats.get("mean_speed_kmh", 0.0),
            "p95_speed": stats.get("p95_speed_kmh", 0.0),
            "max_speed": stats.get("max_speed_kmh", 0.0)
        })
    return sorted(data, key=lambda x: x["mean_speed"], reverse=True)

def transform_violations_for_piechart(violations_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregates raw violation logs into severity breakdown for Recharts PieChart.
    """
    counts = {}
    for v in violations_log:
        sev = v.get("severity", "LOW")
        counts[sev] = counts.get(sev, 0) + 1

    return [
        {
            "name": sev,
            "value": count,
            "fill": SEVERITY_COLORS.get(sev, "#94a3b8")
        }
        for sev, count in sorted(counts.items())
    ]

def demonstrate_recharts_transforms():
    print("=" * 70)
    print("STEP 1: RECHARTS PAYLOAD SHAPING & COLOR TOKEN MAPPING")
    print("=" * 70)

    # Sample backend payload
    counts = {"car": 4, "bus": 2, "person": 3, "truck": 1}
    speed_stats = {
        "car": {"mean_speed_kmh": 32.4, "p95_speed_kmh": 38.0, "max_speed_kmh": 41.2},
        "bus": {"mean_speed_kmh": 28.1, "p95_speed_kmh": 31.0, "max_speed_kmh": 32.5},
        "person": {"mean_speed_kmh": 4.2, "p95_speed_kmh": 5.1, "max_speed_kmh": 5.8}
    }
    violations = [
        {"severity": "CRITICAL"}, {"severity": "HIGH"}, {"severity": "HIGH"}, {"severity": "MEDIUM"}
    ]

    # 1. Bar Chart Data
    bar_data = transform_class_distribution_for_barchart(counts)
    print("Recharts BarChart Payload (Class Distribution):")
    for item in bar_data:
        print(f"  {item['name']:<10}: Count = {item['count']} (Color: {item['fill']})")

    assert len(bar_data) == 4
    assert bar_data[0]["fill"].startswith("#")

    # 2. Speed Stats Data
    speed_data = transform_speed_stats_for_chart(speed_stats)
    print("\nRecharts Multi-Bar Payload (Speed Metrics):")
    for item in speed_data:
        print(f"  {item['name']:<10}: Mean = {item['mean_speed']} km/h | P95 = {item['p95_speed']} km/h")

    assert speed_data[0]["name"] == "Car"

    # 3. Violations Pie Chart Data
    pie_data = transform_violations_for_piechart(violations)
    print("\nRecharts PieChart Payload (Severity Breakdown):")
    for item in pie_data:
        print(f"  {item['name']:<10}: Violations = {item['value']} (Color: {item['fill']})")

    assert len(pie_data) == 3 # CRITICAL, HIGH, MEDIUM
    print("\n[SUCCESS] Recharts data transformations and color palette binding verified.")
    print()

def main():
    demonstrate_recharts_transforms()
    print("=" * 70)
    print("PHASE 9 - MODULE 02: COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
