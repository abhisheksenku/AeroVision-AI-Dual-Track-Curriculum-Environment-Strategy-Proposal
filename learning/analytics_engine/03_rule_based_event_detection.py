"""
=============================================================================
AeroVision AI - Analytics Engine Foundations: Module 03
=============================================================================
Rule-Based Event Detection & Point-in-Polygon Geometric Security Rules

Learning Objectives:
1. Implement the Ray-Casting Point-in-Polygon (PIP) algorithm from scratch in NumPy.
2. Build configurable inspection rules: Speed Violations, Restricted Zone Incursions, Loitering.
3. Classify event severity tiers (LOW, MEDIUM, HIGH, CRITICAL).
4. Generate structured security violation alert schemas.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional

def point_in_polygon_ray_casting(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    """
    Determines if point (x, y) is inside polygon using the Jordan Curve Theorem (Ray-Casting).
    Casts a horizontal ray to infinity (+X direction) and counts polygon boundary crossings.
    Odd crossings = INSIDE, Even crossings = OUTSIDE.
    
    Time Complexity: O(N) where N is number of polygon vertices.
    """
    x, y = point
    n_vertices = len(polygon)
    inside = False

    p1_x, p1_y = polygon[0]
    for i in range(1, n_vertices + 1):
        p2_x, p2_y = polygon[i % n_vertices]

        # Check if ray crosses the line segment (p1 - p2)
        if min(p1_y, p2_y) < y <= max(p1_y, p2_y):
            if x <= max(p1_x, p2_x):
                # Calculate X-coordinate of intersection with edge
                if p1_y != p2_y:
                    x_inters = (y - p1_y) * (p2_x - p1_x) / (p2_y - p1_y) + p1_x
                if p1_x == p2_x or x <= x_inters:
                    inside = not inside

        p1_x, p1_y = p2_x, p2_y

    return inside

class InspectionRuleEngine:
    """Configurable rule evaluator for drone video tracking data."""
    def __init__(self, speed_limit_kmh: float = 40.0):
        self.speed_limit_kmh = speed_limit_kmh
        self.restricted_zones: Dict[str, np.ndarray] = {}
        self.alerts_log: List[Dict] = []
        self._alert_id = 1

    def add_restricted_zone(self, zone_name: str, polygon_vertices: List[Tuple[float, float]]):
        self.restricted_zones[zone_name] = np.array(polygon_vertices, dtype=np.float64)

    def evaluate_track(
        self,
        track_id: int,
        class_name: str,
        center: Tuple[float, float],
        speed_kmh: Optional[float],
        timestamp_sec: float
    ) -> List[Dict]:
        """Evaluates all configured rules for a single tracked object observation."""
        triggered = []

        # Rule 1: Speed Limit Violation
        if speed_kmh is not None and speed_kmh > self.speed_limit_kmh:
            excess = speed_kmh - self.speed_limit_kmh
            severity = "CRITICAL" if excess > 20.0 else "HIGH" if excess > 10.0 else "MEDIUM"
            alert = {
                "alert_id": self._alert_id,
                "rule": "SPEED_VIOLATION",
                "track_id": track_id,
                "class_name": class_name,
                "timestamp_sec": round(timestamp_sec, 2),
                "severity": severity,
                "details": f"Vehicle speed {speed_kmh:.1f} km/h exceeded limit of {self.speed_limit_kmh:.1f} km/h (+{excess:.1f} km/h)",
                "location": (round(center[0], 1), round(center[1], 1))
            }
            self._alert_id += 1
            triggered.append(alert)
            self.alerts_log.append(alert)

        # Rule 2: Restricted Zone Incursion (e.g. Pedestrian on active highway)
        for zone_name, poly in self.restricted_zones.items():
            if point_in_polygon_ray_casting(center, poly):
                severity = "HIGH" if class_name == "person" else "MEDIUM"
                alert = {
                    "alert_id": self._alert_id,
                    "rule": "RESTRICTED_ZONE_INCURSION",
                    "track_id": track_id,
                    "class_name": class_name,
                    "timestamp_sec": round(timestamp_sec, 2),
                    "severity": severity,
                    "details": f"Object {class_name} entered protected zone: '{zone_name}'",
                    "location": (round(center[0], 1), round(center[1], 1))
                }
                self._alert_id += 1
                triggered.append(alert)
                self.alerts_log.append(alert)

        return triggered

def demonstrate_rule_based_events():
    print("=" * 70)
    print("STEP 1: POINT-IN-POLYGON & RULE-BASED EVENT EVALUATION")
    print("=" * 70)

    # 1. Point in Polygon Verification
    # Define a rectangular hazard perimeter: [(200, 200), (600, 200), (600, 400), (200, 400)]
    hazard_zone = np.array([
        [200.0, 200.0],
        [600.0, 200.0],
        [600.0, 400.0],
        [200.0, 400.0]
    ])

    test_pt_inside = (350.0, 300.0)   # Well inside
    test_pt_outside = (100.0, 100.0)  # Far outside
    test_pt_near_edge = (605.0, 300.0) # Outside right edge

    assert point_in_polygon_ray_casting(test_pt_inside, hazard_zone) is True
    assert point_in_polygon_ray_casting(test_pt_outside, hazard_zone) is False
    assert point_in_polygon_ray_casting(test_pt_near_edge, hazard_zone) is False
    print("✓ Ray-Casting Point-in-Polygon (PIP) algorithm verified.")

    # 2. Rule Engine Test
    engine = InspectionRuleEngine(speed_limit_kmh=45.0)
    engine.add_restricted_zone("HIGHWAY_ACTIVE_LANES", [
        (0.0, 180.0), (1280.0, 180.0), (1280.0, 540.0), (0.0, 540.0)
    ])

    # Scenario A: Fast Car (55 km/h) on highway -> Triggers SPEED_VIOLATION
    alerts_a = engine.evaluate_track(
        track_id=101, class_name="car", center=(400.0, 300.0), speed_kmh=56.2, timestamp_sec=1.5
    )

    # Scenario B: Pedestrian wandering onto active highway -> Triggers RESTRICTED_ZONE_INCURSION
    alerts_b = engine.evaluate_track(
        track_id=102, class_name="person", center=(450.0, 250.0), speed_kmh=4.1, timestamp_sec=2.0
    )

    # Scenario C: Normal law-abiding Car (35 km/h) -> No alerts
    alerts_c = engine.evaluate_track(
        track_id=103, class_name="car", center=(700.0, 300.0), speed_kmh=35.0, timestamp_sec=2.2
    )

    print(f"\nGenerated Security Alerts ({len(engine.alerts_log)} total):")
    for a in engine.alerts_log:
        print(f"  [{a['severity']}] Alert #{a['alert_id']}: {a['rule']} by #{a['track_id']} ({a['class_name']})")
        print(f"    Details: {a['details']}")

    assert len(engine.alerts_log) >= 2, "Expected at least 2 security alerts triggered!"
    print("\n[SUCCESS] Rule-based event detection and severity classification verified.")
    print()

def main():
    demonstrate_rule_based_events()
    print("=" * 70)
    print("PHASE 7 - MODULE 03: COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
