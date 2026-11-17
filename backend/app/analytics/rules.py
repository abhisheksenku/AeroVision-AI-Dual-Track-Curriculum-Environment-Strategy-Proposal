"""
=============================================================================
AeroVision AI - Analytics Engine: Inspection Rules & Spatial Geometry
=============================================================================
Defines safety and security rules for aerial drone surveillance:
- Point-in-Polygon (PIP) ray casting
- Restricted Zone Incursions
- Over-speeding Violations
- Stationary Loitering / Obstructions
"""

from typing import List, Tuple, Dict, Optional, Set
import numpy as np

from backend.app.schemas.cv import TrackedObject
from backend.app.schemas.analytics import SecurityViolationEvent

def point_in_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    """
    Ray-Casting algorithm to determine if 2D point is inside arbitrary polygon.
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1_x, p1_y = polygon[0]
    for i in range(1, n + 1):
        p2_x, p2_y = polygon[i % n]
        if min(p1_y, p2_y) < y <= max(p1_y, p2_y):
            if x <= max(p1_x, p2_x):
                if p1_y != p2_y:
                    x_inters = (y - p1_y) * (p2_x - p1_x) / (p2_y - p1_y) + p1_x
                if p1_x == p2_x or x <= x_inters:
                    inside = not inside
        p1_x, p1_y = p2_x, p2_y

    return inside

class RestrictedZone:
    """Defines a polygonal boundary on the video canvas."""
    def __init__(
        self,
        name: str,
        polygon: List[Tuple[float, float]],
        allowed_classes: Optional[Set[str]] = None,
        severity: str = "HIGH",
        color_bgr: Tuple[int, int, int] = (0, 0, 255)
    ):
        self.name = name
        self.polygon = np.array(polygon, dtype=np.float64)
        self.allowed_classes = allowed_classes or set()
        self.severity = severity
        self.color_bgr = color_bgr

    def is_incursion(self, point: Tuple[float, float], class_name: str) -> bool:
        if class_name in self.allowed_classes:
            return False
        return point_in_polygon(point, self.polygon)

class SpeedLimitRule:
    """Evaluates whether moving objects exceed posted velocity limits."""
    def __init__(self, speed_limit_kmh: float = 40.0):
        self.speed_limit_kmh = speed_limit_kmh

    def evaluate(self, obj: TrackedObject) -> Optional[Tuple[str, str]]:
        if obj.speed_kmh is not None and obj.frames_active > 3:
            if obj.speed_kmh > self.speed_limit_kmh:
                excess = obj.speed_kmh - self.speed_limit_kmh
                severity = "CRITICAL" if excess > 20.0 else "HIGH" if excess > 10.0 else "MEDIUM"
                detail = f"Speed {obj.speed_kmh:.1f} km/h exceeded limit of {self.speed_limit_kmh:.1f} km/h (+{excess:.1f} km/h)"
                return severity, detail
        return None
