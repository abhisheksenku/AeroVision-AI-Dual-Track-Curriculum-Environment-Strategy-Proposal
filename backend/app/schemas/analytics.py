"""
=============================================================================
AeroVision AI - Analytics Schemas
=============================================================================
Data models for drone traffic analytics, safety compliance audits, and event alerts.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

@dataclass
class SecurityViolationEvent:
    """Structured security or safety rule violation alert."""
    event_id: int
    rule_type: str        # e.g., 'SPEED_VIOLATION', 'RESTRICTED_ZONE_INCURSION'
    track_id: int
    class_name: str
    timestamp_sec: float
    severity: str         # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    details: str
    location: Tuple[float, float] # (x, y) coordinates

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": int(self.event_id),
            "rule_type": str(self.rule_type),
            "track_id": int(self.track_id),
            "class_name": str(self.class_name),
            "timestamp_sec": round(float(self.timestamp_sec), 2),
            "severity": str(self.severity),
            "details": str(self.details),
            "location": (round(float(self.location[0]), 1), round(float(self.location[1]), 1))
        }

@dataclass
class VehicleClassStatistics:
    """Class-level speed and inventory statistics."""
    class_name: str
    unique_count: int
    mean_speed_kmh: float
    median_speed_kmh: float
    p95_speed_kmh: float
    max_speed_kmh: float
    violations_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": str(self.class_name),
            "unique_count": int(self.unique_count),
            "mean_speed_kmh": round(float(self.mean_speed_kmh), 1),
            "median_speed_kmh": round(float(self.median_speed_kmh), 1),
            "p95_speed_kmh": round(float(self.p95_speed_kmh), 1),
            "max_speed_kmh": round(float(self.max_speed_kmh), 1),
            "violations_count": int(self.violations_count)
        }

@dataclass
class DroneMissionAnalyticsSummary:
    """Complete comprehensive mission analytics report."""
    mission_id: str
    duration_sec: float
    total_unique_objects: int
    counts_by_class: Dict[str, int]
    class_statistics: Dict[str, VehicleClassStatistics]
    line_crossings: Dict[str, int]
    total_violations: int
    violations_log: List[SecurityViolationEvent] = field(default_factory=list)
    compliance_status: str = "COMPLIANT"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission_id": str(self.mission_id),
            "duration_sec": round(float(self.duration_sec), 2),
            "total_unique_objects": int(self.total_unique_objects),
            "counts_by_class": {str(k): int(v) for k, v in self.counts_by_class.items()},
            "class_statistics": {str(k): v.to_dict() for k, v in self.class_statistics.items()},
            "line_crossings": {str(k): int(v) for k, v in self.line_crossings.items()},
            "safety_audit": {
                "total_violations": int(self.total_violations),
                "compliance_status": str(self.compliance_status),
                "violations_log": [v.to_dict() for v in self.violations_log]
            }
        }
