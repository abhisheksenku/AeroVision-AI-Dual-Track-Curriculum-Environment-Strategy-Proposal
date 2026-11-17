"""
=============================================================================
AeroVision AI - Analytics Engine: Core Processing & Dashboard Engine
=============================================================================
Aggregates trajectory analytics, evaluates security & speed rules,
and generates structured mission audit summaries.
"""

from typing import List, Dict, Tuple, Optional, Set, Any
import numpy as np
import cv2

from backend.app.schemas.cv import FrameTrackingResult, TrackedObject
from backend.app.schemas.analytics import (
    SecurityViolationEvent,
    VehicleClassStatistics,
    DroneMissionAnalyticsSummary
)
from backend.app.analytics.rules import RestrictedZone, SpeedLimitRule

class AeroVisionAnalyticsEngine:
    """
    Production Analytics Engine for AeroVision AI.
    Processes sequential FrameTrackingResults, applies safety inspection rules,
    accumulates statistical speed distributions, and generates audit reports.
    """
    def __init__(
        self,
        mission_id: str = "MISSION_001",
        speed_limit_kmh: float = 40.0
    ):
        self.mission_id = mission_id
        self.speed_rule = SpeedLimitRule(speed_limit_kmh=speed_limit_kmh)
        self.zones: Dict[str, RestrictedZone] = {}

        # Violation state
        self._next_event_id = 1
        self.violations_log: List[SecurityViolationEvent] = []
        # Prevent spamming alerts for the same track_id and rule: {(track_id, rule_name): last_alert_time}
        self._alert_cooldowns: Dict[Tuple[int, str], float] = {}
        self.cooldown_period_sec = 2.0

        # Statistical accumulators
        self.speeds_by_class: Dict[str, List[float]] = {}
        self.unique_tracks_by_class: Dict[str, Set[int]] = {}
        self.violations_by_class: Dict[str, int] = {}
        self.total_frames_processed = 0
        self.max_timestamp_sec = 0.0
        self.latest_line_crossings: Dict[str, int] = {}

    def add_restricted_zone(
        self,
        name: str,
        polygon: List[Tuple[float, float]],
        allowed_classes: Optional[Set[str]] = None,
        severity: str = "HIGH",
        color_bgr: Tuple[int, int, int] = (0, 0, 220)
    ):
        """Registers a polygonal zone for intrusion monitoring."""
        self.zones[name] = RestrictedZone(
            name=name,
            polygon=polygon,
            allowed_classes=allowed_classes,
            severity=severity,
            color_bgr=color_bgr
        )

    def process_frame(self, tracking_result: FrameTrackingResult) -> List[SecurityViolationEvent]:
        """
        Analyzes a single frame's tracking results against active inspection rules.
        Returns newly triggered security violation alerts for this frame.
        """
        self.total_frames_processed += 1
        ts = tracking_result.metadata.timestamp_sec
        self.max_timestamp_sec = max(self.max_timestamp_sec, ts)
        self.latest_line_crossings = tracking_result.line_crossings

        new_alerts: List[SecurityViolationEvent] = []

        for obj in tracking_result.tracked_objects:
            cls = obj.class_name
            tid = obj.track_id

            # Register unique track
            if cls not in self.unique_tracks_by_class:
                self.unique_tracks_by_class[cls] = set()
                self.speeds_by_class[cls] = []
                self.violations_by_class[cls] = 0

            self.unique_tracks_by_class[cls].add(tid)

            # Record speed if moving
            if obj.speed_kmh is not None and obj.frames_active > 3:
                self.speeds_by_class[cls].append(obj.speed_kmh)

            # 1. Evaluate Speed Violation
            speed_eval = self.speed_rule.evaluate(obj)
            if speed_eval is not None:
                severity, details = speed_eval
                key = (tid, "SPEED_VIOLATION")
                if key not in self._alert_cooldowns or (ts - self._alert_cooldowns[key]) >= self.cooldown_period_sec:
                    self._alert_cooldowns[key] = ts
                    event = SecurityViolationEvent(
                        event_id=self._next_event_id,
                        rule_type="SPEED_VIOLATION",
                        track_id=tid,
                        class_name=cls,
                        timestamp_sec=ts,
                        severity=severity,
                        details=details,
                        location=obj.box.center
                    )
                    self._next_event_id += 1
                    self.violations_log.append(event)
                    self.violations_by_class[cls] = self.violations_by_class.get(cls, 0) + 1
                    new_alerts.append(event)

            # 2. Evaluate Zone Incursion
            for z_name, zone in self.zones.items():
                if zone.is_incursion(obj.box.center, cls):
                    key = (tid, f"ZONE_{z_name}")
                    if key not in self._alert_cooldowns or (ts - self._alert_cooldowns[key]) >= self.cooldown_period_sec:
                        self._alert_cooldowns[key] = ts
                        event = SecurityViolationEvent(
                            event_id=self._next_event_id,
                            rule_type="RESTRICTED_ZONE_INCURSION",
                            track_id=tid,
                            class_name=cls,
                            timestamp_sec=ts,
                            severity=zone.severity,
                            details=f"Unauthorized {cls} entered restricted zone '{z_name}'",
                            location=obj.box.center
                        )
                        self._next_event_id += 1
                        self.violations_log.append(event)
                        self.violations_by_class[cls] = self.violations_by_class.get(cls, 0) + 1
                        new_alerts.append(event)

        return new_alerts

    def generate_summary(self) -> DroneMissionAnalyticsSummary:
        """Generates comprehensive structured mission analytics."""
        unique_counts = {cls: len(s) for cls, s in self.unique_tracks_by_class.items()}
        total_unique = sum(unique_counts.values())

        class_stats: Dict[str, VehicleClassStatistics] = {}
        for cls, spds in self.speeds_by_class.items():
            if spds:
                arr = np.array(spds)
                mean_s = float(np.mean(arr))
                median_s = float(np.median(arr))
                p95_s = float(np.percentile(arr, 95))
                max_s = float(np.max(arr))
            else:
                mean_s, median_s, p95_s, max_s = 0.0, 0.0, 0.0, 0.0

            class_stats[cls] = VehicleClassStatistics(
                class_name=cls,
                unique_count=unique_counts.get(cls, 0),
                mean_speed_kmh=mean_s,
                median_speed_kmh=median_s,
                p95_speed_kmh=p95_s,
                max_speed_kmh=max_s,
                violations_count=self.violations_by_class.get(cls, 0)
            )

        total_violations = len(self.violations_log)
        status = "COMPLIANT" if total_violations == 0 else "FLAGGED"

        return DroneMissionAnalyticsSummary(
            mission_id=self.mission_id,
            duration_sec=self.max_timestamp_sec,
            total_unique_objects=total_unique,
            counts_by_class=unique_counts,
            class_statistics=class_stats,
            line_crossings=self.latest_line_crossings,
            total_violations=total_violations,
            violations_log=list(self.violations_log),
            compliance_status=status
        )

    def annotate_analytics(
        self,
        frame: np.ndarray,
        active_alerts: List[SecurityViolationEvent]
    ) -> np.ndarray:
        """
        Draws semi-transparent restricted zones, zone labels,
        and bottom alert banners for active violations.
        """
        annotated = frame.copy()
        overlay = frame.copy()

        # 1. Draw Zones (with 25% alpha transparency)
        for zone in self.zones.values():
            pts = zone.polygon.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], zone.color_bgr)
            cv2.polylines(annotated, [pts], isClosed=True, color=zone.color_bgr, thickness=2, lineType=cv2.LINE_AA)

            # Zone Label
            label_x = int(zone.polygon[:, 0].mean())
            label_y = int(zone.polygon[:, 1].min()) + 25
            cv2.putText(annotated, f"ZONE: {zone.name}", (label_x - 80, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, zone.color_bgr, 2, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.20, annotated, 0.80, 0, annotated)

        # 2. Draw Active Violation Banner at Bottom
        if active_alerts:
            h, w = frame.shape[:2]
            cv2.rectangle(annotated, (0, h - 45), (w, h), (0, 0, 180), -1)
            latest = active_alerts[-1]
            alert_text = f"SAFETY ALERT [{latest.severity}]: {latest.details} (#{latest.track_id} at {latest.timestamp_sec:.1f}s)"
            cv2.putText(annotated, alert_text, (20, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        return annotated
