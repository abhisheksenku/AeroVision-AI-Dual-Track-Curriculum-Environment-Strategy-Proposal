"""
=============================================================================
AeroVision AI - Computer Vision Engine: ByteTrack Object Tracker
=============================================================================

Production Multi-Object Tracker wrapping YOLO11 + ByteTrack.
Provides:
1. Persistent object IDs across drone camera motion and occlusions.
2. Centroid trajectory history management with fading visual motion trails.
3. Velocity and calibrated physical speed estimation (km/h).
4. Virtual tripwires / line-crossing event detectors for directional flow counting.
5. Informative anti-aliased HUD and tracking visualization.
"""

import time
import os
from typing import List, Dict, Tuple, Optional, Set
import cv2
import numpy as np
from ultralytics import YOLO

from backend.app.schemas.cv import (
    BoundingBox,
    FrameMetadata,
    TrackedObject,
    FrameTrackingResult
)

# Deterministic distinct color palette generator based on track ID
def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Generates a stable, visually distinct BGR color for a given track ID."""
    np.random.seed(track_id * 101 % 10000)
    color = np.random.randint(60, 255, size=3).tolist()
    return (int(color[0]), int(color[1]), int(color[2]))

def segments_intersect(p1: Tuple[float, float], p2: Tuple[float, float],
                       q1: Tuple[float, float], q2: Tuple[float, float]) -> bool:
    """
    Checks if line segment p1-p2 intersects line segment q1-q2 using 2D cross product.
    """
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))

class Tripwire:
    """Virtual line tripwire for bidirectional object counting."""
    def __init__(self, name: str, start_pt: Tuple[int, int], end_pt: Tuple[int, int]):
        self.name = name
        self.start_pt = start_pt
        self.end_pt = end_pt
        self.crossed_ids: Set[int] = set()
        self.crossing_count = 0

    def check_crossing(self, track_id: int, prev_pt: Tuple[float, float], curr_pt: Tuple[float, float]) -> bool:
        if track_id in self.crossed_ids:
            return False
            
        if segments_intersect(prev_pt, curr_pt, self.start_pt, self.end_pt):
            self.crossed_ids.add(track_id)
            self.crossing_count += 1
            return True
            
        return False

class AeroVisionTracker:
    """
    Production Multi-Object Tracker integrating YOLO11 with ByteTrack.
    Maintains persistent IDs, tracks centroid history, estimates velocities,
    and detects virtual tripwire crossing events.
    """
    def __init__(
        self,
        model_path: str = "models/weights/yolo11n.pt",
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.50,
        tracker_type: str = "bytetrack.yaml",
        target_classes: Optional[List[str]] = None,
        max_trail_len: int = 30,
        gsd_m_per_px: Optional[float] = 0.05, # Default 5 cm/px Ground Sampling Distance
        device: str = "cpu"
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at: {model_path}")

        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.tracker_type = tracker_type
        self.max_trail_len = max_trail_len
        self.gsd_m_per_px = gsd_m_per_px
        self.device = device

        # Load YOLO model
        self.model = YOLO(model_path)
        self.names = self.model.names
        self.name_to_id = {name: idx for idx, name in self.names.items()}

        self.target_class_ids = None
        if target_classes is not None:
            self.target_class_ids = [
                self.name_to_id[c] for c in target_classes if c in self.name_to_id
            ]

        # Internal tracking state:
        # {track_id: {"trajectory": [(cx, cy)], "last_timestamp": float, "class_name": str, "frames_active": int}}
        self._tracks_history: Dict[int, Dict] = {}
        self._cumulative_unique_ids: Set[int] = set()
        self.tripwires: Dict[str, Tripwire] = {}

    def add_tripwire(self, name: str, start_pt: Tuple[int, int], end_pt: Tuple[int, int]):
        """Registers a virtual tripwire line for object crossing counting."""
        self.tripwires[name] = Tripwire(name, start_pt, end_pt)

    def track_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp_sec: float = 0.0
    ) -> FrameTrackingResult:
        """
        Executes ByteTrack on the frame, updates persistent trajectories,
        evaluates line crossings, and returns structured tracking schemas.
        """
        h, w = frame.shape[:2]
        meta = FrameMetadata(
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            width=w,
            height=h
        )

        t_start = time.perf_counter()

        # Run ByteTrack through Ultralytics
        results = self.model.track(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=self.target_class_ids,
            tracker=self.tracker_type,
            persist=True,
            device=self.device,
            verbose=False
        )

        inference_time_ms = (time.perf_counter() - t_start) * 1000.0

        tracked_objects: List[TrackedObject] = []
        active_ids_in_frame: Set[int] = set()

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes_obj = results[0].boxes
            if len(boxes_obj) > 0 and boxes_obj.id is not None:
                xyxy = boxes_obj.xyxy.cpu().numpy()
                confs = boxes_obj.conf.cpu().numpy()
                classes = boxes_obj.cls.cpu().numpy().astype(int)
                track_ids = boxes_obj.id.cpu().numpy().astype(int)

                for box_coord, conf, cls_id, tid in zip(xyxy, confs, classes, track_ids):
                    class_name = self.names.get(cls_id, f"class_{cls_id}")
                    self._cumulative_unique_ids.add(tid)
                    active_ids_in_frame.add(tid)

                    bbox = BoundingBox(
                        x1=float(box_coord[0]),
                        y1=float(box_coord[1]),
                        x2=float(box_coord[2]),
                        y2=float(box_coord[3])
                    )
                    curr_cx, curr_cy = bbox.center

                    # Update track history
                    if tid not in self._tracks_history:
                        self._tracks_history[tid] = {
                            "trajectory": [(curr_cx, curr_cy)],
                            "last_time": timestamp_sec,
                            "class_name": class_name,
                            "frames_active": 1,
                            "velocity": (0.0, 0.0)
                        }
                        prev_pt = (curr_cx, curr_cy)
                        dt = 1.0 / 30.0
                    else:
                        hist = self._tracks_history[tid]
                        prev_pt = hist["trajectory"][-1]
                        dt = max(0.001, timestamp_sec - hist["last_time"])

                        hist["trajectory"].append((curr_cx, curr_cy))
                        if len(hist["trajectory"]) > self.max_trail_len:
                            hist["trajectory"].pop(0)

                        # Instantaneous velocity (pixels/second)
                        vx = (curr_cx - prev_pt[0]) / dt
                        vy = (curr_cy - prev_pt[1]) / dt
                        hist["velocity"] = (vx, vy)
                        hist["last_time"] = timestamp_sec
                        hist["frames_active"] += 1

                    # Physical speed estimation (km/h)
                    vx, vy = self._tracks_history[tid]["velocity"]
                    speed_px_s = np.hypot(vx, vy)
                    speed_kmh = None
                    if self.gsd_m_per_px is not None:
                        speed_m_s = speed_px_s * self.gsd_m_per_px
                        speed_kmh = speed_m_s * 3.6  # m/s to km/h

                    # Check tripwires
                    for tw in self.tripwires.values():
                        tw.check_crossing(tid, prev_pt, (curr_cx, curr_cy))

                    tracked_obj = TrackedObject(
                        track_id=tid,
                        box=bbox,
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=class_name,
                        trajectory=list(self._tracks_history[tid]["trajectory"]),
                        velocity=(vx, vy),
                        speed_kmh=speed_kmh,
                        frames_active=self._tracks_history[tid]["frames_active"],
                        state="active"
                    )
                    tracked_objects.append(tracked_obj)

        crossings_summary = {name: tw.crossing_count for name, tw in self.tripwires.items()}

        return FrameTrackingResult(
            metadata=meta,
            tracked_objects=tracked_objects,
            cumulative_unique_objects=len(self._cumulative_unique_ids),
            line_crossings=crossings_summary,
            inference_time_ms=inference_time_ms
        )

    def annotate(
        self,
        frame: np.ndarray,
        result: FrameTrackingResult,
        draw_trails: bool = True,
        draw_tripwires: bool = True,
        draw_hud: bool = True
    ) -> np.ndarray:
        """
        Renders bounding boxes, persistent track IDs, fading motion trails,
        virtual tripwire lines, and tracking telemetry onto the frame.
        """
        annotated = frame.copy()

        # 1. Draw Virtual Tripwires
        if draw_tripwires:
            for name, tw in self.tripwires.items():
                cv2.line(annotated, tw.start_pt, tw.end_pt, (0, 0, 255), 2, cv2.LINE_AA)
                mid_x = (tw.start_pt[0] + tw.end_pt[0]) // 2
                mid_y = (tw.start_pt[1] + tw.end_pt[1]) // 2
                badge_text = f"TRIPWIRE: {tw.crossing_count}"
                cv2.putText(annotated, badge_text, (mid_x + 5, mid_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        # 2. Draw Tracked Objects & Trails
        for obj in result.tracked_objects:
            color = get_track_color(obj.track_id)
            x1, y1, x2, y2 = obj.box.to_int_tuple()

            # Draw Motion Trail
            if draw_trails and len(obj.trajectory) > 1:
                pts = np.array(obj.trajectory, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

            # Draw Bounding Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Draw Label Badge: #ID Class (Speed km/h)
            label = f"#{obj.track_id} {obj.class_name}"
            if obj.speed_kmh is not None and obj.frames_active > 3:
                label += f" {obj.speed_kmh:.0f}km/h"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            badge_y1 = max(0, y1 - text_h - baseline - 4)
            badge_y2 = y1
            badge_x2 = min(frame.shape[1], x1 + text_w + 6)

            cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                        font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        # 3. Draw Telemetry HUD
        if draw_hud:
            cv2.rectangle(annotated, (frame.shape[1] - 320, 20), (frame.shape[1] - 20, 110), (0, 0, 0), -1)
            cv2.rectangle(annotated, (frame.shape[1] - 320, 20), (frame.shape[1] - 20, 110), (0, 215, 255), 1)

            cv2.putText(annotated, f"ACTIVE TRACKS : {result.active_count}", (frame.shape[1] - 305, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"CUMULATIVE ID : {result.cumulative_unique_objects}", (frame.shape[1] - 305, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"LATENCY       : {result.inference_time_ms:.1f} ms", (frame.shape[1] - 305, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 1, cv2.LINE_AA)

        return annotated
