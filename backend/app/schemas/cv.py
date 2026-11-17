"""
=============================================================================
AeroVision AI - Computer Vision Data Schemas
=============================================================================
Strongly-typed data structures for video frames, bounding boxes, and object detections.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

@dataclass
class BoundingBox:
    """Bounding box in pixel coordinates (xyxy)."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def xywh(self) -> Tuple[float, float, float, float]:
        cx, cy = self.center
        return (cx, cy, self.width, self.height)

    def to_int_tuple(self) -> Tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2)))

@dataclass
class Detection:
    """Single object detection instance."""
    box: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "box": {
                "x1": round(self.box.x1, 2),
                "y1": round(self.box.y1, 2),
                "x2": round(self.box.x2, 2),
                "y2": round(self.box.y2, 2),
                "width": round(self.box.width, 2),
                "height": round(self.box.height, 2)
            },
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "track_id": self.track_id
        }

@dataclass
class FrameMetadata:
    """Metadata of a video frame."""
    frame_id: int
    timestamp_sec: float
    width: int
    height: int

@dataclass
class FrameDetectionResult:
    """Complete detection results for a single processed video frame."""
    metadata: FrameMetadata
    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.detections)

    def count_by_class(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts

    def get_boxes_xyxy(self) -> np.ndarray:
        if not self.detections:
            return np.empty((0, 4), dtype=np.float32)
        return np.array([d.box.xyxy for d in self.detections], dtype=np.float32)

    def get_confidences(self) -> np.ndarray:
        if not self.detections:
            return np.empty((0,), dtype=np.float32)
        return np.array([d.confidence for d in self.detections], dtype=np.float32)

    def get_class_ids(self) -> np.ndarray:
        if not self.detections:
            return np.empty((0,), dtype=np.int32)
        return np.array([d.class_id for d in self.detections], dtype=np.int32)

    def to_dict(self) -> Dict:
        return {
            "frame_id": self.metadata.frame_id,
            "timestamp_sec": round(self.metadata.timestamp_sec, 3),
            "resolution": [self.metadata.width, self.metadata.height],
            "inference_time_ms": round(self.inference_time_ms, 2),
            "detections_count": self.count,
            "counts_by_class": self.count_by_class(),
            "detections": [d.to_dict() for d in self.detections]
        }


# =============================================================================
# OBJECT TRACKING SCHEMAS
# =============================================================================

@dataclass
class TrackedObject:
    """Represents an active multi-frame object track with persistent ID and trajectory."""
    track_id: int
    box: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    velocity: Tuple[float, float] = (0.0, 0.0)
    speed_kmh: Optional[float] = None
    frames_active: int = 1
    state: str = "active"

    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "box": {
                "x1": round(self.box.x1, 2),
                "y1": round(self.box.y1, 2),
                "x2": round(self.box.x2, 2),
                "y2": round(self.box.y2, 2),
            },
            "center": (round(self.box.center[0], 2), round(self.box.center[1], 2)),
            "velocity_px_s": (round(self.velocity[0], 2), round(self.velocity[1], 2)),
            "speed_kmh": round(self.speed_kmh, 2) if self.speed_kmh is not None else None,
            "frames_active": self.frames_active,
            "state": self.state,
            "trail_length": len(self.trajectory)
        }

@dataclass
class FrameTrackingResult:
    """Complete tracking results for a processed video frame."""
    metadata: FrameMetadata
    tracked_objects: List[TrackedObject] = field(default_factory=list)
    cumulative_unique_objects: int = 0
    line_crossings: Dict[str, int] = field(default_factory=dict)
    inference_time_ms: float = 0.0

    @property
    def active_count(self) -> int:
        return len([obj for obj in self.tracked_objects if obj.state == "active"])

    def count_by_class(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for obj in self.tracked_objects:
            if obj.state == "active":
                counts[obj.class_name] = counts.get(obj.class_name, 0) + 1
        return counts

    def to_dict(self) -> Dict:
        return {
            "frame_id": self.metadata.frame_id,
            "timestamp_sec": round(self.metadata.timestamp_sec, 3),
            "active_tracks_count": self.active_count,
            "cumulative_unique_objects": self.cumulative_unique_objects,
            "counts_by_class": self.count_by_class(),
            "line_crossings": self.line_crossings,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "tracks": [t.to_dict() for t in self.tracked_objects]
        }
