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
