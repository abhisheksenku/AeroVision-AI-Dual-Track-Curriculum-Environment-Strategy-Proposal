"""
=============================================================================
AeroVision AI - Computer Vision Engine: YOLO11 Object Detector
=============================================================================

Production Deep Learning Detection Engine Wrapping Ultralytics YOLO11n.
Optimized for CPU inference on aerial drone imagery with configurable confidence
thresholds, Non-Maximum Suppression (NMS), class filtering, and visual annotation.
"""

import time
import os
from typing import List, Optional, Dict, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from backend.app.schemas.cv import (
    BoundingBox,
    Detection,
    FrameMetadata,
    FrameDetectionResult
)

# Standard distinct color palette for aerial object classes (BGR format for OpenCV)
CLASS_COLOR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "person":     (0, 165, 255),    # Orange
    "car":        (255, 144, 30),   # Sky Blue
    "truck":      (50, 205, 50),    # Lime Green
    "bus":        (0, 215, 255),    # Yellow/Gold
    "motorcycle": (255, 105, 180),  # Hot Pink
    "bicycle":    (180, 105, 255),  # Purple
    "airplane":   (238, 130, 238),  # Violet
    "boat":       (255, 191, 0),    # Deep Sky Blue
    "default":    (0, 255, 128)     # Spring Green
}

class YOLO11Detector:
    """
    Production-grade Object Detector wrapper around Ultralytics YOLO11n.
    Executes CPU inference, filters detections, and parses coordinates into
    standardized schemas.
    """
    def __init__(
        self,
        model_path: str = "models/weights/yolo11n.pt",
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        target_classes: Optional[List[str]] = None,
        device: str = "cpu"
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO11 weights not found at: {model_path}")
            
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        
        # Load Ultralytics model
        self.model = YOLO(model_path)
        
        # Class mapping from model: {0: 'person', 1: 'bicycle', ...}
        self.names = self.model.names
        self.name_to_id = {name: idx for idx, name in self.names.items()}
        
        # Configure target classes filter (None = detect all 80 COCO classes)
        self.target_class_ids = None
        if target_classes is not None:
            self.target_class_ids = [
                self.name_to_id[c] for c in target_classes if c in self.name_to_id
            ]

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp_sec: float = 0.0
    ) -> FrameDetectionResult:
        """
        Runs object detection on a single video frame.
        
        Args:
            frame: Input image in BGR format (Height, Width, Channels: 3).
            frame_id: Monotonically increasing frame index.
            timestamp_sec: Video playback timestamp in seconds.
            
        Returns:
            FrameDetectionResult: Structured detection metadata and bounding boxes.
        """
        h, w = frame.shape[:2]
        meta = FrameMetadata(
            frame_id=frame_id,
            timestamp_sec=timestamp_sec,
            width=w,
            height=h
        )

        t_start = time.perf_counter()
        
        # Run Ultralytics inference
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=self.target_class_ids,
            device=self.device,
            verbose=False
        )
        
        inference_time_ms = (time.perf_counter() - t_start) * 1000.0
        
        detections: List[Detection] = []
        
        if results and len(results) > 0:
            boxes_obj = results[0].boxes
            if boxes_obj is not None and len(boxes_obj) > 0:
                xyxy = boxes_obj.xyxy.cpu().numpy()
                confs = boxes_obj.conf.cpu().numpy()
                classes = boxes_obj.cls.cpu().numpy().astype(int)
                
                for box_coord, conf, cls_id in zip(xyxy, confs, classes):
                    class_name = self.names.get(cls_id, f"class_{cls_id}")
                    bbox = BoundingBox(
                        x1=float(box_coord[0]),
                        y1=float(box_coord[1]),
                        x2=float(box_coord[2]),
                        y2=float(box_coord[3])
                    )
                    detections.append(Detection(
                        box=bbox,
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=class_name
                    ))

        return FrameDetectionResult(
            metadata=meta,
            detections=detections,
            inference_time_ms=inference_time_ms
        )

    def annotate(
        self,
        frame: np.ndarray,
        result: FrameDetectionResult,
        show_conf: bool = True,
        show_labels: bool = True,
        line_thickness: int = 2
    ) -> np.ndarray:
        """
        Renders bounding boxes and informative labels onto a frame.
        Returns a new annotated image (leaves original untouched).
        """
        annotated = frame.copy()
        
        for det in result.detections:
            x1, y1, x2, y2 = det.box.to_int_tuple()
            color = CLASS_COLOR_PALETTE.get(det.class_name, CLASS_COLOR_PALETTE["default"])
            
            # Draw bounding box rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness, cv2.LINE_AA)
            
            if show_labels:
                label = det.class_name
                if show_conf:
                    label += f" {det.confidence:.2f}"
                if det.track_id is not None:
                    label = f"#{det.track_id} " + label
                    
                # Font metrics
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Draw label background badge
                badge_y1 = max(0, y1 - text_h - baseline - 4)
                badge_y2 = y1
                badge_x2 = min(frame.shape[1], x1 + text_w + 6)
                
                cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
                
                # Draw text in contrasting black
                cv2.putText(
                    annotated, label, (x1 + 3, y1 - 4),
                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA
                )
                
        return annotated

    def benchmark_latency(self, sample_image_path: str, num_runs: int = 10) -> Dict[str, float]:
        """Runs multiple warm inferences to accurately benchmark CPU inference FPS."""
        if not os.path.exists(sample_image_path):
            raise FileNotFoundError(f"Benchmark image not found: {sample_image_path}")
            
        img = cv2.imread(sample_image_path)
        
        # Warmup run
        _ = self.detect(img)
        
        latencies = []
        for i in range(num_runs):
            t0 = time.perf_counter()
            _ = self.detect(img, frame_id=i)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            
        lat_arr = np.array(latencies)
        mean_lat = float(np.mean(lat_arr))
        std_lat = float(np.std(lat_arr))
        fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0
        
        return {
            "mean_latency_ms": round(mean_lat, 2),
            "std_latency_ms": round(std_lat, 2),
            "fps": round(fps, 1),
            "num_runs": num_runs
        }
