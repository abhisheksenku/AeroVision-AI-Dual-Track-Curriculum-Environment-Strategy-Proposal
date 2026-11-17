"""
=============================================================================
AeroVision AI - Computer Vision & YOLO11 Pipeline Test Suite
=============================================================================
Comprehensive integration and unit test suite validating:
1. Strongly-typed CV schemas (BoundingBox, Detection, FrameDetectionResult).
2. Memory-safe VideoReader with O(1) RAM streaming and stride sampling.
3. Letterbox aspect-ratio preserving spatial transformation.
4. Production YOLO11Detector inference, class filtering, and bounding box annotations.
5. End-to-end drone video detection pipeline, generating annotated output video.
6. CPU latency and memory RSS profiling.
"""

import os
import sys
import time

# Ensure project root is in Python sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import psutil
import numpy as np
import cv2

from backend.app.schemas.cv import (
    BoundingBox,
    Detection,
    FrameMetadata,
    FrameDetectionResult
)
from backend.app.cv.video_reader import VideoReader, letterbox, VideoWriterWrapper
from backend.app.cv.detector import YOLO11Detector

SAMPLE_VIDEO_PATH = "data/samples/drone_flight_sample.mp4"
SAMPLE_IMAGE_PATH = "data/samples/bus.jpg"
MODEL_WEIGHTS_PATH = "models/weights/yolo11n.pt"
OUTPUT_DIR = "data/processed"
ANNOTATED_OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "annotated_drone_flight.mp4")

def test_cv_schemas():
    """Validates dataclass schema invariants, conversions, and serialization."""
    print("\n--- TEST: CV Schemas ---")
    bbox = BoundingBox(x1=100.5, y1=150.0, x2=300.5, y2=450.0)
    assert bbox.width == 200.0
    assert bbox.height == 300.0
    assert bbox.area == 60000.0
    assert bbox.center == (200.5, 300.0)
    assert bbox.to_int_tuple() == (100, 150, 300, 450)

    det = Detection(box=bbox, confidence=0.925, class_id=2, class_name="car")
    det_dict = det.to_dict()
    assert det_dict["class_name"] == "car"
    assert det_dict["confidence"] == 0.925
    assert det_dict["box"]["width"] == 200.0

    meta = FrameMetadata(frame_id=1, timestamp_sec=0.033, width=1280, height=720)
    frame_res = FrameDetectionResult(metadata=meta, detections=[det], inference_time_ms=45.2)
    assert frame_res.count == 1
    assert frame_res.count_by_class() == {"car": 1}
    assert frame_res.get_boxes_xyxy().shape == (1, 4)
    print("✓ CV schemas verified.")

def test_video_reader_metadata():
    """Validates video reader initialization and metadata parsing."""
    print("\n--- TEST: VideoReader Metadata ---")
    assert os.path.exists(SAMPLE_VIDEO_PATH), f"Sample video missing: {SAMPLE_VIDEO_PATH}"
    
    with VideoReader(SAMPLE_VIDEO_PATH) as reader:
        meta = reader.metadata
        print(f"Video Metadata: {meta}")
        assert meta["resolution"] == [1280, 720]
        assert meta["fps"] == 30.0
        assert meta["total_frames"] == 120
        assert meta["duration_sec"] == 4.0
    print("✓ VideoReader metadata verified.")

def test_video_reader_o1_memory_streaming():
    """
    Validates that streaming 120 frames uses O(1) memory without RAM leakage.
    Contrast with naive frame loading which would allocate 120 * 1280 * 720 * 3 = 331.7 MB.
    """
    import gc
    print("\n--- TEST: VideoReader O(1) Memory Streaming ---")
    process = psutil.Process(os.getpid())

    # Warmup OpenCV MSMF decoder backend (loads native Windows media libraries)
    with VideoReader(SAMPLE_VIDEO_PATH) as warmup_reader:
        for _ in warmup_reader.stream_frames(stride=1, max_frames=5):
            pass

    gc.collect()
    mem_before_mb = process.memory_info().rss / (1024 * 1024)

    frame_count = 0
    with VideoReader(SAMPLE_VIDEO_PATH) as reader:
        for meta, frame in reader.stream_frames(stride=1):
            frame_count += 1
            assert frame.shape == (720, 1280, 3)
            assert meta.frame_id == frame_count - 1

    gc.collect()
    mem_after_mb = process.memory_info().rss / (1024 * 1024)
    mem_delta_mb = abs(mem_after_mb - mem_before_mb)

    print(f"Total Streamed Frames : {frame_count}")
    print(f"RAM Before Streaming  : {mem_before_mb:.2f} MB")
    print(f"RAM After Streaming   : {mem_after_mb:.2f} MB")
    print(f"Memory Delta (RSS)    : {mem_delta_mb:.2f} MB (vs ~332 MB for naive list accumulation)")
    
    assert frame_count == 120
    assert mem_delta_mb < 35.0, f"Memory leaked during video streaming! Delta: {mem_delta_mb} MB"
    print("✓ O(1) constant memory streaming verified.")

def test_letterbox_preprocessing():
    """Validates aspect-ratio preserving letterboxing."""
    print("\n--- TEST: Letterbox Preprocessing ---")
    test_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    target_size = (640, 640)
    
    padded, r, (dw, dh) = letterbox(test_img, target_size=target_size)
    assert padded.shape == (640, 640, 3)
    assert r == 640.0 / 1280.0 # 0.5
    assert dw == 0.0
    assert dh == (640 - (720 * 0.5)) / 2.0 # 140.0
    print(f"Letterbox Output Shape: {padded.shape}, Scale: {r}, Padding: ({dw}, {dh})")
    print("✓ Letterbox preprocessing verified.")

def test_yolo_detector_image_inference():
    """Validates YOLO11n image inference, schemas, and visual annotation."""
    print("\n--- TEST: YOLO11Detector Image Inference ---")
    assert os.path.exists(MODEL_WEIGHTS_PATH), f"Weights missing: {MODEL_WEIGHTS_PATH}"
    assert os.path.exists(SAMPLE_IMAGE_PATH), f"Sample image missing: {SAMPLE_IMAGE_PATH}"

    detector = YOLO11Detector(
        model_path=MODEL_WEIGHTS_PATH,
        conf_thresh=0.25,
        iou_thresh=0.45,
        target_classes=["bus", "person"]
    )

    img = cv2.imread(SAMPLE_IMAGE_PATH)
    result = detector.detect(img, frame_id=0, timestamp_sec=0.0)

    print(f"Inference Latency: {result.inference_time_ms:.2f} ms")
    print(f"Detections Count : {result.count}")
    print(f"Counts by Class  : {result.count_by_class()}")

    assert result.count > 0, "No detections returned on test image!"
    assert "bus" in result.count_by_class() or "person" in result.count_by_class()
    
    # Check bounding box bounds
    for det in result.detections:
        assert 0.0 <= det.box.x1 <= img.shape[1]
        assert 0.0 <= det.box.y1 <= img.shape[0]
        assert det.confidence >= 0.25
        assert det.class_name in ["bus", "person"]

    # Test visual annotation
    annotated = detector.annotate(img, result)
    assert annotated.shape == img.shape
    assert annotated.dtype == img.dtype
    print("✓ YOLO11 image detection and annotation verified.")

def test_end_to_end_drone_video_pipeline():
    """
    Executes the complete end-to-end video processing and detection pipeline:
    1. Sequentially streams 120 frames of drone flight footage.
    2. Runs YOLO11n object detector on each frame.
    3. Annotates bounding boxes, class labels, and confidence tags.
    4. Encodes and writes annotated MP4 video to disk.
    5. Measures throughput (FPS) and CPU latency.
    """
    print("\n--- TEST: End-to-End Drone Video Detection Pipeline ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    detector = YOLO11Detector(
        model_path=MODEL_WEIGHTS_PATH,
        conf_thresh=0.20,
        iou_thresh=0.45,
        target_classes=None  # Detect all objects
    )

    total_frames = 0
    total_detections = 0
    latencies = []
    
    with VideoReader(SAMPLE_VIDEO_PATH) as reader:
        fps = reader.fps
        width, height = reader.width, reader.height

        with VideoWriterWrapper(ANNOTATED_OUTPUT_VIDEO, fps, width, height) as writer:
            t_start = time.perf_counter()
            
            for meta, frame in reader.stream_frames(stride=1):
                t0 = time.perf_counter()
                result = detector.detect(frame, frame_id=meta.frame_id, timestamp_sec=meta.timestamp_sec)
                latencies.append((time.perf_counter() - t0) * 1000.0)

                annotated_frame = detector.annotate(frame, result)
                writer.write(annotated_frame)

                total_frames += 1
                total_detections += result.count

            total_elapsed = time.perf_counter() - t_start

    avg_latency = float(np.mean(latencies))
    pipeline_fps = total_frames / total_elapsed

    print(f"Total Video Frames Processed: {total_frames}")
    print(f"Total Detections Logged     : {total_detections}")
    print(f"Mean Detection Latency      : {avg_latency:.2f} ms / frame")
    print(f"Overall Pipeline Throughput : {pipeline_fps:.2f} FPS")
    print(f"Output Annotated Video Path : {ANNOTATED_OUTPUT_VIDEO}")

    # Verify generated video exists and has non-zero size
    assert os.path.exists(ANNOTATED_OUTPUT_VIDEO), "Annotated video not generated!"
    video_size_mb = os.path.getsize(ANNOTATED_OUTPUT_VIDEO) / (1024 * 1024)
    print(f"Annotated Video File Size   : {video_size_mb:.2f} MB")
    assert video_size_mb > 1.0, "Annotated video size is too small!"
    print("✓ End-to-end drone video detection pipeline passed 100%!")

if __name__ == "__main__":
    test_cv_schemas()
    test_video_reader_metadata()
    test_video_reader_o1_memory_streaming()
    test_letterbox_preprocessing()
    test_yolo_detector_image_inference()
    test_end_to_end_drone_video_pipeline()
    print("\n" + "=" * 70)
    print("ALL PHASE 4 & PHASE 5 TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
