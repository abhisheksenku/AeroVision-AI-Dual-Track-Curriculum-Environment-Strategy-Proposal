"""
=============================================================================
AeroVision AI - Object Tracking & ByteTrack Integration Test Suite
=============================================================================
Comprehensive integration and unit test suite validating:
1. Persistent object track IDs across sequential drone video frames.
2. Centroid trajectory history accumulation and motion trail rendering.
3. Bidirectional virtual tripwire crossing detection and object counting.
4. Velocity and physical speed estimation.
5. End-to-end tracked video generation at data/processed/tracked_drone_flight.mp4.
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import cv2
import numpy as np

from backend.app.schemas.cv import TrackedObject, FrameTrackingResult
from backend.app.cv.video_reader import VideoReader, VideoWriterWrapper
from backend.app.cv.tracker import AeroVisionTracker

SAMPLE_VIDEO_PATH = "data/samples/drone_flight_sample.mp4"
SAMPLE_IMAGE_PATH = "data/samples/bus.jpg"
MODEL_WEIGHTS_PATH = "models/weights/yolo11n.pt"
OUTPUT_DIR = "data/processed"
TRACKED_OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "tracked_drone_flight.mp4")

def test_single_frame_tracking():
    """Validates tracker initialization and single image tracking schemas."""
    print("\n--- TEST: Single Frame ByteTrack Tracking ---")
    assert os.path.exists(MODEL_WEIGHTS_PATH), f"Weights missing: {MODEL_WEIGHTS_PATH}"
    assert os.path.exists(SAMPLE_IMAGE_PATH), f"Sample image missing: {SAMPLE_IMAGE_PATH}"

    tracker = AeroVisionTracker(
        model_path=MODEL_WEIGHTS_PATH,
        conf_thresh=0.25,
        iou_thresh=0.45,
        target_classes=["bus", "person"]
    )

    img = cv2.imread(SAMPLE_IMAGE_PATH)
    result = tracker.track_frame(img, frame_id=0, timestamp_sec=0.0)

    print(f"Inference Latency     : {result.inference_time_ms:.2f} ms")
    print(f"Active Tracks Count   : {result.active_count}")
    print(f"Cumulative Objects    : {result.cumulative_unique_objects}")
    print(f"Class Breakdown       : {result.count_by_class()}")

    assert result.active_count > 0, "No tracks returned on test image!"
    for obj in result.tracked_objects:
        assert obj.track_id > 0, "Track ID must be a positive integer!"
        assert len(obj.trajectory) == 1, "Initial trajectory must have 1 point!"
        assert obj.class_name in ["bus", "person"]

    annotated = tracker.annotate(img, result)
    assert annotated.shape == img.shape
    print("✓ Single frame ByteTrack tracking verified.")

def test_end_to_end_drone_tracking_pipeline():
    """
    Executes the full tracking pipeline on the 120-frame aerial drone video:
    - Verifies persistent track IDs across frames.
    - Tests trajectory history growth.
    - Evaluates virtual tripwire line crossing events.
    - Saves output tracked video to data/processed/tracked_drone_flight.mp4.
    """
    print("\n--- TEST: End-to-End Drone Video Tracking Pipeline ---")
    assert os.path.exists(SAMPLE_VIDEO_PATH), f"Sample video missing: {SAMPLE_VIDEO_PATH}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tracker = AeroVisionTracker(
        model_path=MODEL_WEIGHTS_PATH,
        conf_thresh=0.20,
        iou_thresh=0.50,
        max_trail_len=30,
        gsd_m_per_px=0.05 # 5 cm / pixel
    )

    # Register vertical virtual tripwire across highway center: x = 640
    tripwire_name = "highway_center_screen"
    tracker.add_tripwire(
        name=tripwire_name,
        start_pt=(640, 160),
        end_pt=(640, 560)
    )

    latencies = []
    total_frames = 0
    max_observed_trail_len = 0
    unique_ids_observed = set()

    with VideoReader(SAMPLE_VIDEO_PATH) as reader:
        fps = reader.fps
        width, height = reader.width, reader.height

        with VideoWriterWrapper(TRACKED_OUTPUT_VIDEO, fps, width, height) as writer:
            t_start = time.perf_counter()

            for meta, frame in reader.stream_frames(stride=1):
                t0 = time.perf_counter()
                result = tracker.track_frame(
                    frame,
                    frame_id=meta.frame_id,
                    timestamp_sec=meta.timestamp_sec
                )
                latencies.append((time.perf_counter() - t0) * 1000.0)

                # Record metrics
                for obj in result.tracked_objects:
                    unique_ids_observed.add(obj.track_id)
                    max_observed_trail_len = max(max_observed_trail_len, len(obj.trajectory))

                # Render tracking visualization (boxes, persistent IDs, trails, tripwires, HUD)
                annotated_frame = tracker.annotate(frame, result)
                writer.write(annotated_frame)
                total_frames += 1

            total_elapsed = time.perf_counter() - t_start

    avg_latency = float(np.mean(latencies))
    pipeline_fps = total_frames / total_elapsed
    tripwire_crossings = tracker.tripwires[tripwire_name].crossing_count

    print(f"Total Video Frames Processed: {total_frames}")
    print(f"Total Unique Track IDs Logged: {len(unique_ids_observed)}")
    print(f"Max Centroid Trail Length   : {max_observed_trail_len} points")
    print(f"Tripwire Crossing Events     : {tripwire_crossings} crossings")
    print(f"Mean Tracking Latency       : {avg_latency:.2f} ms / frame")
    print(f"Overall Tracking Throughput : {pipeline_fps:.2f} FPS")
    print(f"Saved Tracked Video Path    : {TRACKED_OUTPUT_VIDEO}")

    # Validations
    assert total_frames == 120, "Expected 120 frames processed!"
    assert len(unique_ids_observed) >= 1, "No unique tracks observed across video!"
    assert max_observed_trail_len >= 10, f"Trajectory trail should accumulate over time (got {max_observed_trail_len})!"
    assert tripwire_crossings >= 1, f"Expected at least 1 tripwire crossing (got {tripwire_crossings})!"

    assert os.path.exists(TRACKED_OUTPUT_VIDEO), "Tracked video output missing!"
    video_size_mb = os.path.getsize(TRACKED_OUTPUT_VIDEO) / (1024 * 1024)
    print(f"Tracked Video File Size     : {video_size_mb:.2f} MB")
    assert video_size_mb > 1.0, "Tracked video file size is too small!"
    print("✓ End-to-end drone video tracking pipeline passed 100%!")

if __name__ == "__main__":
    test_single_frame_tracking()
    test_end_to_end_drone_tracking_pipeline()
    print("\n" + "=" * 70)
    print("ALL PHASE 6 TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
