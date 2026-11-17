"""
=============================================================================
AeroVision AI - Analytics Engine Integration Test Suite
=============================================================================
Comprehensive integration and unit test suite validating:
1. Ray-casting point-in-polygon and safety inspection rule evaluation.
2. Statistical speed aggregations, percentiles, and counts.
3. End-to-end integration: VideoReader -> Tracker -> AnalyticsEngine -> VideoWriter.
4. Structured analytics report export at data/processed/mission_analytics_summary.json.
5. Annotated analytics video export at data/processed/analytics_drone_flight.mp4.
"""

import os
import sys
import json
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import cv2
import numpy as np

from backend.app.schemas.cv import BoundingBox, TrackedObject
from backend.app.schemas.analytics import DroneMissionAnalyticsSummary
from backend.app.cv.video_reader import VideoReader, VideoWriterWrapper
from backend.app.cv.tracker import AeroVisionTracker
from backend.app.analytics.rules import point_in_polygon, RestrictedZone, SpeedLimitRule
from backend.app.analytics.engine import AeroVisionAnalyticsEngine

SAMPLE_VIDEO_PATH = "data/samples/drone_flight_sample.mp4"
MODEL_WEIGHTS_PATH = "models/weights/yolo11n.pt"
OUTPUT_DIR = "data/processed"
ANALYTICS_OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "analytics_drone_flight.mp4")
ANALYTICS_SUMMARY_JSON = os.path.join(OUTPUT_DIR, "mission_analytics_summary.json")

def test_point_in_polygon_and_rules():
    """Validates geometric point-in-polygon logic and individual rule evaluations."""
    print("\n--- TEST: Point-in-Polygon & Rules Logic ---")
    poly = np.array([
        [100.0, 100.0],
        [400.0, 100.0],
        [400.0, 300.0],
        [100.0, 300.0]
    ])

    assert point_in_polygon((250.0, 200.0), poly) is True
    assert point_in_polygon((50.0, 50.0), poly) is False
    assert point_in_polygon((450.0, 200.0), poly) is False

    # Test Speed Rule
    speed_rule = SpeedLimitRule(speed_limit_kmh=40.0)
    slow_obj = TrackedObject(
        track_id=1, box=BoundingBox(10, 10, 50, 50), confidence=0.9, class_id=2,
        class_name="car", speed_kmh=35.0, frames_active=5
    )
    fast_obj = TrackedObject(
        track_id=2, box=BoundingBox(10, 10, 50, 50), confidence=0.9, class_id=2,
        class_name="car", speed_kmh=52.0, frames_active=5
    )
    assert speed_rule.evaluate(slow_obj) is None
    assert speed_rule.evaluate(fast_obj) is not None
    severity, detail = speed_rule.evaluate(fast_obj)
    assert severity == "HIGH"
    assert "exceeded limit" in detail

    # Test Restricted Zone
    zone = RestrictedZone("CONSTRUCTION_AREA", [(0, 0), (200, 0), (200, 200), (0, 200)], allowed_classes={"person"})
    assert zone.is_incursion((100, 100), "car") is True       # Car not allowed
    assert zone.is_incursion((100, 100), "person") is False   # Person allowed
    assert zone.is_incursion((300, 300), "car") is False      # Outside zone
    print("✓ Point-in-polygon and individual inspection rules verified.")

def test_end_to_end_analytics_pipeline():
    """
    Executes full pipeline integration on the 120-frame drone flight video:
    VideoReader -> ByteTrack Tracker -> AnalyticsEngine -> VideoWriter
    """
    print("\n--- TEST: End-to-End Analytics Pipeline ---")
    assert os.path.exists(SAMPLE_VIDEO_PATH), f"Video missing: {SAMPLE_VIDEO_PATH}"
    assert os.path.exists(MODEL_WEIGHTS_PATH), f"Weights missing: {MODEL_WEIGHTS_PATH}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize tracker
    tracker = AeroVisionTracker(
        model_path=MODEL_WEIGHTS_PATH,
        conf_thresh=0.20,
        iou_thresh=0.50,
        max_trail_len=30,
        gsd_m_per_px=0.05
    )
    tracker.add_tripwire("highway_center_screen", (640, 160), (640, 560))

    # Initialize analytics engine with safety rules
    engine = AeroVisionAnalyticsEngine(
        mission_id="AEROVISION_UAV_MISSION_ALPHA",
        speed_limit_kmh=35.0 # Speed limit of 35 km/h
    )

    # Define restricted pedestrian corridor on the sidewalk: y in [540, 680]
    # Unauthorized vehicles are prohibited in the pedestrian walkway!
    engine.add_restricted_zone(
        name="PEDESTRIAN_SIDEWALK_WALKWAY",
        polygon=[(0.0, 540.0), (1280.0, 540.0), (1280.0, 680.0), (0.0, 680.0)],
        allowed_classes={"person"}, # Only pedestrians permitted
        severity="HIGH",
        color_bgr=(0, 0, 255)
    )

    total_frames = 0
    total_violations_triggered = 0

    with VideoReader(SAMPLE_VIDEO_PATH) as reader:
        fps = reader.fps
        width, height = reader.width, reader.height

        with VideoWriterWrapper(ANALYTICS_OUTPUT_VIDEO, fps, width, height) as writer:
            t_start = time.perf_counter()

            for meta, frame in reader.stream_frames(stride=1):
                # 1. Track objects
                tracking_res = tracker.track_frame(
                    frame,
                    frame_id=meta.frame_id,
                    timestamp_sec=meta.timestamp_sec
                )

                # 2. Evaluate analytics rules
                new_alerts = engine.process_frame(tracking_res)
                total_violations_triggered += len(new_alerts)

                # 3. Render visual tracking + analytics overlay
                annotated = tracker.annotate(frame, tracking_res, draw_trails=True, draw_tripwires=True)
                final_frame = engine.annotate_analytics(annotated, active_alerts=new_alerts)

                writer.write(final_frame)
                total_frames += 1

            total_elapsed = time.perf_counter() - t_start

    # 4. Generate structured mission report
    summary = engine.generate_summary()
    summary_dict = summary.to_dict()

    with open(ANALYTICS_SUMMARY_JSON, "w") as f:
        json.dump(summary_dict, f, indent=2)

    print(f"Total Video Frames Processed: {total_frames}")
    print(f"Total Unique Objects Logged : {summary.total_unique_objects}")
    print(f"Counts by Class             : {summary.counts_by_class}")
    print(f"Total Violations Triggered  : {summary.total_violations}")
    print(f"Mission Compliance Status   : {summary.compliance_status}")
    print(f"Pipeline Execution Time     : {total_elapsed:.2f} s ({total_frames/total_elapsed:.2f} FPS)")
    print(f"Exported Summary JSON       : {ANALYTICS_SUMMARY_JSON}")
    print(f"Exported Analytics Video    : {ANALYTICS_OUTPUT_VIDEO}")

    # Validations
    assert total_frames == 120, "Expected 120 frames processed!"
    assert summary.total_unique_objects > 0, "No unique objects recorded in summary!"
    assert os.path.exists(ANALYTICS_SUMMARY_JSON), "Analytics summary JSON not found!"
    assert os.path.exists(ANALYTICS_OUTPUT_VIDEO), "Analytics video not found!"

    # Ensure JSON has expected keys and content
    with open(ANALYTICS_SUMMARY_JSON, "r") as f:
        loaded_json = json.load(f)
        assert loaded_json["mission_id"] == "AEROVISION_UAV_MISSION_ALPHA"
        assert "class_statistics" in loaded_json
        assert "safety_audit" in loaded_json

    video_size_mb = os.path.getsize(ANALYTICS_OUTPUT_VIDEO) / (1024 * 1024)
    print(f"Analytics Video File Size   : {video_size_mb:.2f} MB")
    assert video_size_mb > 1.0, "Video file size unexpectedly small!"
    print("✓ End-to-end analytics pipeline passed 100%!")

if __name__ == "__main__":
    test_point_in_polygon_and_rules()
    test_end_to_end_analytics_pipeline()
    print("\n" + "=" * 70)
    print("ALL PHASE 7 TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
