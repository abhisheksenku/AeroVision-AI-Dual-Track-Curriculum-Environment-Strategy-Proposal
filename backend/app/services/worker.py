"""
=============================================================================
AeroVision AI - Asynchronous Computer Vision Worker Service
=============================================================================
Executes full drone video analysis (Decoding -> ByteTrack -> Analytics -> Encoding)
in a background thread pool, persisting state and progress to SQLite.
"""

import os
import json
import logging
from datetime import datetime, timezone
import numpy as np

from backend.app.db.database import SessionLocal
from backend.app.db.models import JobRecord, MissionAnalyticsRecord, ViolationEventRecord
from backend.app.cv.video_reader import VideoReader, VideoWriterWrapper
from backend.app.cv.tracker import AeroVisionTracker
from backend.app.analytics.engine import AeroVisionAnalyticsEngine

logger = logging.getLogger("aerovision.worker")

def process_video_job(job_id: str, model_path: str = "models/weights/yolo11n.pt"):
    """
    Background worker function executing the full CV inference & analytics pipeline.
    """
    session = SessionLocal()
    try:
        job = session.query(JobRecord).filter(JobRecord.job_id == job_id).first()
        if not job:
            logger.error(f"Job not found for ID: {job_id}")
            return

        job.status = "PROCESSING"
        job.progress_percent = 0
        session.commit()

        input_path = job.file_path
        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)
        output_video_path = os.path.join(output_dir, f"{job_id}_annotated.mp4")

        # 1. Initialize Video Reader
        with VideoReader(input_path) as reader:
            total_frames = max(1, reader.total_frames)
            fps = reader.fps
            width, height = reader.width, reader.height

            job.total_frames = total_frames
            session.commit()

            # 2. Initialize Tracker & Analytics Engine
            tracker = AeroVisionTracker(
                model_path=model_path,
                conf_thresh=0.20,
                iou_thresh=0.50,
                max_trail_len=30,
                gsd_m_per_px=0.05
            )
            tracker.add_tripwire("highway_center_screen", (width // 2, int(height * 0.2)), (width // 2, int(height * 0.8)))

            engine = AeroVisionAnalyticsEngine(
                mission_id=f"MISSION_{job_id}",
                speed_limit_kmh=35.0
            )
            # Default restricted zone (sidewalk walkway)
            engine.add_restricted_zone(
                name="PEDESTRIAN_SIDEWALK",
                polygon=[(0.0, float(height * 0.75)), (float(width), float(height * 0.75)),
                         (float(width), float(height * 0.95)), (0.0, float(height * 0.95))],
                allowed_classes={"person"},
                severity="HIGH",
                color_bgr=(0, 0, 220)
            )

            # 3. Stream & Process Frames
            with VideoWriterWrapper(output_video_path, fps, width, height) as writer:
                processed_count = 0

                for meta, frame in reader.stream_frames(stride=1):
                    # Track objects
                    tracking_res = tracker.track_frame(
                        frame,
                        frame_id=meta.frame_id,
                        timestamp_sec=meta.timestamp_sec
                    )

                    # Evaluate analytics rules
                    new_alerts = engine.process_frame(tracking_res)

                    # Render visualizations
                    annotated = tracker.annotate(frame, tracking_res, draw_trails=True, draw_tripwires=True)
                    final_frame = engine.annotate_analytics(annotated, active_alerts=new_alerts)
                    writer.write(final_frame)

                    processed_count += 1

                    # Update progress in database every 15 frames or at the end
                    if processed_count % 15 == 0 or processed_count == total_frames:
                        job.processed_frames = processed_count
                        job.progress_percent = int((processed_count / total_frames) * 100.0)
                        session.commit()

            # 4. Generate & Persist Mission Summary
            summary = engine.generate_summary()
            summary_dict = summary.to_dict()

            analytics_record = MissionAnalyticsRecord(
                job_id=job.job_id,
                mission_id=summary.mission_id,
                duration_sec=summary.duration_sec,
                total_unique_objects=summary.total_unique_objects,
                counts_by_class_json=json.dumps(summary.counts_by_class),
                class_stats_json=json.dumps({k: v.to_dict() for k, v in summary.class_statistics.items()}),
                line_crossings_json=json.dumps(summary.line_crossings),
                total_violations=summary.total_violations,
                compliance_status=summary.compliance_status
            )
            session.add(analytics_record)

            # Save individual violation records
            for v in summary.violations_log:
                v_record = ViolationEventRecord(
                    job_id=job.job_id,
                    event_id=v.event_id,
                    rule_type=v.rule_type,
                    track_id=v.track_id,
                    class_name=v.class_name,
                    timestamp_sec=v.timestamp_sec,
                    severity=v.severity,
                    details=v.details,
                    location_x=v.location[0],
                    location_y=v.location[1]
                )
                session.add(v_record)

            # Mark Job Completed
            job.status = "COMPLETED"
            job.progress_percent = 100
            job.annotated_video_path = output_video_path
            job.completed_at = datetime.now(timezone.utc)
            session.commit()
            logger.info(f"Successfully processed job {job_id}: {total_frames} frames, {summary.total_violations} violations.")

    except Exception as e:
        logger.error(f"Error processing video job {job_id}: {e}", exc_info=True)
        session.rollback()
        job = session.query(JobRecord).filter(JobRecord.job_id == job_id).first()
        if job:
            job.status = "FAILED"
            job.error_message = str(e)
            session.commit()
    finally:
        session.close()
