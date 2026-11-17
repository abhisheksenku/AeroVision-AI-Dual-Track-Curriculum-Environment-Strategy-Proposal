"""
=============================================================================
AeroVision AI - Database Relational Models
=============================================================================
SQLAlchemy 2.0 ORM models for Video Processing Jobs, Analytics Summaries,
and Safety/Security Violation Event logs.
"""

from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import String, Float, Integer, ForeignKey, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.database import Base

class JobRecord(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    annotated_video_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    
    # State Machine: 'QUEUED', 'PROCESSING', 'COMPLETED', 'FAILED'
    status: Mapped[str] = mapped_column(String(32), default="QUEUED", index=True)
    progress_percent: Mapped[int] = mapped_column(Integer, default=0)
    total_frames: Mapped[int] = mapped_column(Integer, default=0)
    processed_frames: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    analytics: Mapped[Optional["MissionAnalyticsRecord"]] = relationship(
        back_populates="job", uselist=False, cascade="all, delete-orphan"
    )
    violations: Mapped[List["ViolationEventRecord"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )

class MissionAnalyticsRecord(Base):
    __tablename__ = "mission_analytics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.job_id"), unique=True, nullable=False)
    mission_id: Mapped[str] = mapped_column(String(64), nullable=False)
    duration_sec: Mapped[float] = mapped_column(Float, default=0.0)
    total_unique_objects: Mapped[int] = mapped_column(Integer, default=0)
    
    # JSON-encoded summaries for flexible storage
    counts_by_class_json: Mapped[str] = mapped_column(Text, default="{}")
    class_stats_json: Mapped[str] = mapped_column(Text, default="{}")
    line_crossings_json: Mapped[str] = mapped_column(Text, default="{}")
    
    total_violations: Mapped[int] = mapped_column(Integer, default=0)
    compliance_status: Mapped[str] = mapped_column(String(32), default="COMPLIANT")

    job: Mapped[JobRecord] = relationship(back_populates="analytics")

class ViolationEventRecord(Base):
    __tablename__ = "violation_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.job_id"), index=True, nullable=False)
    event_id: Mapped[int] = mapped_column(Integer, nullable=False)
    rule_type: Mapped[str] = mapped_column(String(64), nullable=False)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    class_name: Mapped[str] = mapped_column(String(64), nullable=False)
    timestamp_sec: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    details: Mapped[str] = mapped_column(Text, nullable=False)
    location_x: Mapped[float] = mapped_column(Float, nullable=False)
    location_y: Mapped[float] = mapped_column(Float, nullable=False)

    job: Mapped[JobRecord] = relationship(back_populates="violations")
