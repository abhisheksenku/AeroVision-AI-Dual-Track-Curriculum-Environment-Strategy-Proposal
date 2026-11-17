"""
=============================================================================
AeroVision AI - Backend Engineering Foundations: Module 02
=============================================================================
SQLAlchemy 2.0 ORM, Relational Schemas, and Transaction Lifecycles

Learning Objectives:
1. Define database entities using SQLAlchemy 2.0 DeclarativeBase.
2. Establish One-to-Many relational foreign keys (VideoJob -> Violations).
3. Manage transaction commits, rollbacks, and session closing.
4. Execute type-safe CRUD operations with the modern select() syntax.
"""

from datetime import datetime, timezone
from typing import List
from sqlalchemy import create_engine, String, Float, Integer, ForeignKey, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

# 1. Base Class Definition
class Base(DeclarativeBase):
    pass

# 2. Relational Database Models
class VideoJobRecord(Base):
    __tablename__ = "video_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="QUEUED")
    total_frames: Mapped[int] = mapped_column(Integer, default=0)
    processed_frames: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))

    # One-to-Many Relationship
    violations: Mapped[List["ViolationRecord"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )

class ViolationRecord(Base):
    __tablename__ = "violations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("video_jobs.id"), nullable=False)
    rule_type: Mapped[str] = mapped_column(String(100), nullable=False)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    timestamp_sec: Mapped[float] = mapped_column(Float, nullable=False)

    # Inverse Relationship
    job: Mapped[VideoJobRecord] = relationship(back_populates="violations")

def demonstrate_sqlalchemy_orm():
    print("=" * 70)
    print("STEP 1: SQLALCHEMY 2.0 RELATIONAL PERSISTENCE (IN-MEMORY SQLITE)")
    print("=" * 70)

    # In-memory SQLite database for testing
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    # 1. Create: Insert Video Job with child Violations
    with Session() as session:
        job = VideoJobRecord(
            filename="flight_corridor_01.mp4",
            status="PROCESSING",
            total_frames=120,
            processed_frames=60
        )
        session.add(job)
        session.flush() # Flushes to DB to generate job.id

        v1 = ViolationRecord(
            job_id=job.id,
            rule_type="SPEED_VIOLATION",
            track_id=101,
            severity="CRITICAL",
            timestamp_sec=2.4
        )
        v2 = ViolationRecord(
            job_id=job.id,
            rule_type="RESTRICTED_ZONE_INCURSION",
            track_id=102,
            severity="HIGH",
            timestamp_sec=3.1
        )
        session.add_all([v1, v2])
        session.commit()
        job_id = job.id
        print(f"Created Job #{job_id} with 2 linked violations.")

    # 2. Read: Query using modern SQLAlchemy 2.0 select()
    with Session() as session:
        stmt = select(VideoJobRecord).where(VideoJobRecord.id == job_id)
        retrieved_job = session.scalars(stmt).one()

        print(f"\nRetrieved Job Record:")
        print(f"  ID        : {retrieved_job.id}")
        print(f"  Filename  : {retrieved_job.filename}")
        print(f"  Status    : {retrieved_job.status}")
        print(f"  Violations: {len(retrieved_job.violations)} linked entities")
        for v in retrieved_job.violations:
            print(f"    - [{v.severity}] {v.rule_type} by #{v.track_id} at {v.timestamp_sec}s")

        assert retrieved_job.status == "PROCESSING"
        assert len(retrieved_job.violations) == 2

    # 3. Update: Mark job completed
    with Session() as session:
        stmt = select(VideoJobRecord).where(VideoJobRecord.id == job_id)
        job_to_update = session.scalars(stmt).one()
        job_to_update.status = "COMPLETED"
        job_to_update.processed_frames = 120
        session.commit()

    # 4. Verify Update
    with Session() as session:
        updated = session.get(VideoJobRecord, job_id)
        print(f"\nJob #{job_id} Post-Update Status: {updated.status} (Processed {updated.processed_frames}/{updated.total_frames})")
        assert updated.status == "COMPLETED"

    print("\n[SUCCESS] SQLAlchemy 2.0 ORM relationships and CRUD transactions verified.")
    print()

def main():
    demonstrate_sqlalchemy_orm()
    print("=" * 70)
    print("PHASE 8 - MODULE 02: COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
