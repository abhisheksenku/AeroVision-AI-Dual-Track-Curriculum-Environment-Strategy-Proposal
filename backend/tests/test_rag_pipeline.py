"""
=============================================================================
AeroVision AI - RAG Vector Database & Statutory Retrieval Integration Tests
=============================================================================
Validates Qdrant collection indexing, semantic vector similarity,
category-filtered retrieval, and RESTful API endpoint handlers.
"""

import os
import sys
import json
from fastapi import HTTPException

# Ensure workspace root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import backend.app.utils.qdrant_bootstrap
from backend.app.db.database import Base, engine, SessionLocal
from backend.app.db.models import JobRecord, ViolationEventRecord
from backend.app.services.rag_service import get_rag_service
from backend.app.schemas.rag import RAGSearchRequest
from backend.app.api.v1.rag import (
    search_regulatory_knowledge_base,
    get_mission_rag_context,
    list_indexed_statutes
)


def setup_module():
    Base.metadata.create_all(bind=engine)


def test_rag_service_and_api():
    print("=" * 70, flush=True)
    print("TEST: RAG Vector Database & Statutory Retrieval Endpoints", flush=True)
    print("=" * 70, flush=True)

    db = SessionLocal()
    rag_service = get_rag_service()

    # 1. Verify Qdrant Collection & Indexing
    statutes = rag_service.get_all_statutes()
    print(f"Total Statutes Indexed in Qdrant: {len(statutes)}", flush=True)
    assert len(statutes) >= 10, f"Expected at least 10 statutes, found {len(statutes)}"

    # 2. Test Direct Semantic Search
    query = "Vehicle traveling at high speed beyond facility limits"
    citations = rag_service.search_regulations(query=query, top_k=2)
    print(f"\nTop Citation for '{query}':", flush=True)
    print(f"  Statute  : {citations[0].statute_code}", flush=True)
    print(f"  Score    : {citations[0].relevance_score}", flush=True)
    print(f"  Category : {citations[0].category}", flush=True)
    assert citations[0].relevance_score > 0.70
    assert len(citations) == 2

    # 3. Test Category Filtering
    faa_citations = rag_service.search_regulations(query=query, top_k=2, category="FAA")
    for c in faa_citations:
        assert c.category == "FAA"
    print("✓ Category filtering by 'FAA' verified.", flush=True)

    # 4. Create dummy job with violations in SQLite
    test_job_id = "job_test_rag_01"
    existing = db.query(JobRecord).filter(JobRecord.job_id == test_job_id).first()
    if existing:
        db.delete(existing)
        db.commit()

    job = JobRecord(
        job_id=test_job_id,
        filename="perimeter_flight.mp4",
        file_path="data/uploads/perimeter_flight.mp4",
        status="COMPLETED",
        progress_percent=100,
        total_frames=120,
        processed_frames=120
    )
    db.add(job)
    db.commit()

    v1 = ViolationEventRecord(
        job_id=test_job_id,
        event_id=1,
        rule_type="SPEED_LIMIT",
        track_id=3,
        class_name="car",
        timestamp_sec=2.1,
        severity="CRITICAL",
        details="Vehicle exceeded external roadway speed limit of 50 km/h",
        location_x=120.0,
        location_y=340.0
    )
    v2 = ViolationEventRecord(
        job_id=test_job_id,
        event_id=2,
        rule_type="RESTRICTED_ZONE_INTRUSION",
        track_id=8,
        class_name="truck",
        timestamp_sec=3.5,
        severity="CRITICAL",
        details="Commercial truck breached active runway perimeter gate",
        location_x=450.0,
        location_y=510.0
    )
    db.add_all([v1, v2])
    db.commit()

    # 5. Test Route Handler: POST /api/v1/rag/search
    search_req = RAGSearchRequest(query="drone beyond visual line of sight operation", top_k=2)
    search_res = search_regulatory_knowledge_base(search_req)
    assert search_res.total_results == 2
    assert "107" in search_res.results[0].statute_code
    print("✓ Route Handler: POST /api/v1/rag/search returned valid semantic citations.", flush=True)

    # 6. Test Route Handler: GET /api/v1/rag/context/{job_id}
    ctx_res = get_mission_rag_context(job_id=test_job_id, db=db)
    assert ctx_res.job_id == test_job_id
    assert ctx_res.total_infractions == 2
    assert len(ctx_res.augmented_infractions[0].citations) > 0
    print("✓ Route Handler: GET /api/v1/rag/context/{job_id} augmented infractions with statutes.", flush=True)

    # 7. Test Route Handler: GET /api/v1/rag/statutes
    statutes_res = list_indexed_statutes()
    assert len(statutes_res) >= 10
    print("✓ Route Handler: GET /api/v1/rag/statutes returned all indexed regulatory articles.", flush=True)

    # 8. Test Error Handling: 400 on empty query, 404 on invalid job
    try:
        search_regulatory_knowledge_base(RAGSearchRequest(query=""))
        assert False, "Expected HTTPException 400"
    except HTTPException as e:
        assert e.status_code == 400
        print("✓ Empty query raised HTTPException 400 as expected.", flush=True)

    try:
        get_mission_rag_context(job_id="job_nonexistent_xyz", db=db)
        assert False, "Expected HTTPException 404"
    except HTTPException as e:
        assert e.status_code == 404
        print("✓ Non-existent job raised HTTPException 404 as expected.", flush=True)

    db.close()
    print("\n[SUCCESS] All RAG & Qdrant pipeline tests passed!", flush=True)


if __name__ == "__main__":
    setup_module()
    test_rag_service_and_api()
