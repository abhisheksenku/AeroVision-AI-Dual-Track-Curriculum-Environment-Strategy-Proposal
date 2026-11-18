"""
=============================================================================
AeroVision AI - RAG Vector Search & Regulatory Context REST API Routes
=============================================================================
Endpoints for querying Qdrant vector database and retrieving augmented statutory citations.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.db.models import JobRecord, ViolationEventRecord
from backend.app.schemas.rag import (
    RAGSearchRequest,
    RAGSearchResponse,
    RegulatoryCitation,
    MissionRAGContextResponse,
    AugmentedInfraction
)
from backend.app.services.rag_service import get_rag_service

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/search", response_model=RAGSearchResponse, status_code=status.HTTP_200_OK)
def search_regulatory_knowledge_base(request: RAGSearchRequest):
    """
    Executes semantic vector similarity search against statutory regulations in Qdrant.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query must not be empty.")

    rag_service = get_rag_service()
    citations = rag_service.search_regulations(
        query=request.query,
        top_k=request.top_k or 3,
        category=request.category
    )

    return RAGSearchResponse(
        query=request.query,
        total_results=len(citations),
        results=citations
    )


@router.get("/context/{job_id}", response_model=MissionRAGContextResponse, status_code=status.HTTP_200_OK)
def get_mission_rag_context(job_id: str, db: Session = Depends(get_db)):
    """
    Retrieves all registered infractions for a mission, dynamically augmented
    with governing statutory legal references retrieved via Qdrant vector search.
    """
    job = db.query(JobRecord).filter(JobRecord.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    violations = db.query(ViolationEventRecord).filter(ViolationEventRecord.job_id == job_id).all()
    rag_service = get_rag_service()
    augmented = rag_service.augment_violations(violations)

    return MissionRAGContextResponse(
        job_id=job_id,
        total_infractions=len(augmented),
        augmented_infractions=augmented
    )


@router.get("/statutes", response_model=List[Dict[str, Any]], status_code=status.HTTP_200_OK)
def list_indexed_statutes():
    """
    Lists all indexed statutory articles and regulatory codes present in the Qdrant collection.
    """
    rag_service = get_rag_service()
    statutes = rag_service.get_all_statutes()
    return statutes
