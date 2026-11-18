"""
=============================================================================
AeroVision AI - RAG Vector Database Pydantic Schemas
=============================================================================
Data contracts for semantic retrieval over statutory drone & safety regulations.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class RegulatoryCitation(BaseModel):
    statute_code: str = Field(description="Legal statute code (e.g. 14 CFR § 107.51, OSHA 1910.178)")
    section_title: str = Field(description="Title of the statutory section")
    category: str = Field(description="Regulatory authority / domain (FAA, OSHA, PERIMETER_SECURITY)")
    relevance_score: float = Field(description="Cosine similarity relevance (0.0 to 1.0)")
    text: str = Field(description="Excerpt of requirement and enforcement provisions")


class RAGSearchRequest(BaseModel):
    query: str = Field(description="Natural language query describing surveillance incident or rule")
    top_k: Optional[int] = Field(default=3, description="Number of candidate statutes to retrieve")
    category: Optional[str] = Field(default=None, description="Optional regulatory category filter")


class RAGSearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[RegulatoryCitation]


class AugmentedInfraction(BaseModel):
    track_id: int
    class_name: str
    rule_type: str
    severity: str
    details: str
    citations: List[RegulatoryCitation] = Field(default_factory=list)


class MissionRAGContextResponse(BaseModel):
    job_id: str
    total_infractions: int
    augmented_infractions: List[AugmentedInfraction]
