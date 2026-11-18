"""
=============================================================================
AeroVision AI - Mission AI Report Pydantic Schemas
=============================================================================
Data contracts for Gemini multimodal aerial compliance briefings and grounding audits.
"""

from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field


class IncidentDetail(BaseModel):
    track_id: int = Field(description="Unique tracking ID of entity")
    class_name: str = Field(description="Detected classification (car, bus, etc.)")
    rule_type: str = Field(description="Infraction rule category")
    severity: Literal["WARNING", "CRITICAL"] = Field(description="Severity flag")
    narrative: str = Field(description="Contextual explanation of risk and observed kinematics")


class GroundingAuditResult(BaseModel):
    grounding_score: float = Field(description="Composite grounding fidelity (0.0 to 1.0)")
    verdict: Literal["VERIFIED_GROUNDED", "MINOR_DEVIATION", "UNRELIABLE_HALLUCINATION"]
    entity_score: float
    speed_score: float
    incident_score: float
    hallucinated_classes: List[str] = Field(default_factory=list)
    speed_delta_kmh: float = 0.0


class MissionComplianceReport(BaseModel):
    mission_id: str
    job_id: str
    executive_summary: str
    threat_level: Literal["LOW", "ELEVATED", "CRITICAL"]
    compliance_status: Literal["COMPLIANT", "AUDIT_FLAGGED"]
    total_tracked_objects: int
    peak_speed_kmh: float
    incidents: List[IncidentDetail] = Field(default_factory=list)
    operational_recommendations: List[str] = Field(default_factory=list)
    grounding_audit: GroundingAuditResult
    model_used: str
    generated_at: str

    class Config:
        from_attributes = True


class GenerateReportRequest(BaseModel):
    model_name: Optional[str] = "gemini-2.5-flash"
    temperature: Optional[float] = 0.2
