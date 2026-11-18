"""
=============================================================================
AeroVision AI - Autonomous AI Agent Schemas (LangGraph)
=============================================================================
Pydantic contracts for multi-agent incident response graphs, tactical action plans,
agent execution traces, and interactive operator chat dialogues.
"""

from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field


class TacticalDirective(BaseModel):
    step_number: int = Field(description="Order of execution in containment sequence")
    action: str = Field(description="Actionable instruction for drone or security personnel")
    target_unit: str = Field(description="Responsible entity: DRONE_GIMBAL, PERIMETER_GATE, GROUND_SECURITY")
    priority: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(default="HIGH")


class IncidentActionPlan(BaseModel):
    mission_id: str
    job_id: str
    incident_type: str
    threat_level: Literal["LOW", "ELEVATED", "CRITICAL"]
    kinematics_summary: str
    statutory_citations: List[str]
    containment_radius_m: float
    tactical_directives: List[TacticalDirective]
    grounding_score: float
    grounding_verdict: Literal["VERIFIED_GROUNDED", "MINOR_DEVIATION", "UNRELIABLE_HALLUCINATION"]
    generated_at: str


class AgentStepTrace(BaseModel):
    step_index: int
    node_name: str
    action_summary: str
    details: Optional[Dict[str, Any]] = None


class AutonomousResponseResult(BaseModel):
    job_id: str
    status: Literal["SUCCESS", "FAILED"]
    action_plan: IncidentActionPlan
    execution_trace: List[AgentStepTrace]
    iterations_completed: int


class TriggerIncidentResponseRequest(BaseModel):
    urgency_level: Optional[Literal["STANDARD", "EMERGENCY"]] = "EMERGENCY"
    custom_instruction: Optional[str] = None


class AgentChatRequest(BaseModel):
    message: str = Field(description="Operator prompt or question regarding mission telemetry")
    job_id: Optional[str] = Field(default=None, description="Optional job ID context")


class AgentChatResponse(BaseModel):
    reply: str
    tools_invoked: List[str]
    data_grounding: Dict[str, Any]
