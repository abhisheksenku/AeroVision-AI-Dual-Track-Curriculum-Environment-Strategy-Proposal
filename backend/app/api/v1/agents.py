"""
=============================================================================
AeroVision AI - Autonomous Agent REST API Endpoints
=============================================================================
Routes for executing multi-agent incident response StateGraphs and interactive
operator telemetry dialogues via LangGraph.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.schemas.agent import (
    AutonomousResponseResult,
    TriggerIncidentResponseRequest,
    AgentChatRequest,
    AgentChatResponse
)
from backend.app.services.agent_service import AeroVisionAgentService

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/incident-response/{job_id}", response_model=AutonomousResponseResult, status_code=status.HTTP_200_OK)
def trigger_incident_response_graph(
    job_id: str,
    request: Optional[TriggerIncidentResponseRequest] = None,
    db: Session = Depends(get_db)
):
    """
    Executes the multi-agent incident response StateGraph for a completed or active flight mission.
    Orchestrates the Kinematics Specialist, Compliance Specialist (Qdrant), Tactical Planner,
    and Reflection Auditor to formulate a 100% grounded Incident Action Plan.
    """
    try:
        result = AeroVisionAgentService.execute_incident_response(job_id=job_id, db=db)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent graph execution failed: {str(e)}")


@router.post("/chat", response_model=AgentChatResponse, status_code=status.HTTP_200_OK)
def operator_agent_dialogue(
    request: AgentChatRequest,
    db: Session = Depends(get_db)
):
    """
    Provides interactive conversational agent console for flight operators,
    supporting autonomous tool invocation for kinematics, Qdrant statutory search,
    and containment perimeter calculations.
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Operator message must not be empty.")

    try:
        response = AeroVisionAgentService.chat_with_agent(
            message=request.message,
            job_id=request.job_id,
            db=db
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Operator chat dialogue failed: {str(e)}")
