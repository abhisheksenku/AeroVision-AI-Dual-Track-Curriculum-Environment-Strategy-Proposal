"""
=============================================================================
AeroVision AI - Module 03: Supervisor Multi-Agent System (LangGraph)
=============================================================================
Teaches the Supervisor Multi-Agent Pattern:
1. Orchestrating specialized worker agents via a centralized Lead Incident Commander.
2. Managing inter-agent state aggregation and specialist handoffs.
3. Assembling multi-disciplinary intelligence (Kinematics, Statutory Law, Tactical Defense).
4. Emitting a unified, prioritized Incident Action Plan (IAP).
=============================================================================
"""

import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, END


# =============================================================================
# 1. SHARED MULTI-AGENT STATE SCHEMA
# =============================================================================
class MultiAgentMissionState(TypedDict):
    mission_id: str
    incident_type: str
    raw_telemetry: Dict[str, Any]
    next_worker: Optional[Literal["kinematics_specialist", "compliance_officer", "tactical_planner", "FINISH"]]
    kinematics_report: Optional[str]
    compliance_report: Optional[str]
    tactical_orders: Optional[List[str]]
    final_action_plan: Optional[str]
    execution_trace: Annotated[List[str], operator.add]


# =============================================================================
# 2. SPECIALIST WORKER AGENTS
# =============================================================================
def kinematics_specialist_node(state: MultiAgentMissionState) -> Dict[str, Any]:
    """
    Worker 1: Specialized in vector kinematics, velocity thresholds, and trajectory dynamics.
    """
    telemetry = state["raw_telemetry"]
    speed = telemetry.get("speed_kmh", 0.0)
    accel = telemetry.get("acceleration_ms2", 0.0)
    dwell = telemetry.get("dwell_time_sec", 0.0)

    report = (
        f"KINEMATICS REPORT: Entity observed at {speed:.1f} km/h (Peak threshold: 50.0 km/h). "
        f"Radial acceleration: {accel:+.2f} m/s^2. Loiter dwell time: {dwell:.1f}s within secure sector."
    )
    return {
        "kinematics_report": report,
        "execution_trace": ["Worker [KinematicsSpecialist]: Completed motion & trajectory audit."]
    }


def compliance_officer_node(state: MultiAgentMissionState) -> Dict[str, Any]:
    """
    Worker 2: Specialized in aviation and industrial perimeter statutory codes.
    """
    report = (
        "COMPLIANCE REPORT: Violation of FAA 14 CFR § 107.51 (groundspeed exceeds operational limit) "
        "and NIST SP 800-82 § A.11 (Active perimeter zone unauthorized infiltration). Liability: HIGH."
    )
    return {
        "compliance_report": report,
        "execution_trace": ["Worker [ComplianceOfficer]: Formulated legal citations and liability profile."]
    }


def tactical_planner_node(state: MultiAgentMissionState) -> Dict[str, Any]:
    """
    Worker 3: Specialized in security protocols, containment, and drone surveillance redirection.
    """
    orders = [
        "1. Lock Sector 4 perimeter security turnstiles and vehicle barriers.",
        "2. Direct Drone Alpha to lock optical gimbal tracking on Track #3.",
        "3. Sound localized audible warning siren at Gate Bravo.",
        "4. Dispatch ground rapid response unit via West Access Road."
    ]
    return {
        "tactical_orders": orders,
        "execution_trace": ["Worker [TacticalPlanner]: Generated 4-step emergency containment SOP."]
    }


# =============================================================================
# 3. SUPERVISOR (INCIDENT COMMANDER) ORCHESTRATOR
# =============================================================================
def supervisor_node(state: MultiAgentMissionState) -> Dict[str, Any]:
    """
    Lead Commander: Assesses current state and decides which specialist to invoke next,
    or compiles the final comprehensive Incident Action Plan (IAP) once all specialists finish.
    """
    trace = []
    
    # Check what reports are still missing
    if not state.get("kinematics_report"):
        trace.append("Supervisor: Dispatching task to Kinematics Specialist.")
        return {"next_worker": "kinematics_specialist", "execution_trace": trace}

    if not state.get("compliance_report"):
        trace.append("Supervisor: Dispatching task to Statutory Compliance Officer.")
        return {"next_worker": "compliance_officer", "execution_trace": trace}

    if not state.get("tactical_orders"):
        trace.append("Supervisor: Dispatching task to Tactical Containment Planner.")
        return {"next_worker": "tactical_planner", "execution_trace": trace}

    # All three specialist reports are assembled! Synthesize Incident Action Plan.
    trace.append("Supervisor: All specialist inputs received. Compiling final Incident Action Plan.")
    
    iap = (
        f"=== AEROVISION INCIDENT ACTION PLAN (IAP) ===\n"
        f"Mission ID    : {state['mission_id']}\n"
        f"Incident Type : {state['incident_type']}\n\n"
        f"1. Telemetry Intelligence:\n   {state['kinematics_report']}\n\n"
        f"2. Statutory Legal Grounding:\n   {state['compliance_report']}\n\n"
        f"3. Tactical Directives:\n"
    )
    for order in state["tactical_orders"]:
        iap += f"   {order}\n"

    return {
        "next_worker": "FINISH",
        "final_action_plan": iap,
        "execution_trace": trace
    }


def supervisor_router(state: MultiAgentMissionState) -> Literal["kinematics_specialist", "compliance_officer", "tactical_planner", "FINISH"]:
    """
    Directs the graph edge according to supervisor's routing command.
    """
    return state.get("next_worker", "FINISH")


# =============================================================================
# 4. GRAPH COMPILATION
# =============================================================================
def build_supervisor_graph():
    graph = StateGraph(MultiAgentMissionState)

    # Register all nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("kinematics_specialist", kinematics_specialist_node)
    graph.add_node("compliance_officer", compliance_officer_node)
    graph.add_node("tactical_planner", tactical_planner_node)

    # Entry point is always supervisor
    graph.set_entry_point("supervisor")

    # Conditional routing from supervisor to workers or finish
    graph.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "kinematics_specialist": "kinematics_specialist",
            "compliance_officer": "compliance_officer",
            "tactical_planner": "tactical_planner",
            "FINISH": END
        }
    )

    # Each worker reports back directly to supervisor
    graph.add_edge("kinematics_specialist", "supervisor")
    graph.add_edge("compliance_officer", "supervisor")
    graph.add_edge("tactical_planner", "supervisor")

    return graph.compile()


# =============================================================================
# 5. EXECUTION & VERIFICATION
# =============================================================================
def main():
    print("=" * 70)
    print("STEP 1: COMPILING SUPERVISOR MULTI-AGENT STATE GRAPH")
    print("=" * 70)
    app = build_supervisor_graph()
    print("✓ Multi-agent StateGraph compiled successfully.")

    print("\n" + "=" * 70)
    print("STEP 2: DISPATCHING SUPERVISOR ON HIGH-RISK PERIMETER INTRUSION")
    print("=" * 70)
    initial_state: MultiAgentMissionState = {
        "mission_id": "MS-PERIMETER-990",
        "incident_type": "HIGH_SPEED_RUNWAY_INTRUSION",
        "raw_telemetry": {
            "speed_kmh": 72.8,
            "acceleration_ms2": 1.45,
            "dwell_time_sec": 34.0,
            "track_id": 9
        },
        "next_worker": None,
        "kinematics_report": None,
        "compliance_report": None,
        "tactical_orders": None,
        "final_action_plan": None,
        "execution_trace": []
    }

    final_state = app.invoke(initial_state)

    print("\nOrchestration Trace:")
    for step in final_state["execution_trace"]:
        print(f"  -> {step}")

    print("\n" + final_state["final_action_plan"])

    # Verifications
    assert final_state["kinematics_report"] is not None
    assert final_state["compliance_report"] is not None
    assert final_state["tactical_orders"] is not None
    assert len(final_state["tactical_orders"]) == 4
    assert "AEROVISION INCIDENT ACTION PLAN" in final_state["final_action_plan"]
    print("[SUCCESS] Module 03 Supervisor Multi-Agent System verified!")


if __name__ == "__main__":
    main()
