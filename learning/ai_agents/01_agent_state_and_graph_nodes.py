"""
=============================================================================
AeroVision AI - Module 01: Agent State & Graph Nodes (LangGraph)
=============================================================================
Teaches the core primitives of LangGraph:
1. Defining typed graph state schemas with TypedDict and reducers.
2. Building deterministic node functions that update state.
3. Adding directed edges and conditional routing edges.
4. Compiling the StateGraph into an executable runnable and tracing state evolution.
=============================================================================
"""

import operator
from typing import TypedDict, Annotated, List, Dict, Any, Literal
from langgraph.graph import StateGraph, END


# =============================================================================
# 1. DEFINING AGENT STATE SCHEMA
# =============================================================================
class DroneMissionState(TypedDict):
    """
    State dictionary passed between all nodes in the incident response graph.
    Annotations specify update semantics:
    - 'operator.add' appends new events to existing logs.
    - default replaces fields with the newly returned dictionary keys.
    """
    mission_id: str
    altitude_m: float
    groundspeed_kmh: float
    active_threat: bool
    threat_severity: Literal["NORMAL", "ELEVATED", "CRITICAL"]
    detected_entities: List[str]
    system_alerts: Annotated[List[str], operator.add]
    execution_trace: Annotated[List[str], operator.add]


# =============================================================================
# 2. DEFINING COMPUTATIONAL GRAPH NODES
# =============================================================================
def telemetry_ingestion_node(state: DroneMissionState) -> Dict[str, Any]:
    """
    Node 1: Ingests raw telemetry and initializes verification flags.
    """
    trace_msg = f"Node [telemetry_ingestion]: Ingested state for {state['mission_id']}"
    alerts = []
    
    if state["groundspeed_kmh"] > 50.0:
        alerts.append(f"Kinematic Alert: Velocity {state['groundspeed_kmh']} km/h exceeds 50 km/h baseline")
    if state["altitude_m"] < 15.0:
        alerts.append(f"Aviation Alert: Low altitude {state['altitude_m']}m within urban obstacle envelope")
        
    return {
        "system_alerts": alerts,
        "execution_trace": [trace_msg]
    }


def threat_evaluation_node(state: DroneMissionState) -> Dict[str, Any]:
    """
    Node 2: Evaluates compound risk score from alerts and telemetry.
    """
    alerts = state["system_alerts"]
    active = len(alerts) > 0
    severity: Literal["NORMAL", "ELEVATED", "CRITICAL"] = "NORMAL"

    if len(alerts) >= 2 or state["groundspeed_kmh"] > 60.0:
        severity = "CRITICAL"
    elif len(alerts) == 1:
        severity = "ELEVATED"

    trace_msg = f"Node [threat_evaluation]: Calculated threat level -> {severity}"
    return {
        "active_threat": active,
        "threat_severity": severity,
        "execution_trace": [trace_msg]
    }


def critical_response_node(state: DroneMissionState) -> Dict[str, Any]:
    """
    Node 3A: Executes emergency tactical response for CRITICAL incidents.
    """
    trace_msg = "Node [critical_response]: Dispatched priority intercept drone & audible perimeter siren"
    return {
        "system_alerts": ["TACTICAL ACTION: Emergency geofence lock & ground warden alert triggered."],
        "execution_trace": [trace_msg]
    }


def standard_monitoring_node(state: DroneMissionState) -> Dict[str, Any]:
    """
    Node 3B: Logs standard observational telemetry when risk is non-critical.
    """
    trace_msg = "Node [standard_monitoring]: Flight parameters nominal or manageable; logged to audit table"
    return {
        "execution_trace": [trace_msg]
    }


# =============================================================================
# 3. CONDITIONAL ROUTER FUNCTION
# =============================================================================
def route_by_severity(state: DroneMissionState) -> Literal["critical_response", "standard_monitoring"]:
    """
    Determines next node based on dynamically computed threat severity.
    """
    if state["threat_severity"] == "CRITICAL":
        return "critical_response"
    return "standard_monitoring"


# =============================================================================
# 4. BUILDING & COMPILING THE STATE GRAPH
# =============================================================================
def build_mission_graph():
    graph = StateGraph(DroneMissionState)

    # Register nodes
    graph.add_node("telemetry_ingestion", telemetry_ingestion_node)
    graph.add_node("threat_evaluation", threat_evaluation_node)
    graph.add_node("critical_response", critical_response_node)
    graph.add_node("standard_monitoring", standard_monitoring_node)

    # Set entry point
    graph.set_entry_point("telemetry_ingestion")

    # Linear edge: Ingestion -> Threat Evaluation
    graph.add_edge("telemetry_ingestion", "threat_evaluation")

    # Conditional branching: Threat Evaluation -> (Critical Response | Standard Monitoring)
    graph.add_conditional_edges(
        "threat_evaluation",
        route_by_severity,
        {
            "critical_response": "critical_response",
            "standard_monitoring": "standard_monitoring"
        }
    )

    # Terminal edges
    graph.add_edge("critical_response", END)
    graph.add_edge("standard_monitoring", END)

    # Compile the graph
    return graph.compile()


# =============================================================================
# 5. EXECUTION & VERIFICATION
# =============================================================================
def main():
    print("=" * 70)
    print("STEP 1: COMPILING LANGGRAPH STATE GRAPH")
    print("=" * 70)
    app = build_mission_graph()
    print("✓ StateGraph compiled successfully into executable Runnable.")

    print("\n" + "=" * 70)
    print("STEP 2: RUNNING TEST SCENARIO A (CRITICAL OVERSPEED & LOW ALTITUDE)")
    print("=" * 70)
    initial_state_critical: DroneMissionState = {
        "mission_id": "MS-ALPHA-101",
        "altitude_m": 12.4,
        "groundspeed_kmh": 68.5,
        "active_threat": False,
        "threat_severity": "NORMAL",
        "detected_entities": ["car", "truck"],
        "system_alerts": [],
        "execution_trace": []
    }

    final_critical = app.invoke(initial_state_critical)
    print(f"Final Threat Severity: {final_critical['threat_severity']}")
    print("System Alerts:")
    for alert in final_critical["system_alerts"]:
        print(f"  - {alert}")
    print("\nExecution Trace History:")
    for step in final_critical["execution_trace"]:
        print(f"  -> {step}")

    assert final_critical["threat_severity"] == "CRITICAL"
    assert any("TACTICAL ACTION" in a for a in final_critical["system_alerts"])

    print("\n" + "=" * 70)
    print("STEP 3: RUNNING TEST SCENARIO B (NOMINAL RECON FLIGHT)")
    print("=" * 70)
    initial_state_nominal: DroneMissionState = {
        "mission_id": "MS-BETA-202",
        "altitude_m": 45.0,
        "groundspeed_kmh": 28.0,
        "active_threat": False,
        "threat_severity": "NORMAL",
        "detected_entities": ["car"],
        "system_alerts": [],
        "execution_trace": []
    }

    final_nominal = app.invoke(initial_state_nominal)
    print(f"Final Threat Severity: {final_nominal['threat_severity']}")
    print(f"Total Alerts: {len(final_nominal['system_alerts'])}")
    print("Execution Trace History:")
    for step in final_nominal["execution_trace"]:
        print(f"  -> {step}")

    assert final_nominal["threat_severity"] == "NORMAL"
    assert len(final_nominal["system_alerts"]) == 0

    print("\n[SUCCESS] Module 01 Agent State & Graph Nodes verified!")


if __name__ == "__main__":
    main()
