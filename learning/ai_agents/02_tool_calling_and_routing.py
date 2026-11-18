"""
=============================================================================
AeroVision AI - Module 02: Tool Calling & Dynamic Routing (LangGraph)
=============================================================================
Teaches autonomous tool integration and cyclic execution loops:
1. Creating strongly-typed domain tools for drone surveillance and physics.
2. Building an Agent Reasoning Node that inspects telemetry and formulates tool calls.
3. Implementing a Tool Dispatcher Node that executes requested tools.
4. Implementing a cyclic conditional edge (ReAct pattern) routing between agent and tools.
=============================================================================
"""

import json
import math
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal
import operator
from langgraph.graph import StateGraph, END


# =============================================================================
# 1. TOOL DEFINITIONS (AEROSPACE & SURVEILLANCE DOMAIN)
# =============================================================================
def inspect_tracking_telemetry(track_id: int) -> Dict[str, Any]:
    """
    Simulates looking up deep spatial kinematics for an individual tracked entity.
    """
    database = {
        3: {
            "track_id": 3,
            "class_name": "car",
            "mean_speed_kmh": 68.4,
            "peak_speed_kmh": 74.2,
            "heading_degrees": 134.0,
            "in_restricted_zone": True,
            "dwell_time_sec": 42.0
        },
        7: {
            "track_id": 7,
            "class_name": "truck",
            "mean_speed_kmh": 32.1,
            "peak_speed_kmh": 35.0,
            "heading_degrees": 210.0,
            "in_restricted_zone": False,
            "dwell_time_sec": 12.0
        }
    }
    return database.get(track_id, {"error": f"Track ID {track_id} not found in telemetry registry."})


def query_statutory_regulations(topic: str) -> str:
    """
    Simulates querying statutory legal corpus for drone or perimeter rules.
    """
    topic_lower = topic.lower()
    if "speed" in topic_lower or "velocity" in topic_lower:
        return "14 CFR § 107.51: Drone maximum groundspeed 87 knots (100 mph); Industrial yard speed limit 50 km/h."
    elif "restricted" in topic_lower or "zone" in topic_lower or "runway" in topic_lower:
        return "ISO/IEC 27001-A.11.1 & NIST SP 800-82: Perimeter boundary breach requires immediate containment dispatch."
    return "FAA Part 107 General Duty: Remote pilot must ensure flight path clear of hazardous ground obstacles."


def calculate_containment_radius(speed_kmh: float, reaction_sec: float = 1.5, friction_coeff: float = 0.7) -> float:
    """
    Computes emergency containment perimeter radius using kinematic stopping distance formula:
    r = (v * t_reaction) + (v^2 / (2 * mu * g))
    """
    v_ms = speed_kmh / 3.6
    g = 9.81
    d_reaction = v_ms * reaction_sec
    d_braking = (v_ms ** 2) / (2.0 * friction_coeff * g)
    total_radius = round(d_reaction + d_braking, 2)
    return total_radius


# =============================================================================
# 2. AGENT STATE SCHEMA WITH TOOL CALL CHANNELS
# =============================================================================
class ToolCall(TypedDict):
    tool_name: str
    arguments: Dict[str, Any]


class ToolResult(TypedDict):
    tool_name: str
    output: Any


class IncidentAgentState(TypedDict):
    query: str
    target_track_id: int
    pending_tool_calls: List[ToolCall]
    tool_results: Annotated[List[ToolResult], operator.add]
    iterations: int
    final_assessment: Optional[str]
    execution_trace: Annotated[List[str], operator.add]


# =============================================================================
# 3. REASONING & TOOL DISPATCH NODES
# =============================================================================
def agent_reasoner_node(state: IncidentAgentState) -> Dict[str, Any]:
    """
    Autonomous ReAct reasoner node: evaluates observations and determines
    whether to invoke tools or produce the final incident containment directive.
    """
    iterations = state.get("iterations", 0) + 1
    trace = [f"Step {iterations}: Agent reasoning on query '{state['query']}'"]

    # Step 1: In initial state, look up telemetry
    if not state.get("tool_results"):
        trace.append(f"Reasoning: Inspecting telemetry for Track #{state['target_track_id']}")
        return {
            "iterations": iterations,
            "pending_tool_calls": [
                {"tool_name": "inspect_tracking_telemetry", "arguments": {"track_id": state["target_track_id"]}}
            ],
            "execution_trace": trace
        }

    # Step 2: Telemetry received, now look up regulation and compute containment radius
    has_regulation = any(r["tool_name"] == "query_statutory_regulations" for r in state["tool_results"])
    has_containment = any(r["tool_name"] == "calculate_containment_radius" for r in state["tool_results"])

    if not has_regulation or not has_containment:
        # Extract speed from telemetry result
        telemetry = next((r["output"] for r in state["tool_results"] if r["tool_name"] == "inspect_tracking_telemetry"), {})
        speed = telemetry.get("peak_speed_kmh", 50.0)

        calls = []
        if not has_regulation:
            calls.append({"tool_name": "query_statutory_regulations", "arguments": {"topic": "speeding in restricted zone"}})
        if not has_containment:
            calls.append({"tool_name": "calculate_containment_radius", "arguments": {"speed_kmh": speed, "reaction_sec": 1.5}})

        trace.append(f"Reasoning: Triggering legal lookup & kinematic stopping distance for speed {speed} km/h")
        return {
            "iterations": iterations,
            "pending_tool_calls": calls,
            "execution_trace": trace
        }

    # Step 3: All observations gathered, synthesize final decision
    telemetry = next(r["output"] for r in state["tool_results"] if r["tool_name"] == "inspect_tracking_telemetry")
    regulation = next(r["output"] for r in state["tool_results"] if r["tool_name"] == "query_statutory_regulations")
    radius = next(r["output"] for r in state["tool_results"] if r["tool_name"] == "calculate_containment_radius")

    assessment = (
        f"DECISION: Track #{telemetry['track_id']} ({telemetry['class_name']}) confirmed in RESTRICTED ZONE "
        f"at {telemetry['peak_speed_kmh']} km/h. Governing Law: '{regulation}'. "
        f"Kinematic containment zone locked at {radius}m radius."
    )
    trace.append("Reasoning: All tools resolved. Emitting final assessment.")
    return {
        "iterations": iterations,
        "pending_tool_calls": [],
        "final_assessment": assessment,
        "execution_trace": trace
    }


def tool_execution_node(state: IncidentAgentState) -> Dict[str, Any]:
    """
    Executes pending tool calls and collects results.
    """
    results = []
    trace = []

    for call in state["pending_tool_calls"]:
        name = call["tool_name"]
        args = call["arguments"]
        trace.append(f"Tool Exec: Invoking '{name}' with {json.dumps(args)}")

        if name == "inspect_tracking_telemetry":
            res = inspect_tracking_telemetry(**args)
        elif name == "query_statutory_regulations":
            res = query_statutory_regulations(**args)
        elif name == "calculate_containment_radius":
            res = calculate_containment_radius(**args)
        else:
            res = f"Error: Unknown tool '{name}'"

        results.append({"tool_name": name, "output": res})

    return {
        "tool_results": results,
        "pending_tool_calls": [],
        "execution_trace": trace
    }


# =============================================================================
# 4. CONDITIONAL ROUTING EDGE
# =============================================================================
def should_continue(state: IncidentAgentState) -> Literal["tools", "end"]:
    """
    Cyclic router: routes to 'tools' if calls are pending, else terminates at 'end'.
    """
    if state.get("pending_tool_calls") and len(state["pending_tool_calls"]) > 0:
        return "tools"
    return "end"


# =============================================================================
# 5. GRAPH COMPILATION
# =============================================================================
def build_tool_agent_graph():
    graph = StateGraph(IncidentAgentState)

    graph.add_node("agent", agent_reasoner_node)
    graph.add_node("tools", tool_execution_node)

    graph.set_entry_point("agent")

    # Conditional edge from agent to either tools or END
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )

    # Cyclic edge from tools back to agent
    graph.add_edge("tools", "agent")

    return graph.compile()


# =============================================================================
# 6. EXECUTION & VERIFICATION
# =============================================================================
def main():
    print("=" * 70)
    print("STEP 1: INITIALIZING AUTONOMOUS REACTION GRAPH")
    print("=" * 70)
    runner = build_tool_agent_graph()
    print("✓ Tool calling StateGraph compiled.")

    print("\n" + "=" * 70)
    print("STEP 2: DISPATCHING INCIDENT AUDIT FOR INTRUDER TRACK #3")
    print("=" * 70)
    initial: IncidentAgentState = {
        "query": "Assess containment and legal exposure for speeding vehicle in sector 4",
        "target_track_id": 3,
        "pending_tool_calls": [],
        "tool_results": [],
        "iterations": 0,
        "final_assessment": None,
        "execution_trace": []
    }

    result = runner.invoke(initial)

    print("\nExecution History:")
    for step in result["execution_trace"]:
        print(f"  {step}")

    print("\nFinal Decision Output:")
    print(result["final_assessment"])

    # Assertions
    assert result["final_assessment"] is not None
    assert "RESTRICTED ZONE" in result["final_assessment"]
    assert "radius" in result["final_assessment"]
    assert len(result["tool_results"]) == 3
    print("\n[SUCCESS] Module 02 Tool Calling & Dynamic Routing verified!")


if __name__ == "__main__":
    main()
