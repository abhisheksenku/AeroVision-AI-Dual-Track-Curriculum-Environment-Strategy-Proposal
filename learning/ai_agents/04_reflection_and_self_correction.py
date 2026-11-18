"""
=============================================================================
AeroVision AI - Module 04: Agent Reflection & Self-Correction (LangGraph)
=============================================================================
Teaches agentic self-reflection, critique, and self-correction loops:
1. Drafting an initial incident assessment (which may contain unverified claims).
2. Subjecting the draft to a strict Critic / Auditor node checking ground truth.
3. Formulating explicit critique feedback when grounding score < 0.90.
4. Routing through a cyclic self-correction edge to revise and eliminate hallucinations.
=============================================================================
"""

import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, END


# =============================================================================
# 1. STATE SCHEMA FOR REFLECTION LOOP
# =============================================================================
class ReflectionAgentState(TypedDict):
    mission_id: str
    ground_truth_telemetry: Dict[str, Any]
    current_draft: str
    critique_feedback: Optional[str]
    grounding_score: float
    revision_count: int
    is_approved: bool
    execution_trace: Annotated[List[str], operator.add]


# =============================================================================
# 2. COMPUTATIONAL NODES: DRAFTER, CRITIC, AND REFINER
# =============================================================================
def draft_generator_node(state: ReflectionAgentState) -> Dict[str, Any]:
    """
    Initial generator: generates a preliminary assessment.
    Simulates a typical LLM draft that includes an ungrounded speculation in round 1.
    """
    trace = ["Generator: Generating initial incident briefing draft."]
    
    # In round 1, draft contains an exaggerated claim (e.g., helicopter at 120 km/h)
    initial_draft = (
        "INCIDENT BRIEFING: Surveillance drone observed a high-speed helicopter traveling at 120 km/h "
        "violating perimeter airspace near the fuel depot. Emergency intercept scrambled."
    )
    return {
        "current_draft": initial_draft,
        "revision_count": 0,
        "execution_trace": trace
    }


def critic_auditor_node(state: ReflectionAgentState) -> Dict[str, Any]:
    """
    Critic Node: Evaluates current draft against strict CV ground truth telemetry.
    Computes factual grounding score and issues specific critique notes.
    """
    draft = state["current_draft"].lower()
    gt = state["ground_truth_telemetry"]
    revisions = state.get("revision_count", 0)

    trace = [f"Critic: Auditing draft (Revision {revisions})..."]
    flaws = []

    # Check 1: Entity classification fidelity
    expected_class = gt["class_name"].lower()
    if expected_class not in draft:
        flaws.append(f"Entity mismatch: Ground truth entity is '{expected_class}', but draft mentions non-grounded entity.")

    # Check 2: Speed threshold fidelity
    actual_speed = gt["peak_speed_kmh"]
    # Check if actual speed is referenced or if exaggerated speed is present
    if f"{int(actual_speed)}" not in draft and f"{actual_speed:.1f}" not in draft:
        flaws.append(f"Kinematic mismatch: Actual recorded speed is {actual_speed} km/h, not 120 km/h.")

    # Calculate grounding score
    if not flaws:
        score = 1.0
        approved = True
        critique = "All claims verified against Computer Vision telemetry. Zero hallucinations."
        trace.append("Critic Verdict: PASS (Grounding score: 1.0). Certified Grounded.")
    else:
        score = max(0.0, 1.0 - (0.45 * len(flaws)))
        approved = False
        critique = "REVISE: " + "; ".join(flaws)
        trace.append(f"Critic Verdict: REJECT (Grounding score: {score:.2f}). Directing refiner to fix claims.")

    return {
        "grounding_score": score,
        "is_approved": approved,
        "critique_feedback": critique,
        "execution_trace": trace
    }


def refiner_node(state: ReflectionAgentState) -> Dict[str, Any]:
    """
    Refiner Node: Ingests the critique feedback and ground truth to rewrite the draft.
    """
    revisions = state.get("revision_count", 0) + 1
    gt = state["ground_truth_telemetry"]
    trace = [f"Refiner: Applying critique feedback. Generating revision #{revisions}."]

    # Re-generate draft strictly aligned with ground truth
    corrected_draft = (
        f"INCIDENT BRIEFING: Surveillance drone observed a {gt['class_name']} traveling at {gt['peak_speed_kmh']} km/h "
        f"(exceeding the {gt['speed_limit_kmh']} km/h limit) in Sector {gt['sector_id']}. "
        f"Ground safety warden notified for perimeter speed containment."
    )

    return {
        "current_draft": corrected_draft,
        "revision_count": revisions,
        "execution_trace": trace
    }


# =============================================================================
# 3. CONDITIONAL ROUTING EDGE
# =============================================================================
def reflection_router(state: ReflectionAgentState) -> Literal["refiner", "end"]:
    """
    Routes to 'refiner' if draft failed audit and attempts remain, else terminates.
    """
    if state["is_approved"] or state.get("revision_count", 0) >= 3:
        return "end"
    return "refiner"


# =============================================================================
# 4. COMPILING REFLECTION GRAPH
# =============================================================================
def build_reflection_graph():
    graph = StateGraph(ReflectionAgentState)

    graph.add_node("generator", draft_generator_node)
    graph.add_node("critic", critic_auditor_node)
    graph.add_node("refiner", refiner_node)

    # Initial flow: Generator -> Critic
    graph.set_entry_point("generator")
    graph.add_edge("generator", "critic")

    # Conditional decision: Critic -> (Refiner | END)
    graph.add_conditional_edges(
        "critic",
        reflection_router,
        {
            "refiner": "refiner",
            "end": END
        }
    )

    # Cyclic feedback: Refiner -> Critic
    graph.add_edge("refiner", "critic")

    return graph.compile()


# =============================================================================
# 5. EXECUTION & VERIFICATION
# =============================================================================
def main():
    print("=" * 70)
    print("STEP 1: COMPILING AGENT REFLECTION & SELF-CORRECTION GRAPH")
    print("=" * 70)
    runner = build_reflection_graph()
    print("✓ Self-correction StateGraph compiled successfully.")

    print("\n" + "=" * 70)
    print("STEP 2: RUNNING AUTONOMOUS AUDIT & REVISION CYCLE")
    print("=" * 70)
    initial_state: ReflectionAgentState = {
        "mission_id": "MS-REFLECT-007",
        "ground_truth_telemetry": {
            "track_id": 4,
            "class_name": "car",
            "peak_speed_kmh": 68.4,
            "speed_limit_kmh": 50.0,
            "sector_id": "B-4"
        },
        "current_draft": "",
        "critique_feedback": None,
        "grounding_score": 0.0,
        "revision_count": 0,
        "is_approved": False,
        "execution_trace": []
    }

    final_state = runner.invoke(initial_state)

    print("\nOrchestration Trace History:")
    for step in final_state["execution_trace"]:
        print(f"  -> {step}")

    print("\nFinal Approved Assessment:")
    print(final_state["current_draft"])
    print(f"Final Grounding Score: {final_state['grounding_score']:.2f}")
    print(f"Total Revisions Made: {final_state['revision_count']}")

    # Verifications
    assert final_state["is_approved"] is True
    assert final_state["grounding_score"] == 1.0
    assert final_state["revision_count"] == 1
    assert "car" in final_state["current_draft"]
    assert "68.4 km/h" in final_state["current_draft"]
    assert "helicopter" not in final_state["current_draft"]
    print("\n[SUCCESS] Module 04 Reflection & Self-Correction verified!")


if __name__ == "__main__":
    main()
