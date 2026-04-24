from langgraph.graph import StateGraph, END
from nodes import (
    AgentState,
    classify_node,
    coder_node,
    thinker_node,
    image_node,
    final_output_node,
    audit_node,
)
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver


# ──────────────────────────────────────────────
# MCQ NODE  (human-in-the-loop pause)
# ──────────────────────────────────────────────

def mcq_node(state: AgentState):
    """
    Pauses the graph and asks the user to answer the MCQ axes
    produced by the specialist node.

    Guard: 'writing' / 'general' queries skip specialist nodes so
    state['scaffold'] may not exist — .get() prevents a KeyError.
    """
    scaffold = state.get("scaffold") or {}
    mcq_axes = scaffold.get("mcq_axes", [])

    if not mcq_axes:
        return {"mcq_answer": {"info": "No specific questions required"}}

    print("\n------- PAUSING FOR USER INPUT --------")
    answer = interrupt({
        "message": "Please provide details for the following axes:",
        "questions": mcq_axes,
    })
    return {"mcq_answer": answer}


# ──────────────────────────────────────────────
# GRAPH DEFINITION
# ──────────────────────────────────────────────

memory   = MemorySaver()
workflow = StateGraph(AgentState)

workflow.add_node("classify",     classify_node)
workflow.add_node("coder",        coder_node)
workflow.add_node("thinker",      thinker_node)
workflow.add_node("image",        image_node)
workflow.add_node("mcq",          mcq_node)
workflow.add_node("audit",        audit_node)
workflow.add_node("final_output", final_output_node)

workflow.set_entry_point("classify")


# ── Router: classify → specialist (or straight to final) ──────────────────────

def route_by_category(state: AgentState) -> str:
    category = state["category"]
    if category == "coding":
        return "coder"
    elif category in ("thinking", "data", "writing", "general"):
        return "thinker"
    elif category == "image":
        return "image"
    else:
        # writing / general: no scaffold needed, skip to final
        return "final_output"


workflow.add_conditional_edges(
    "classify",
    route_by_category,
    {
        "coder":        "coder",
        "thinker":      "thinker",
        "image":        "image",
        "final_output": "final_output",
    },
)

workflow.add_edge("coder",   "mcq")
workflow.add_edge("thinker", "mcq")
workflow.add_edge("image",   "mcq")
workflow.add_edge("mcq",     "audit")


# ── Router: audit → final ─────────────────────────────────────────────────────

def route_audit(state: AgentState) -> str:
    # Revision loop (disabled — uncomment to enable):
    # MAX_ITERATIONS = 2
    # if state.get("iteration_count", 0) < MAX_ITERATIONS:
    #     category = state["category"]
    #     if category == "coding":        return "coder"
    #     elif category in ("thinking", "data"): return "thinker"
    #     elif category == "image":       return "image"
    return "final_output"


workflow.add_conditional_edges(
    "audit",
    route_audit,
    {
        "coder":        "coder",
        "thinker":      "thinker",
        "image":        "image",
        "final_output": "final_output",
    },
)

workflow.add_edge("final_output", END)

app = workflow.compile(checkpointer=memory)


# ──────────────────────────────────────────────
# RUNNER
# ──────────────────────────────────────────────

def run_pipeline(user_query: str, complexity: str, thread_id: str = "session-1") -> None:
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "user_query":      user_query,
        "iteration_count": 0,
        "revision_count":  [],
        "mcq_answer":      {},
        "complexity":      "moderate",
        "target_model":    "gemini-3.1",   # overwritten by classify_node
    }

    # ── Phase 1: stream until interrupt (or completion) ───────────────
    
    interrupted_value = None
    print("\n=== PHASE 1: Running pipeline ===")

    for event in app.stream(initial_state, config=config, stream_mode="updates"):
        if "__interrupt__" in event:
            interrupted_value = event["__interrupt__"][0].value
            break
        for node_name in event:
            print(f"  ✓ Node completed: {node_name}")

    # ── Graph finished without needing user input ─────────────────────
    if interrupted_value is None:
        print("\n=== Pipeline completed (no MCQ pause needed) ===")
        _print_result(app.get_state(config).values)
        return

    # ── Phase 2: collect answers then resume ──────────────────────────
    print("\n=== QUESTIONS FROM AGENT ===")
    for q in interrupted_value.get("questions", []):
        print(f"  • {q}")

    user_answers = _collect_answers(interrupted_value.get("questions", []))

    print("\n=== PHASE 2: Resuming with your answers ===")
    final_state = None

    for event in app.stream(
        Command(resume=user_answers),
        config=config,
        stream_mode="updates",   # ← must be "updates", NOT "values"
    ):
        # Catch an unexpected second interrupt (e.g., future revision loop)
        if "__interrupt__" in event:
            print("\n[WARNING] Graph paused again unexpectedly.")
            break
        for node_name in event:
            print(f"  ✓ Node completed: {node_name}")

    # Read the final persisted state from the checkpointer
    final_state = app.get_state(config).values
    _print_result(final_state)


def _collect_answers(questions: list) -> dict:
    """
    Replace this stub with real CLI / UI input collection.
    The keys should match the MCQ axes the agent produced.
    """
    print("\n[Stub] Using demo answers — replace _collect_answers() for production.\n")
    return {
        "error_handling": "Strict — raise HTTP 401 on all auth failures",
        "token_location": "Header: Authorization Bearer",
        "audience":       "Senior",
    }


def _print_result(state: dict | None) -> None:
    if state is None:
        print("\n[ERROR] Graph ended without producing any state.")
        return

    if "final_response" in state:
        print("\n\n══════════════ FINAL PROMPT ══════════════")
        print(state["final_response"])
        print("══════════════════════════════════════════\n")
    else:
        print("\n[ERROR] 'final_response' missing from final state.")
        print(f"  category:        {state.get('category')}")
        print(f"  iteration_count: {state.get('iteration_count')}")
        print(f"  scaffold keys:   {list((state.get('scaffold') or {}).keys())}")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline(
        user_query="i want to create a professional looking thumbnail like dhruv rathee and nitish rajput",
        thread_id="session-1",
        complexity="moderate"
    )
