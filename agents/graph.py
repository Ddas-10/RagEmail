"""
agents/graph.py — Full LangGraph StateGraph definition.

Node inventory:
  entity_resolution   Agent 1
  metadata_prefilter  Lightweight rule-based
  query_expansion     Step-back + HyDE + cluster routing
  query_breakdown     Agent 2
  retrieval           BM25 + FAISS + RRF + reranker
  citation_validator  Agent 3
  timeline_synthesis  Bonus: temporal Q handler
  clarify             Early-exit for ambiguous queries
  respond             Final assembly

Routing rules:
  entity_resolution → clarify          if clarify_needed
  entity_resolution → metadata_prefilter  otherwise
  citation_validator → retrieval       if retrieval_insufficient (max 1 retry)
  citation_validator → timeline_synthesis  otherwise
  timeline_synthesis → respond         always
"""

from langgraph.graph import StateGraph, END

from utils.models import PipelineState
from agents.agents import (
    entity_resolution_agent,
    metadata_prefilter_node,
    query_expansion_node,
    query_breakdown_agent,
    citation_validator_agent,
    timeline_synthesis_agent,
    clarify_node,
    respond_node,
    ROUTE_CLARIFY,
    ROUTE_PREFILTER,
    ROUTE_RESPOND,
    ROUTE_RETRIEVAL,
    ROUTE_TIMELINE,
)
from retrieval.hybrid import retrieval_node


# ──────────────────────────────────────────────
# Routing functions (pure — read state, return string)
# ──────────────────────────────────────────────

def route_entity_resolution(state: PipelineState) -> str:
    """Fire clarify edge only on genuine ambiguity."""
    return ROUTE_CLARIFY if state.get("clarify_needed") else ROUTE_PREFILTER


def route_citation_validator(state: PipelineState) -> str:
    """
    Retry retrieval if grounding score is below threshold.
    Hard cap at MAX_RETRIEVAL_ATTEMPTS to prevent loops.
    """
    # retrieval_attempt is incremented by retrieval_node before we reach here.
    # After attempt 1 → value is 2 (allow retry).  After retry → value is 3 (stop).
    if state.get("retrieval_insufficient") and state.get("retrieval_attempt", 1) <= 2:
        return ROUTE_RETRIEVAL
    return ROUTE_TIMELINE


def route_timeline(state: PipelineState) -> str:
    """Timeline synthesis always routes to respond."""
    return ROUTE_RESPOND


# ──────────────────────────────────────────────
# Graph construction
# ──────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    # ── Nodes ──
    graph.add_node("entity_resolution",   entity_resolution_agent)
    graph.add_node("metadata_prefilter",  metadata_prefilter_node)
    graph.add_node("query_expansion",     query_expansion_node)
    graph.add_node("query_breakdown",     query_breakdown_agent)
    graph.add_node("retrieval",           retrieval_node)
    graph.add_node("citation_validator",  citation_validator_agent)
    graph.add_node("timeline_synthesis",  timeline_synthesis_agent)
    graph.add_node("clarify",             clarify_node)
    graph.add_node("respond",             respond_node)

    # ── Entry ──
    graph.set_entry_point("entity_resolution")

    # ── Conditional: entity_resolution → clarify | metadata_prefilter ──
    graph.add_conditional_edges(
        "entity_resolution",
        route_entity_resolution,
        {
            ROUTE_CLARIFY:    "clarify",
            ROUTE_PREFILTER:  "metadata_prefilter",
        },
    )

    # ── Linear: prefilter → expansion → breakdown → retrieval ──
    graph.add_edge("metadata_prefilter", "query_expansion")
    graph.add_edge("query_expansion",    "query_breakdown")
    graph.add_edge("query_breakdown",    "retrieval")
    graph.add_edge("retrieval",          "citation_validator")

    # ── Conditional: citation_validator → retry | timeline ──
    graph.add_conditional_edges(
        "citation_validator",
        route_citation_validator,
        {
            ROUTE_RETRIEVAL: "retrieval",     # retry with wider top-k
            ROUTE_TIMELINE:  "timeline_synthesis",
        },
    )

    # ── Linear: timeline → respond → END ──
    graph.add_edge("timeline_synthesis", "respond")

    # ── Terminal edges ──
    graph.add_edge("clarify", END)
    graph.add_edge("respond", END)

    return graph


# Compiled graph — import this in api/main.py
compiled_graph = build_graph().compile()


# ──────────────────────────────────────────────
# Session runner
# ──────────────────────────────────────────────

def run_turn(
    session_id: str,
    thread_id: str,
    raw_query: str,
    turn_history: list[dict],
    entity_register: dict,
    search_outside_thread: bool = False,
) -> dict:
    """
    Execute one conversational turn through the full pipeline.
    Returns the final state after graph execution.
    """
    initial_state: PipelineState = {
        # Session
        "session_id": session_id,
        "thread_id": thread_id,
        "turn_history": turn_history[-5:],  # rolling last-5 turns
        "search_outside_thread": search_outside_thread,
        "summarize_thread": False,  # set by query_expansion when CLS routing → "summary" cluster
        # Agent 1
        "entity_register": entity_register,
        "raw_query": raw_query,
        "resolved_query": "",
        "clarify_needed": False,
        "clarify_text": None,
        # Prefilter
        "active_thread_ids": [thread_id],
        "date_filter": None,
        "scope_hint": "both",
        # Expansion
        "stepback_query": None,
        "hyde_query": None,
        "cluster_label": None,
        "expansion_queries": [],
        # Agent 2
        "sub_queries": [],
        # Retrieval
        "retrieved_chunks": [],
        "retrieval_attempt": 1,
        # Agent 3
        "draft_answer": "",
        "citations": [],
        "grounding_score": 0.0,
        "retrieval_insufficient": False,
        # Timeline
        "is_temporal_query": False,
        "timeline_events": None,
        # Output
        "final_answer": "",
        "agent_trace": [],
    }

    return compiled_graph.invoke(initial_state)
