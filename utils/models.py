"""
utils/models.py — All shared schemas. Import from here everywhere.
"""

from __future__ import annotations
from typing import Any, Literal, TypedDict
from pydantic import BaseModel, Field
import uuid


# ─────────────────────────────────────────────
# Core data models
# ─────────────────────────────────────────────

class EmailRecord(BaseModel):
    message_id: str
    thread_id: str
    date: str                  # ISO-8601
    from_addr: str
    to_cc: list[str]
    subject: str
    body: str
    attachment_ids: list[str] = Field(default_factory=list)
    participants: list[str] = Field(default_factory=list)  # display names from From/To/Cc


class Chunk(BaseModel):
    doc_id: str                # unique chunk id
    thread_id: str
    message_id: str
    source: Literal["email", "attachment"]
    text: str
    page_no: int | None = None   # attachments only
    token_count: int = 0
    embedding: list[float] | None = None  # not serialised to JSON
    bm25_score: float = 0.0
    vector_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0


class SubQuery(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str
    scope: Literal["email", "attachment", "both"] = "both"
    priority: int = 1           # lower = higher priority; parallel when same


class Citation(BaseModel):
    claim_text: str
    message_id: str
    page_no: int | None = None
    chunk_text: str             # the supporting chunk excerpt
    token_f1: float             # overlap score between claim and chunk
    confidence: float


# ─────────────────────────────────────────────
# LangGraph shared state
# All agents read/write from this single object.
# Each agent only touches its designated fields.
# ─────────────────────────────────────────────

class PipelineState(TypedDict):
    # ── Session context (set once at start_session) ──
    session_id: str
    thread_id: str
    turn_history: list[dict]          # {role, content} pairs, rolling last-5
    search_outside_thread: bool       # flag from UI toggle
    summarize_thread: bool            # True when cluster_label == "summary" (set by query_expansion)

    # ── Agent 1: Entity Resolution ──
    entity_register: dict[str, Any]   # {entity_text: {type, message_id, turn}}
    raw_query: str
    resolved_query: str
    clarify_needed: bool
    clarify_text: str | None

    # ── Metadata prefilter (rule-based node) ──
    active_thread_ids: list[str]      # usually [thread_id], expanded if cross-thread
    date_filter: dict | None          # {"after": ISO, "before": ISO} or None
    scope_hint: str                   # "email" | "attachment" | "both"

    # ── Query expansion node ──
    stepback_query: str | None        # abstract reformulation
    hyde_query: str | None            # hypothetical document text
    cluster_label: str | None         # nearest prototype cluster
    expansion_queries: list[str]      # all expanded forms fed to retrieval

    # ── Agent 2: Query Breakdown ──
    sub_queries: list[dict]           # list[SubQuery.dict()]

    # ── Retrieval node ──
    retrieved_chunks: list[dict]      # list[Chunk.dict()], scored + ranked
    retrieval_attempt: int            # starts at 1, max 2

    # ── Agent 3: Citation Validator ──
    draft_answer: str
    citations: list[dict]             # list[Citation.dict()]
    grounding_score: float            # 0.0–1.0
    retrieval_insufficient: bool

    # ── Timeline synthesis (conditional) ──
    timeline_events: list[dict] | None   # [{event, date, message_id}]
    is_temporal_query: bool

    # ── Final output ──
    final_answer: str
    agent_trace: list[dict]           # one entry per node fired this turn
