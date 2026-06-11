"""
api/main.py — FastAPI application.

Endpoints:
  POST /start_session      Start a new session for a thread
  POST /ask                Submit a query, get grounded answer + agent trace
  POST /switch_thread      Switch active thread (preserves entity register)
  POST /reset_session      Clear session state
  GET  /threads            List all indexed thread_ids
  GET  /health             Liveness check

Session state is in-memory (dict keyed by session_id).
In production: use Redis for multi-worker deployments.
"""

from __future__ import annotations
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.graph import run_turn
from utils.tracer import init_run, read_trace


# ──────────────────────────────────────────────
# Startup: load index once when the process starts
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_run()
    try:
        from ingest import load_index
        load_index()
    except FileNotFoundError:
        print("[WARN] Index not found — run `python ingest.py --data-dir data/slice/` first")
    except Exception as e:
        print(f"[WARN] Index load failed: {e}")
    yield


app = FastAPI(title="Email RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# In-memory session store
# ──────────────────────────────────────────────

class Session:
    def __init__(self, session_id: str, thread_id: str):
        self.session_id = session_id
        self.thread_id = thread_id
        self.turn_history: list[dict] = []
        self.entity_register: dict[str, Any] = {}
        self.turn_count: int = 0

_sessions: dict[str, Session] = {}


# ──────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    thread_id: str

class StartSessionResponse(BaseModel):
    session_id: str
    thread_id: str
    message: str


class AskRequest(BaseModel):
    session_id: str
    text: str
    search_outside_thread: bool = False

class CitationOut(BaseModel):
    claim_text: str
    message_id: str
    page_no: int | None = None
    confidence: float

class AgentTraceEntry(BaseModel):
    agent: str
    routing: str
    latency_ms: float
    sub_queries: int | None = None
    grounding_score: float | None = None
    retry: bool | None = None

class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    rewrite: str                    # resolved query after Agent 1
    stepback: str | None            # step-back query
    retrieved: list[dict]           # top-k retrieved chunk summaries
    grounding_score: float
    trace_id: str
    agent_trace: list[dict]         # one entry per agent node fired
    entity_register: dict[str, Any] # current entity register state for UI


class SwitchThreadRequest(BaseModel):
    session_id: str
    thread_id: str

class ResetSessionRequest(BaseModel):
    session_id: str


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.post("/start_session", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest):
    session_id = str(uuid.uuid4())
    _sessions[session_id] = Session(session_id=session_id, thread_id=req.thread_id)
    return StartSessionResponse(
        session_id=session_id,
        thread_id=req.thread_id,
        message=f"Session started for thread {req.thread_id}",
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Call /start_session first.")

    # Run the full LangGraph pipeline
    final_state = run_turn(
        session_id=session.session_id,
        thread_id=session.thread_id,
        raw_query=req.text,
        turn_history=session.turn_history,
        entity_register=session.entity_register,
        search_outside_thread=req.search_outside_thread,
    )

    # Update session state from pipeline output
    session.entity_register = final_state.get("entity_register", session.entity_register)
    session.turn_count += 1
    session.turn_history.append({"role": "user",      "content": req.text})
    session.turn_history.append({"role": "assistant",  "content": final_state.get("final_answer", "")})
    # Rolling window: keep last 10 turns (5 exchanges)
    session.turn_history = session.turn_history[-10:]

    # Build retrieved summary (no full text in response — keep payload small)
    retrieved_summary = [
        {
            "doc_id":       c.get("doc_id", ""),
            "message_id":   c.get("message_id", ""),
            "thread_id":    c.get("thread_id", ""),
            "page_no":      c.get("page_no"),
            "source":       c.get("source", ""),
            "rerank_score": round(c.get("rerank_score", c.get("rrf_score", 0.0)), 4),
            "preview":      c.get("text", "")[:120] + "…",
        }
        for c in final_state.get("retrieved_chunks", [])
    ]

    # Build agent trace from tracer log (records from this turn)
    trace_records = read_trace(session_id=session.session_id)
    current_turn_records = [
        r for r in trace_records
        if r.get("turn") == session.turn_count - 1
    ]

    trace_id = current_turn_records[0]["trace_id"] if current_turn_records else str(uuid.uuid4())

    agent_trace_out = [
        {
            "agent":       r["agent"],
            "routing":     r.get("routing", ""),
            "latency_ms":  r.get("latency_ms", 0),
            "token_count": r.get("token_count", 0),
            **{k: v for k, v in r.get("output", {}).items()
               if k in {"sub_queries", "grounding_score", "retrieval_insufficient",
                        "n_events", "cluster_label", "resolved_query"}},
        }
        for r in current_turn_records
    ]

    citations_out = [
        CitationOut(
            claim_text=c.get("claim_text", ""),
            message_id=c.get("message_id", ""),
            page_no=c.get("page_no"),
            confidence=round(c.get("confidence", 0.0), 3),
        )
        for c in final_state.get("citations", [])
    ]

    return AskResponse(
        answer=final_state.get("final_answer", ""),
        citations=citations_out,
        rewrite=final_state.get("resolved_query", req.text),
        stepback=final_state.get("stepback_query"),
        retrieved=retrieved_summary,
        grounding_score=round(final_state.get("grounding_score", 0.0), 3),
        trace_id=trace_id,
        agent_trace=agent_trace_out,
        entity_register=session.entity_register,
    )


@app.post("/switch_thread")
def switch_thread(req: SwitchThreadRequest):
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    old_thread = session.thread_id
    session.thread_id = req.thread_id
    # Preserve entity register and history across thread switch
    return {"message": f"Switched from {old_thread} to {req.thread_id}"}


@app.post("/reset_session")
def reset_session(req: ResetSessionRequest):
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    thread_id = session.thread_id
    _sessions[req.session_id] = Session(session_id=req.session_id, thread_id=thread_id)
    return {"message": "Session reset. Entity register and history cleared."}


@app.get("/threads")
def list_threads():
    """Return all indexed thread IDs (loaded from index manifest at startup)."""
    try:
        from ingest import get_all_thread_ids
        threads = get_all_thread_ids()
        return {"threads": threads}
    except Exception:
        return {"threads": [], "note": "Run ingest.py first to populate the index."}


@app.get("/health")
def health():
    return {"status": "ok", "sessions_active": len(_sessions)}
