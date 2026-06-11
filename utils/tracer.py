"""
utils/tracer.py — Writes one JSON record per agent invocation to runs/<timestamp>/trace.jsonl.
Import and call log_agent() at the top of every agent function.
"""

from __future__ import annotations
import json
import os
import time
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

_run_dir: Path | None = None


def init_run() -> Path:
    """Call once at process start. Returns the run directory."""
    global _run_dir
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    _run_dir = Path("runs") / ts
    _run_dir.mkdir(parents=True, exist_ok=True)
    return _run_dir


def _get_run_dir() -> Path:
    global _run_dir
    if _run_dir is None:
        _run_dir = init_run()
    return _run_dir


class AgentTimer:
    """Context manager — use inside every agent to measure latency."""
    def __init__(self):
        self._start: float = 0.0
        self.latency_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.latency_ms = round((time.perf_counter() - self._start) * 1000, 1)


def log_agent(
    *,
    session_id: str,
    agent: str,
    turn: int,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    routing: str,
    latency_ms: float,
    token_count: int = 0,
    trace_id: str | None = None,
) -> str:
    """
    Append one record to trace.jsonl. Returns the trace_id.

    Call pattern inside every agent:

        with AgentTimer() as t:
            # ... do work ...
        trace_id = log_agent(
            session_id=state["session_id"],
            agent="entity_resolution",
            turn=len(state["turn_history"]),
            inputs={"raw_query": state["raw_query"]},
            outputs={"resolved_query": resolved, "clarify_needed": False},
            routing="metadata_prefilter",
            latency_ms=t.latency_ms,
            token_count=tokens_used,
        )
    """
    tid = trace_id or str(uuid.uuid4())
    record = {
        "trace_id": tid,
        "session_id": session_id,
        "agent": agent,
        "turn": turn,
        "input": inputs,
        "output": outputs,
        "routing": routing,
        "latency_ms": latency_ms,
        "token_count": token_count,
        "ts": datetime.now(UTC).isoformat(),
    }
    log_path = _get_run_dir() / "trace.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return tid


def read_trace(session_id: str | None = None) -> list[dict]:
    """Read all records from the current run, optionally filtered by session."""
    log_path = _get_run_dir() / "trace.jsonl"
    if not log_path.exists():
        return []
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    if session_id:
        records = [r for r in records if r.get("session_id") == session_id]
    return records
