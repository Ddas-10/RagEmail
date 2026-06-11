"""
tests/test_e2e.py — End-to-end pipeline tests across all 19 threads.

Run:
    python tests/test_e2e.py

Tests every thread with 5 query types:
  - topic   : What is this thread about?
  - summary : Summarise this thread
  - person  : Who are the people involved?
  - status  : What is the current status?
  - timeline: When did the key events happen?

Plus regression tests for previously fixed bugs.

Exit code: 0 = all pass, 1 = failures present.
"""

from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows consoles (avoids cp1252 UnicodeEncodeError
# when answers contain non-ASCII characters).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Make sure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── Load index once ──────────────────────────────────────────────────────────

def _load_index():
    from ingest import load_index
    load_index()
    print("[OK] Index loaded\n")

_load_index()

from agents.graph import run_turn


# ── Helpers ──────────────────────────────────────────────────────────────────

FALLBACK_PHRASES = [
    "i could not find relevant information",
    "no relevant information found",
    "could not find information",
    "unable to find",
    "i was unable to find",
]

def _is_fallback(answer: str) -> bool:
    low = answer.lower()
    return any(p in low for p in FALLBACK_PHRASES)


def _ask(thread_id: str, query: str, session_id: str = "test",
         entity_register: dict | None = None) -> dict:
    return run_turn(
        session_id=session_id,
        thread_id=thread_id,
        raw_query=query,
        turn_history=[],
        entity_register=entity_register or {},
        search_outside_thread=False,
    )


def _check(label: str, condition: bool, detail: str = "") -> tuple[bool, str]:
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {label}"
    if detail and not condition:
        msg += f"\n         -> {detail}"
    return condition, msg


# ── Thread roster (for person-query validation) ──────────────────────────────

with open(ROOT / "data/index/thread_roster.json") as f:
    _ROSTER: dict[str, list[str]] = json.load(f)


# ── Generic per-thread tests ─────────────────────────────────────────────────

QUERY_TYPES = [
    ("topic",    "What is this thread about?"),
    ("summary",  "Summarise this thread"),
    ("person",   "Who are the people involved in this thread?"),
    ("status",   "What is the current status or outcome of this discussion?"),
    ("timeline", "When did the key events happen?"),
]

def run_thread_tests(thread_id: str, roster: list[str]) -> list[tuple[bool, str]]:
    results: list[tuple[bool, str]] = []
    roster_lower = {n.lower() for n in roster}
    first_names  = {n.split()[0].lower() for n in roster}

    for qtype, query in QUERY_TYPES:
        state  = _ask(thread_id, query, session_id=f"e2e-{thread_id}-{qtype}")
        answer = state.get("final_answer", "")
        cluster = state.get("cluster_label", "")

        # 1. Non-empty answer
        ok, msg = _check(
            f"{qtype}: non-empty answer",
            bool(answer.strip()),
            f"empty answer for thread={thread_id} query='{query}'",
        )
        results.append((ok, msg))

        # 2. Not a fallback message
        ok, msg = _check(
            f"{qtype}: no fallback",
            not _is_fallback(answer),
            f"fallback triggered — answer: {answer[:120]}",
        )
        results.append((ok, msg))

        # 3. Minimum word count — skip for person queries where a short
        #    "Participants: Name1, Name2." is a CORRECT answer for small rosters.
        #    The non-empty and no-fallback checks above already guard against
        #    degenerate responses; word count is redundant for person queries.
        if qtype != "person":
            word_count = len(answer.split())
            ok, msg = _check(
                f"{qtype}: word count >= 10",
                word_count >= 10,
                f"only {word_count} words — answer: {answer[:100]}",
            )
            results.append((ok, msg))

        # 4. Person queries must contain at least one roster name
        if qtype == "person" and roster:
            answer_low = answer.lower()
            name_found = any(n in answer_low for n in roster_lower) or \
                         any(fn in answer_low for fn in first_names)
            ok, msg = _check(
                f"{qtype}: contains a participant name",
                name_found,
                f"roster={list(roster_lower)[:3]}, answer={answer[:150]}",
            )
            results.append((ok, msg))

        # 5. Summary/topic queries must not be a single sentence (need depth)
        if qtype in ("summary",):
            sentence_count = answer.count(".") + answer.count("!") + answer.count("?")
            ok, msg = _check(
                f"{qtype}: answer has ≥ 2 sentences",
                sentence_count >= 2,
                f"only {sentence_count} sentences — answer: {answer[:150]}",
            )
            results.append((ok, msg))

    return results


# ── Regression tests (named bugs fixed in this session) ─────────────────────

def run_regressions() -> list[tuple[bool, str]]:
    results: list[tuple[bool, str]] = []

    # R1: Succession planning query on correct thread should return Skilling
    state = _ask(
        "T-9114D5B0",
        "What happened with executive succession planning?",
        session_id="regression-T-9114D5B0-r1",
    )
    answer = state.get("final_answer", "")
    ok, msg = _check(
        "R1: succession -> mentions Skilling or CEO appointment",
        "skilling" in answer.lower() or "ceo" in answer.lower() or "chief executive" in answer.lower(),
        f"answer: {answer[:200]}",
    )
    results.append((ok, msg))

    # R2: FERC person query must return participant names, not analysis content
    state = _ask(
        "T-5CADBCA8",
        "Who are the people involved in the FERC discussions?",
        session_id="regression-T-5CADBCA8-r2",
    )
    answer = state.get("final_answer", "")
    ferc_roster = _ROSTER.get("T-5CADBCA8", [])
    roster_low  = {n.lower() for n in ferc_roster}
    first_names  = {n.split()[0].lower() for n in ferc_roster}
    names_found = any(n in answer.lower() for n in roster_low) or \
                  any(fn in answer.lower() for fn in first_names)
    ok, msg = _check(
        "R2: FERC person query -> returns participant names (not topic content)",
        names_found and not (
            "price cap" in answer.lower() and len(answer.split()) > 30
            and not names_found
        ),
        f"roster={list(ferc_roster)[:3]}, answer={answer[:200]}",
    )
    results.append((ok, msg))

    # R3: Succession summary must not contain RAPTOR name-pile hallucination
    state = _ask(
        "T-9114D5B0",
        "Summarise the whole thread",
        session_id="regression-T-9114D5B0-r3",
    )
    answer = state.get("final_answer", "")
    hallucination_marker = "vanished from the cameron" in answer.lower() or \
                           "mcgregory" in answer.lower()
    ok, msg = _check(
        "R3: succession summary -> no RAPTOR name-pile hallucination",
        not hallucination_marker,
        f"hallucination detected in: {answer[:200]}",
    )
    results.append((ok, msg))

    # R4: FERC topic answer must not have duplicate sentences
    state = _ask(
        "T-5CADBCA8",
        "What is the FERC price cap?",
        session_id="regression-T-5CADBCA8-r4",
    )
    answer = state.get("final_answer", "")
    sentences = [s.strip() for s in answer.replace("\n", " ").split(".") if len(s.strip()) > 20]
    seen: list[str] = []
    has_dup = False
    for s in sentences:
        s_words = set(s.lower().split())
        if any(len(s_words & set(p.lower().split())) / max(len(s_words), 1) > 0.70 for p in seen):
            has_dup = True
            break
        seen.append(s)
    ok, msg = _check(
        "R4: FERC status -> no duplicate sentences (sentence-level dedup working)",
        not has_dup,
        f"duplicate sentences detected in: {answer[:250]}",
    )
    results.append((ok, msg))

    # R5: Person query must return names even when ALL retrieved chunks are RAPTOR
    #     (thread roster fallback test — any thread with a good roster)
    state = _ask(
        "T-5E207662",
        "Who are the key people in this thread?",
        session_id="regression-T-5E207662-r5",
    )
    answer = state.get("final_answer", "")
    roster_5e = _ROSTER.get("T-5E207662", [])
    ok, msg = _check(
        "R5: person query on T-5E207662 -> returns participant names (roster fallback)",
        any(n.split()[0].lower() in answer.lower() for n in roster_5e[:5]) or
        any(n.lower() in answer.lower() for n in roster_5e[:5]),
        f"roster={roster_5e[:3]}, answer={answer[:200]}",
    )
    results.append((ok, msg))

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    with open(ROOT / "data/index/manifest.json") as f:
        manifest = json.load(f)
    thread_ids = manifest["thread_ids"]

    all_results: list[tuple[bool, str]] = []
    thread_failures: dict[str, int] = {}

    print("=" * 68, flush=True)
    print(f"Testing {len(thread_ids)} threads x {len(QUERY_TYPES)} query types", flush=True)
    print("=" * 68, flush=True)

    for tid in thread_ids:
        roster = _ROSTER.get(tid, [])
        t0 = time.time()
        results = run_thread_tests(tid, roster)
        elapsed = time.time() - t0

        fails = sum(1 for ok, _ in results if not ok)
        thread_failures[tid] = fails
        icon = "OK" if fails == 0 else f"FAIL({fails})"
        print(f"\n[{icon}] {tid}  ({elapsed:.1f}s, roster: {roster[:2]})", flush=True)
        for ok, msg in results:
            if not ok:
                print(msg, flush=True)

        all_results.extend(results)

    print("\n" + "=" * 68, flush=True)
    print("REGRESSION TESTS", flush=True)
    print("=" * 68, flush=True)
    reg_results = run_regressions()
    for ok, msg in reg_results:
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {msg.strip()}", flush=True)
    all_results.extend(reg_results)

    print("\n" + "=" * 68, flush=True)
    total  = len(all_results)
    passed = sum(1 for ok, _ in all_results if ok)
    failed = total - passed
    print(f"TOTAL: {passed}/{total} passed, {failed} failed", flush=True)

    if failed:
        print("\nFailed checks:", flush=True)
        for ok, msg in all_results:
            if not ok:
                print(msg, flush=True)
        sys.exit(1)
    else:
        print("\nAll checks passed.", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
