"""
tests/create_submission_trace.py

Runs a full demonstration conversation through the pipeline and writes
one clean runs/<timestamp>/trace.jsonl for submission.

Covers:
  - All 3 mandatory agents (entity_resolution, query_breakdown, citation_validator)
  - Pronoun/ellipsis resolution across turns
  - Timeline synthesis agent
  - Attachment citation
  - Agent 3 retry (grounding uplift)
  - Graceful failure

Run:
    python tests/create_submission_trace.py
"""

import sys
import pathlib
import time
import json

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# ── 1. Init a fresh run directory before anything else ──────────────────────
from utils.tracer import init_run
run_dir = init_run()
print(f"Run directory: {run_dir}")

# ── 2. Load index ────────────────────────────────────────────────────────────
print("Loading index...")
from ingest import load_index
load_index()

from agents.graph import run_turn

# ── 3. Conversation definition ───────────────────────────────────────────────
#
# SESSION A — T-9114D5B0 (Succession Plan)
#   Turn 0 : broad summary           → summary cluster, MMR, 6 citations
#   Turn 1 : participants             → person cluster, roster NER
#   Turn 2 : leadership status        → status cluster, 4 citations
#   Turn 3 : when announcements       → timeline cluster, timeline_synthesis fires
#   Turn 4 : "Who made that?"         → pronoun resolution demo
#   Turn 5 : attachment page 1        → attachment cluster, doc citation
#
# SESSION B — T-5CADBCA8 (FERC price caps)
#   Turn 0 : Enron's FERC position    → Agent 3 retry (grounding 50% → retry → 100%)
#
# SESSION C — T-BADA59FE (Congratulations)
#   Turn 0 : graceful failure query   → no formal approval in a congrats thread

SESSIONS = [
    {
        "session_id": "submission-succession",
        "thread_id":  "T-9114D5B0",
        "label":      "Succession Plan — main demo thread",
        "turns": [
            "What is this email thread about?",
            "Who are the participants in this conversation?",
            "What is the status of the leadership transition?",
            "When did the key announcements happen?",
            "Who made that announcement?",
            "What is on page 1 of the attachment?",
        ],
    },
    {
        "session_id": "submission-ferc",
        "thread_id":  "T-5CADBCA8",
        "label":      "FERC price caps — Agent 3 retry / grounding uplift demo",
        "turns": [
            "What is Enron's position on FERC price caps?",
        ],
    },
    {
        "session_id": "submission-graceful",
        "thread_id":  "T-BADA59FE",
        "label":      "Congratulations thread — graceful failure demo",
        "turns": [
            "Was anything formally approved or decided?",
        ],
    },
]

# ── 4. Run ───────────────────────────────────────────────────────────────────
results = []
total_turns = sum(len(s["turns"]) for s in SESSIONS)
done = 0

for session in SESSIONS:
    print(f"\n{'-'*60}")
    print(f"SESSION: {session['label']}")
    print(f"Thread:  {session['thread_id']}")

    turn_history   = []
    entity_register = {}

    for turn_idx, query in enumerate(session["turns"]):
        done += 1
        print(f"\n  [{done}/{total_turns}] Turn {turn_idx}: {query}")
        t0 = time.time()

        state = run_turn(
            session_id=session["session_id"],
            thread_id=session["thread_id"],
            raw_query=query,
            turn_history=turn_history,
            entity_register=entity_register,
            search_outside_thread=False,
        )

        ms = int((time.time() - t0) * 1000)

        # Carry forward entity register for pronoun resolution across turns
        entity_register = state.get("entity_register", entity_register)
        turn_history.append({"role": "user",      "content": query})
        turn_history.append({"role": "assistant",  "content": state.get("final_answer", "")})

        answer   = state.get("final_answer", "")
        cits     = state.get("citations", [])
        grounding = state.get("grounding_score", 0.0)
        cluster  = state.get("cluster_label", "?")
        rewrite  = state.get("resolved_query", query)
        attempt  = state.get("retrieval_attempt", 1)

        print(f"  cluster={cluster}  grounding={grounding:.0%}  "
              f"citations={len(cits)}  attempt={attempt}  {ms}ms")
        print(f"  rewrite : {rewrite}")
        print(f"  answer  : {answer[:160]}{'…' if len(answer) > 160 else ''}")
        for c in cits:
            page = f", page={c['page_no']}" if c.get("page_no") else ""
            print(f"    cite: [msg: {c['message_id'][:45]}{page}]  "
                  f"conf={c['confidence']:.0%}")

        results.append({
            "session_id": session["session_id"],
            "thread_id":  session["thread_id"],
            "turn":       turn_idx,
            "query":      query,
            "rewrite":    rewrite,
            "cluster":    cluster,
            "grounding":  grounding,
            "attempt":    attempt,
            "citations":  len(cits),
            "ms":         ms,
            "answer_preview": answer[:200],
        })

# ── 5. Summary ───────────────────────────────────────────────────────────────
trace_path = run_dir / "trace.jsonl"
n_records  = sum(1 for _ in open(trace_path))

print(f"\n{'='*60}")
print("SUBMISSION TRACE COMPLETE")
print(f"  File   : {trace_path}")
print(f"  Records: {n_records}  ({total_turns} turns × ~7 agents each)")
print(f"\nPer-turn summary:")
for r in results:
    status = "RETRY" if r["attempt"] > 2 else "OK"
    print(f"  [{r['session_id'][:20]}] t{r['turn']}  "
          f"cluster={r['cluster']:<10} grounding={r['grounding']:.0%}  "
          f"cit={r['citations']}  {r['ms']}ms  {status}")

# ── 6. Verify all 3 mandatory agents appear ──────────────────────────────────
print(f"\nAgent coverage check:")
agents_seen = set()
with open(trace_path) as f:
    for line in f:
        rec = json.loads(line)
        agents_seen.add(rec["agent"])

mandatory = {"entity_resolution", "query_breakdown", "citation_validator"}
for agent in sorted(agents_seen):
    tag = "[MANDATORY]" if agent in mandatory else "[bonus]   "
    print(f"  {tag}  {agent}")

missing = mandatory - agents_seen
if missing:
    print(f"\n  MISSING: {missing}")
else:
    print(f"\n  All 3 mandatory agents present in trace - OK")

print(f"\nTrace saved to: {trace_path}")
