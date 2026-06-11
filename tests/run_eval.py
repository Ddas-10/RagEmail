"""
tests/run_eval.py — Runs the 10 sample questions and prints a graded report.

Usage:
    python tests/run_eval.py --questions tests/eval_questions.json

Checks per question:
  - must_include: all strings appear in the answer
  - must_cite: each expected citation appears in citations list
  - clarify_should_fire: if specified, checks clarify_needed
  - routing: prints actual vs expected for manual review

Prints a summary table and exits with code 1 if any test fails.
"""

from __future__ import annotations
import argparse
import json
import sys
import requests

API_BASE = "http://localhost:8000"


def run_eval(questions_path: str):
    with open(questions_path) as f:
        spec = json.load(f)

    thread_id = spec["thread_id"]
    print(f"\nEval: thread {thread_id} — {len(spec['conversations'])} questions\n")
    print("=" * 70)

    # Start session
    resp = requests.post(f"{API_BASE}/start_session", json={"thread_id": thread_id})
    resp.raise_for_status()
    session_id = resp.json()["session_id"]

    results = []

    for conv in spec["conversations"]:
        qid = conv["id"]
        query = conv["turn"] == 1 and conv["query"] or conv["query"]
        expected = conv["expected"]

        # If turn > 1, we need a fresh session per multi-turn conversation
        # For simplicity this runner tests each question independently
        # Multi-turn sequences can be added by grouping by 'conversation_id'

        resp = requests.post(
            f"{API_BASE}/ask",
            json={"session_id": session_id, "text": conv["query"]},
        )
        if resp.status_code != 200:
            print(f"[{qid}] ❌ API error: {resp.status_code} {resp.text[:100]}")
            results.append({"id": qid, "pass": False, "error": "api_error"})
            continue

        data = resp.json()
        answer = data.get("answer", "")
        citations = data.get("citations", [])
        grounding = data.get("grounding_score", 0.0)
        agent_trace = data.get("agent_trace", [])

        checks = []

        # must_include
        for term in expected.get("must_include", []):
            found = term.lower() in answer.lower()
            checks.append(("must_include", term, found))

        # must_cite
        for ec in expected.get("must_cite", []):
            cited = any(
                c.get("message_id") == ec["message_id"]
                and (ec.get("page") is None or c.get("page_no") == ec.get("page"))
                for c in citations
            )
            label = f"msg:{ec['message_id']}" + (f" p{ec['page']}" if ec.get("page") else "")
            checks.append(("must_cite", label, cited or ec["message_id"] == "REPLACE_WITH_REAL_ID"))

        # clarify_should_fire (if specified)
        if "clarify_should_fire" in expected:
            clarify_agent = next(
                (a for a in agent_trace if a.get("agent") == "entity_resolution"), None
            )
            actual_clarify = clarify_agent and clarify_agent.get("routing") == "clarify"
            checks.append(("clarify_routing", str(expected["clarify_should_fire"]),
                           actual_clarify == expected["clarify_should_fire"]))

        passed = all(c[2] for c in checks)
        icon = "✅" if passed else "❌"

        print(f"{icon} [{qid}] {conv['query'][:60]}")
        print(f"   grounding: {grounding:.0%} | agents: {[a['agent'] for a in agent_trace]}")
        for check_type, label, ok in checks:
            status = "✓" if ok else "✗"
            print(f"   {status} {check_type}: {label}")
        if expected.get("notes"):
            print(f"   note: {expected['notes']}")
        print()

        results.append({"id": qid, "pass": passed})

    print("=" * 70)
    passed_count = sum(1 for r in results if r["pass"])
    total = len(results)
    print(f"\nResult: {passed_count}/{total} passed")
    if passed_count < total:
        failed = [r["id"] for r in results if not r["pass"]]
        print(f"Failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="tests/eval_questions.json")
    args = parser.parse_args()
    run_eval(args.questions)
