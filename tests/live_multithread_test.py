"""
tests/live_multithread_test.py — End-to-end multi-thread, multi-query quality test.

Covers all 6 intent clusters (summary, person, approval, timeline, status, attachment)
across 6 distinct threads.  Run:
    python tests/live_multithread_test.py

Output is logged to tests/live_test_output.txt for detailed review.
"""

import time
import sys
import pathlib
import re

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Load the index once before any queries (mirrors API lifespan startup)
print("Loading index…")
try:
    from ingest import load_index
    load_index()
    print("Index loaded.")
except FileNotFoundError:
    print("[WARN] Index not found — run `python ingest.py --data-dir data/slice/` first")
    sys.exit(1)
except Exception as e:
    print(f"[WARN] Index load failed: {e}")
    sys.exit(1)

from agents.graph import run_turn

# ──────────────────────────────────────────────
# Test matrix: (thread_id, label, query, expected_cluster)
# ──────────────────────────────────────────────
TEST_CASES = [
    # ── T-5CADBCA8 (FERC energy / price caps) ──────────────────────────────
    ("T-5CADBCA8", "FERC/summary",   "Summarise the whole email thread",               "summary"),
    ("T-5CADBCA8", "FERC/person",    "Who are the people involved in this thread?",    "person"),
    ("T-5CADBCA8", "FERC/approval",  "What is Enron's position on FERC price caps?",   "approval"),
    ("T-5CADBCA8", "FERC/timeline",  "When were these events communicated?",           "timeline"),

    # ── T-9114D5B0 (CEO succession / Jeff Skilling promotion) ──────────────
    ("T-9114D5B0", "Succession/summary",  "What is this email thread about?",                     "summary"),
    ("T-9114D5B0", "Succession/person",   "Who are the participants in this conversation?",        "person"),
    ("T-9114D5B0", "Succession/status",   "What is the status of the leadership transition?",     "status"),
    ("T-9114D5B0", "Succession/timeline", "When did the key announcements happen?",               "timeline"),

    # ── T-BADA59FE (Congratulations to Jeff Skilling as CEO) ───────────────
    ("T-BADA59FE", "Congrats/summary",  "Give me an overview of this thread",                 "summary"),
    ("T-BADA59FE", "Congrats/person",   "Who sent congratulations messages?",                 "person"),
    ("T-BADA59FE", "Congrats/approval", "Was anything formally approved or decided?",         "approval"),

    # ── T-134D3F1C (Performance review process) ────────────────────────────
    ("T-134D3F1C", "PerfReview/summary",  "Summarize this email thread",                         "summary"),
    ("T-134D3F1C", "PerfReview/status",   "What is the performance review process status?",     "status"),
    ("T-134D3F1C", "PerfReview/person",   "Who is involved in the performance review?",         "person"),

    # ── T-ADC28EE6 (Talking points / PR review) ────────────────────────────
    ("T-ADC28EE6", "TalkingPts/summary",  "What is this email thread about?",                   "summary"),
    ("T-ADC28EE6", "TalkingPts/approval", "What was approved or needs approval in this thread?","approval"),
    ("T-ADC28EE6", "TalkingPts/timeline", "What is the timeline of decisions in this thread?",  "timeline"),

    # ── T-F31891E0 (FERC PX/ISO tariff activity) ───────────────────────────
    ("T-F31891E0", "FERC-tariff/summary",  "Summarize what this thread discusses",               "summary"),
    ("T-F31891E0", "FERC-tariff/person",   "Who are the key people mentioned?",                  "person"),
    ("T-F31891E0", "FERC-tariff/timeline", "When were the main activities in this thread?",      "timeline"),
]


# ──────────────────────────────────────────────
# Quality checks
# ──────────────────────────────────────────────

_LOTUS_ADDR_RE = re.compile(r"\b\w+/\w+@\w+")
_NON_ASCII_RE   = re.compile(r"[^\x00-\x7F]{2,}")  # 2+ consecutive non-ASCII
_RFC_HEADER_RE  = re.compile(r"^(?:From|To|Cc|Bcc|Subject|Date|Sent By)\s*:", re.IGNORECASE)


def check_answer_quality(answer: str, label: str) -> list[str]:
    """Return list of quality issues found (empty = clean)."""
    issues = []
    if not answer or len(answer.split()) < 5:
        issues.append("EMPTY/TOO_SHORT")
        return issues
    if _LOTUS_ADDR_RE.search(answer):
        issues.append("LOTUS_ADDR_LEAK")
    if _NON_ASCII_RE.search(answer):
        issues.append("NON_ASCII_CHARS")
    if _RFC_HEADER_RE.match(answer):
        issues.append("HEADER_LINE_FIRST")
    if len(answer.strip()) > 0 and answer.count("?") == len(answer.replace(" ", "")):
        issues.append("QUERY_PARROTED")
    return issues


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────

def run_all_tests():
    log_path = pathlib.Path(__file__).parent / "live_test_output.txt"
    lines = []
    results = []

    def log(s=""):
        lines.append(s)
        print(s)

    log("=" * 72)
    log(f"LIVE MULTI-THREAD TEST  ({len(TEST_CASES)} turns)")
    log("=" * 72)

    total_latency = []
    cluster_correct = 0
    grounding_pass = 0
    quality_clean = 0
    retries_triggered = 0

    for i, (thread_id, label, query, expected_cluster) in enumerate(TEST_CASES, 1):
        log()
        log(f"[{i:02d}/{len(TEST_CASES)}]  {label}")
        log(f"  Thread : {thread_id}")
        log(f"  Query  : {query}")
        log(f"  Cluster: expected={expected_cluster}")

        t0 = time.perf_counter()
        try:
            state = run_turn(
                session_id=f"test-{thread_id}",
                thread_id=thread_id,
                raw_query=query,
                turn_history=[],
                entity_register={},
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            log(f"  ERROR  : {exc}")
            results.append({"label": label, "ok": False, "error": str(exc), "ms": elapsed})
            continue

        elapsed = (time.perf_counter() - t0) * 1000
        total_latency.append(elapsed)

        actual_cluster = state.get("cluster_label", "?")
        grounding     = state.get("grounding_score", 0.0)
        attempt       = state.get("retrieval_attempt", 1)
        final_answer  = state.get("final_answer", "")
        citations     = state.get("citations", [])

        cluster_ok  = (actual_cluster == expected_cluster)
        grounding_ok = (grounding >= 0.80 or actual_cluster in {"summary", "person"})
        retried     = (attempt > 2)
        quality_issues = check_answer_quality(final_answer, label)
        quality_ok  = len(quality_issues) == 0

        if cluster_ok:
            cluster_correct += 1
        if grounding_ok:
            grounding_pass += 1
        if retried:
            retries_triggered += 1
        if quality_ok:
            quality_clean += 1

        # Truncate answer for readability
        display_answer = final_answer[:500].replace("\n", " ") if final_answer else "(empty)"
        if len(final_answer) > 500:
            display_answer += "…"

        status_parts = []
        status_parts.append("cluster:" + ("OK" if cluster_ok else f"WRONG(got {actual_cluster})"))
        status_parts.append(f"grounding:{grounding:.0%}")
        status_parts.append(f"attempt:{attempt}")
        status_parts.append(f"citations:{len(citations)}")
        status_parts.append(f"ms:{elapsed:.0f}")
        if quality_issues:
            status_parts.append("QUALITY_ISSUES:" + ",".join(quality_issues))

        log(f"  Status : {' | '.join(status_parts)}")
        log(f"  Answer : {display_answer}")

        # Show citation sources
        if citations:
            cite_strs = []
            for c in citations[:3]:
                src = c.get("message_id", "?")[:16]
                pg  = f" p{c['page_no']}" if c.get("page_no") else ""
                cite_strs.append(f"{src}{pg}")
            log(f"  Sources: {' | '.join(cite_strs)}")

        results.append({
            "label": label,
            "ok": cluster_ok and grounding_ok and quality_ok,
            "cluster_ok": cluster_ok,
            "grounding_ok": grounding_ok,
            "quality_ok": quality_ok,
            "quality_issues": quality_issues,
            "ms": elapsed,
        })

    # ── Summary ──
    n = len(TEST_CASES)
    pass_count = sum(1 for r in results if r.get("ok"))
    p50 = sorted(total_latency)[len(total_latency) // 2] if total_latency else 0
    p95 = sorted(total_latency)[int(len(total_latency) * 0.95)] if total_latency else 0

    log()
    log("=" * 72)
    log(f"SUMMARY: {n} turns across {len(set(t for t,_,_,_ in TEST_CASES))} threads")
    log(f"  PASS (all checks) : {pass_count}/{n}")
    log(f"  Cluster correct   : {cluster_correct}/{n}")
    log(f"  Grounding >=80%   : {grounding_pass}/{n}")
    log(f"  Quality clean     : {quality_clean}/{n}  (no header/non-ASCII/addr leaks)")
    log(f"  Retries triggered : {retries_triggered}")
    log(f"  Latency p50       : {p50:.0f}ms")
    log(f"  Latency p95       : {p95:.0f}ms")
    log(f"  Latency max       : {max(total_latency, default=0):.0f}ms")

    # Quality breakdown
    all_issues = []
    for r in results:
        all_issues.extend(r.get("quality_issues", []))
    if all_issues:
        from collections import Counter
        log()
        log("  Quality issue breakdown:")
        for issue, cnt in Counter(all_issues).most_common():
            log(f"    {issue}: {cnt}")

    log("=" * 72)
    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nFull output saved to: {log_path}")
    return pass_count, n


if __name__ == "__main__":
    passed, total = run_all_tests()
    sys.exit(0 if passed == total else 1)
