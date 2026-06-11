import sys
import pathlib
import time
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

print("Loading index...")
from ingest import load_index
load_index()

from agents.graph import run_turn

tests = [
    ("T-9114D5B0", "What does the attached document say about the succession plan?"),
    ("T-9114D5B0", "What is on page 1 of the attachment?"),
    ("T-F31891E0", "What does the briefing document say about the California energy crisis?"),
    ("T-F31891E0", "What are the key risks mentioned in the attached briefing?"),
]

for thread_id, query in tests:
    t0 = time.time()
    result = run_turn(
        session_id=f"att-test-{thread_id}",
        thread_id=thread_id,
        raw_query=query,
        turn_history=[],
        entity_register={},
        search_outside_thread=False,
    )
    ms = int((time.time() - t0) * 1000)

    print(f"\n{'='*60}")
    print(f"Thread : {thread_id}")
    print(f"Query  : {query}")
    print(f"Time   : {ms}ms  grounding: {result.get('grounding_score', 0):.0%}  citations: {len(result.get('citations', []))}")
    print(f"Cluster: {result.get('cluster_label', '?')}  scope: {result.get('scope_hint', '?')}")
    print(f"\nAnswer:\n{result.get('final_answer', 'NO ANSWER')[:500]}")
    print(f"\nCitations:")
    for c in result.get('citations', []):
        page = f", page={c['page_no']}" if c.get('page_no') else ""
        print(f"  [msg: {c['message_id'][:50]}{page}]  conf={c['confidence']:.0%}")
        print(f"  claim: {c['claim_text'][:80]}")
    att = [c for c in result.get('retrieved_chunks', []) if c.get('source') == 'attachment']
    print(f"\nAttachment chunks in top-k: {len(att)}")
    for c in att:
        print(f"  page={c.get('page_no')}  text: {c.get('text','')[:120]}")
