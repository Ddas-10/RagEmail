import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['USE_TF'] = '0'
from agents.agents import entity_resolution_agent

def _base_state(**kw):
    s = {
        'session_id':'t','thread_id':'T-0001','turn_history':[],'search_outside_thread':False,
        'entity_register':{},'raw_query':'x','resolved_query':'x','clarify_needed':False,
        'clarify_text':None,'active_thread_ids':['T-0001'],'date_filter':None,'scope_hint':'both',
        'stepback_query':None,'hyde_query':None,'cluster_label':None,'expansion_queries':[],
        'sub_queries':[],'retrieved_chunks':[],'retrieval_attempt':1,'draft_answer':'',
        'citations':[],'grounding_score':0.0,'retrieval_insufficient':False,
        'is_temporal_query':False,'timeline_events':None,'final_answer':'','agent_trace':[],
    }
    s.update(kw); return s

queries = [
    "What did finance approve for the storage vendor?",
    "Who sent the contract?",
    "Summarise this thread.",
    "What is the total amount in the proposal?",
    "When was the last email sent?",
]
for q in queries:
    r = entity_resolution_agent(_base_state(raw_query=q))
    label = "CLARIFY" if r.get("clarify_needed") else "OK"
    print(f"[{label}] {q}")
