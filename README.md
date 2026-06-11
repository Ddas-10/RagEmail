# Email + Attachment RAG with Agentic AI Pipeline



A focused RAG prototype for email threads with attachments, built on a **6-node LangGraph pipeline**. Every factual claim in every answer is grounded to a specific `message_id` and `page_no`. The system stays disciplined to the active thread by default and exposes every internal decision — rewrites, sub-queries, grounding scores, routing — in the API response and `trace.jsonl`.

---

## Quick Start

```bash
# 1. Clone and build
git clone <repo>
cd email_rag
docker compose build

# 2. Run the ingest pipeline FIRST (writes data/index/ to the shared volume)
# Enron slice is already pre-processed in data/slice/
docker compose run --rm ingest

# 3. Start the API + UI (api loads the index built in step 2)
docker compose up api ui

# 4. Open the UI
open http://localhost:8501

# 5. Or hit the API directly
curl -X POST http://localhost:8000/start_session \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "T-0001"}'
```

---

## Repository Layout

```
email_rag/
├── ingestion/
│   ├── parse_eml.py          # .eml → structured EmailRecord
│   ├── parse_attachments.py  # PDF/DOC/TXT/HTML → chunks with page_no
│   ├── chunker.py            # Late Chunking + naive fallback
│   ├── raptor.py             # RAPTOR tree builder per thread
│   └── thread_position.py   # Sinusoidal temporal position encoding
│
├── agents/
│   ├── agents.py             # All 8 agent functions (entity_resolution, metadata_prefilter,
│   │                         #   query_expansion, query_breakdown, citation_validator,
│   │                         #   timeline_synthesis, clarify, respond) + helpers
│   └── graph.py              # StateGraph wiring — the full pipeline
│
├── retrieval/
│   └── hybrid.py             # BM25 + FAISS + RRF fusion + cross-encoder reranker (unified)
│
├── api/
│   └── main.py               # FastAPI: /start_session /ask /switch_thread /reset_session
│
├── ui/
│   └── app.py                # Streamlit: thread selector, chat, agent debug panel
│
├── utils/
│   ├── tracer.py             # trace.jsonl writer — one record per agent invocation
│   ├── models.py             # Pydantic schemas: Chunk, Citation, SubQuery, PipelineState
│   ├── llm.py                # flan-t5-base wrapper (stepback, HyDE, grounding, synthesis)
│   └── intent_classifier.py  # SetFit intent classifier (6 clusters, trained)
│
├── tests/
│   ├── test_agents.py              # Unit tests: state isolation, routing logic
│   ├── test_retrieval.py           # Recall@k vs baseline on sample questions
│   ├── test_e2e.py                 # End-to-end pipeline tests
│   ├── live_multithread_test.py    # Live 20-turn multi-thread quality test
│   └── eval_questions.json         # 10 sample Q&A with expected citations + routing paths
│
├── data/
│   └── slice/                # Enron slice (see DATASET.md)
│
├── runs/                     # trace.jsonl files land here
├── ingest.py                 # Entry point for indexing pipeline
├── DATASET.md
├── docker-compose.yml
└── Dockerfile
```

---

## Architecture

### Pipeline Graph (LangGraph StateGraph)

```
user query
    │
    ▼
[entity_resolution]  ── clarify_needed=True ──► [clarify] ──► END
    │
    ▼  clarify_needed=False
[metadata_prefilter]   (rule-based: thread_id, date, scope)
    │
    ▼
[query_expansion]      (step-back → HyDE → cluster routing)
    │
    ▼
[query_breakdown]      (sub-query list with scope + priority)
    │
    ▼
[retrieval]            (BM25 + FAISS + RRF + reranker)
    │
    ▼
[citation_validator] ── retrieval_insufficient=True ──► [retrieval] (1× retry)
    │
    ▼  grounding_score ≥ 0.80
[timeline_synthesis]   (fires only on temporal queries — conditional)
    │
    ▼
[respond] ──► END
```

### Why each node exists

| Node | LLM call? | Purpose |
|---|---|---|
| `entity_resolution` | Yes (small) | Resolve "that document", "her email" → concrete IDs |
| `metadata_prefilter` | No | Pre-filter index to thread+date scope before any retrieval |
| `query_expansion` | Yes | Step-back abstraction + HyDE hypothetical doc generation |
| `query_breakdown` | Yes | Decompose compound questions into scoped sub-queries |
| `retrieval` | No | BM25 + FAISS + RRF fusion + cross-encoder rerank |
| `citation_validator` | Yes (small) | Audit claim→chunk grounding, compute score, trigger retry |
| `timeline_synthesis` | Yes (small) | Temporal questions → ordered event list with citations |
| `respond` | No | Assemble final answer + agent_trace for API response |

---

## Retrieval Design

### Late Chunking
Emails and threads are encoded as a full sequence through `bge-small-en-v1.5`, then chunk boundaries are applied *after* the transformer — so cross-sentence coreferences ("it", "that contract", "the attached file") are preserved in each chunk's embedding. Normal chunking breaks these.

### Query Expansion Stack
1. **Step-back**: LLM rewrites the resolved query to a broader principle (e.g. "what did finance approve?" → "what financial decisions were communicated in this thread?"). This hits RAPTOR intermediate nodes.
2. **HyDE**: LLM generates a 3-sentence hypothetical email that would answer the question. Embed it. Search by document-to-document similarity rather than query-to-document.
3. **Cluster routing**: Embed the resolved query, compute cosine similarity to 20 hand-labelled prototype queries across 5 intent clusters (`approval`, `timeline`, `person`, `attachment`, `status`). Winning cluster weights retrieval scope (e.g. `approval` cluster biases toward attachment chunks).

### Fusion
RRF over three ranked lists: BM25 results, vector results from HyDE embedding, vector results from resolved query embedding. `score(d) = Σ 1 / (60 + rank_i(d))`. Top-20 → cross-encoder reranker → top-8 to Agent 3.

### RAPTOR
One tree per thread. Leaf nodes = message chunks. Intermediate = LLM summaries of 3-5 clustered messages (UMAP + GMM clustering on embeddings). Root = full thread summary. At retrieval time, step-back queries hit intermediate/root nodes; specific queries hit leaf nodes.

---

## Chunking Parameters

| Source | Strategy | Chunk size | Overlap |
|---|---|---|---|
| Email body | Late chunking, per-message boundary | Full message | N/A |
| PDF attachment | Late chunking, sliding window | 300 tokens | 40 tokens |
| TXT/HTML attachment | Late chunking, paragraph boundary | 250 tokens | 30 tokens |
| RAPTOR intermediate | LLM summary, cluster-level | ~200 tokens | N/A |

---

## Citation Format

Every factual statement in every answer carries an inline citation:

```
The storage vendor contract was approved for $240,000 [msg: <m_9b2>].
The approval amount appears on page 2 of the attached proposal [msg: <m_9b2>, page: 2].
```

---

## Grounding

Agent 3 computes a token-level F1 overlap between each stated claim and its cited chunk. If `grounding_score < 0.80`, it triggers one retry with `top_k × 2`. The grounding score is exposed in every API response so the UI can display it.

---

## API

```
POST /start_session      { "thread_id": "T-0042" }
POST /ask                { "session_id": "...", "text": "..." }
POST /switch_thread      { "thread_id": "..." }
POST /reset_session

# /ask response shape:
{
  "answer": "...",
  "citations": [...],
  "rewrite": "resolved query after Agent 1",
  "retrieved": [{"doc_id": "...", "score": 0.87, "thread_id": "...", "page_no": 2}],
  "trace_id": "uuid",
  "agent_trace": [
    {"agent": "entity_resolution", "routing": "metadata_prefilter", "latency_ms": 45},
    {"agent": "query_expansion",   "strategies": ["stepback", "hyde"], "latency_ms": 180},
    {"agent": "query_breakdown",   "sub_queries": 2, "latency_ms": 55},
    {"agent": "citation_validator","grounding_score": 0.91, "retry": false, "latency_ms": 120}
  ]
}
```

---

## Performance Budget

| Stage | Target p95 |
|---|---|
| Agent 1 (entity resolution) | ≤ 180ms |
| Metadata prefilter | ≤ 10ms |
| Query expansion (step-back + HyDE) | ≤ 280ms |
| Agent 2 (query breakdown) | ≤ 120ms |
| Retrieval (BM25 + FAISS + RRF + rerank) | ≤ 600ms |
| Agent 3 (citation validator) | ≤ 400ms |
| **Total (no retry)** | **≤ 1.6s** |
| **Total (with retry)** | **≤ 2.8s** |

All LLM calls use `distilbart-cnn-12-6` or `t5-small` for entity resolution and citation validation. HyDE generation uses the same small model. No paid APIs required.

---

## Known Limitations

1. HyDE can hallucinate — if the small LLM generates a hypothetical document far from corpus vocabulary, retrieval degrades. Mitigated by fusing with direct query vector (RRF).
2. RAPTOR clustering quality depends on thread size — threads < 5 messages produce trivial summaries. Skipped for threads below threshold (min 5 messages).
3. Cross-encoder reranker (`bge-reranker-base`) is the bottleneck at ~300ms on CPU. On GPU this drops to ~40ms.
4. OCR for scanned PDFs (Tesseract) is slow — pre-processed at index time, not query time.
5. The entity register does not persist across sessions — starts fresh on `/start_session`.
6. **Latency on CPU exceeds spec.** Observed p50 ≈ 5–10s, p95 ≈ 15–50s on a CPU-only laptop (no GPU). The bottleneck is the query expansion step (flan-t5-base HyDE generation: ~8s cold, ~200ms warm via model caching) and the bge-reranker cross-encoder (~7s for 24-chunk batch on CPU). Warmup eliminates the model load cost after the first query. With a GPU (or ONNX-quantised models), the target ≤ 3.5s p95 is achievable. Optimisation path: (a) ONNX-quantise both models at int8, (b) replace full HyDE with rule-based template expansion when the intent is `summary` or `person`, (c) reduce reranker batch size by pre-filtering BM25 candidates more aggressively.

---

## How to Test

```bash
# Run the 10 sample Q&A eval
python tests/run_eval.py --questions tests/eval_questions.json --session

# Run agent unit tests
pytest tests/test_agents.py -v

# Run retrieval recall test
pytest tests/test_retrieval.py -v
```

---

## Models Used (all free/OSS)

| Model | Use |
|---|---|
| `bge-small-en-v1.5` | Chunk embeddings (late chunking) |
| `bge-reranker-base` | Cross-encoder reranking |
| `t5-small` | Entity resolution rewrite, step-back abstraction |
| `distilbart-cnn-12-6` | HyDE hypothetical generation, RAPTOR summaries |
| `en_core_web_sm` | spaCy NER for entity extraction |

---

## See Also

- `DATASET.md` — dataset source, slice selection, preprocessing
- `tests/eval_questions.json` — 10 sample questions with expected citations and routing paths
- `runs/20260610T174442/trace.jsonl` — submission trace: 8 turns across 3 sessions, all 3 mandatory agents + bonus agents, showing retry (grounding 50%→100%), timeline synthesis, and attachment citation
