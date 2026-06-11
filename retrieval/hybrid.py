"""
retrieval/hybrid.py — Retrieval node for the LangGraph pipeline.

Execution per sub-query:
  1. BM25 search (rank_bm25) scoped to active_thread_ids
  2. Vector search (FAISS) with three query variants:
       a. resolved_query embedding
       b. stepback_query embedding
       c. hyde_query embedding (document-to-document matching)
  3. Thread synopsis query (new): one extra FAISS search using the
     temporally-diverse synopsis vector computed at index time.
     This ensures representative content always scores well in RRF,
     even for meta-queries like "summarise thread" where BM25 has no match.
  4. RRF fusion of all ranked lists
  5. For summary intent: MMR selection (Maximum Marginal Relevance) to pick
     diverse representative chunks from across the thread arc.
     For specific queries: cross-encoder reranker (bge-reranker-base) on top-20.

The retrieval_attempt counter is checked here:
  - attempt 1: top_k = 8 (10 for summary mode)
  - attempt 2 (retry): top_k doubles
"""

from __future__ import annotations
import os
import math
from collections import defaultdict
from typing import Any

# Prevent TF/Keras import clash on Windows before any heavy imports
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from utils.models import PipelineState, Chunk
from utils.tracer import AgentTimer, log_agent

# ─────────────────────────────────────────────────
# Index singletons (loaded once at startup by ingest.py)
# ─────────────────────────────────────────────────

_bm25_index: Any = None          # rank_bm25.BM25Okapi
_bm25_chunks: list[dict] = []    # parallel list of chunk dicts

_faiss_index: Any = None         # faiss.IndexFlatIP
_faiss_chunks: list[dict] = []   # parallel list of chunk dicts

_embed_model: Any = None
_reranker: Any = None

# Thread synopsis vectors: thread_id → position-aware representative embedding.
# Built during ingest from temporally-diverse chunk representatives — not biased
# toward the most common email type (usually farewell replies in announcement threads).
_thread_synopses: dict[str, np.ndarray] = {}

# doc_id → FAISS matrix row index; built from faiss_chunks in load_indices.
# Needed by _mmr_select to retrieve embeddings for MMR without re-encoding.
_doc_id_to_faiss_idx: dict[str, int] = {}


def load_indices(bm25_index, bm25_chunks, faiss_index, faiss_chunks, embed_model, reranker):
    """Called once from ingest.py / startup."""
    global _bm25_index, _bm25_chunks, _faiss_index, _faiss_chunks
    global _embed_model, _reranker, _doc_id_to_faiss_idx
    _bm25_index   = bm25_index
    _bm25_chunks  = bm25_chunks
    _faiss_index  = faiss_index
    _faiss_chunks = faiss_chunks
    _embed_model  = embed_model
    _reranker     = reranker
    _doc_id_to_faiss_idx = {c["doc_id"]: i for i, c in enumerate(faiss_chunks)}


def load_thread_synopses(synopses: dict[str, list]):
    """Load pre-computed thread synopsis vectors (called from ingest.load_index)."""
    global _thread_synopses
    _thread_synopses = {k: np.array(v, dtype="float32") for k, v in synopses.items()}


def get_thread_chunks(thread_ids: list[str]) -> list[dict]:
    """
    Return ALL BM25-indexed chunks for the given thread_ids.
    Used by the summary-query path so _build_thread_summary has full coverage.
    RAPTOR summary chunks come last (they are often garbled by distilbart).
    """
    if not _bm25_chunks:
        return []
    matching = [dict(c) for c in _bm25_chunks if c["thread_id"] in thread_ids]
    email_chunks  = [c for c in matching if "__raptor__" not in c.get("doc_id", "")]
    raptor_chunks = [c for c in matching if "__raptor__"     in c.get("doc_id", "")]
    return email_chunks + raptor_chunks


# ─────────────────────────────────────────────────
# BM25 retrieval
# ─────────────────────────────────────────────────

def _bm25_search(
    query: str,
    thread_ids: list[str],
    scope: str,
    top_n: int = 20,
) -> list[tuple[int, float]]:
    """Returns list of (chunk_index, score) sorted descending."""
    if _bm25_index is None:
        return []

    tokens = query.lower().split()
    scores = _bm25_index.get_scores(tokens)

    filtered = []
    for i, chunk in enumerate(_bm25_chunks):
        if thread_ids and chunk["thread_id"] not in thread_ids:
            continue
        if scope != "both" and chunk.get("source") != scope:
            continue
        filtered.append((i, float(scores[i])))

    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered[:top_n]


# ─────────────────────────────────────────────────
# Vector retrieval
# ─────────────────────────────────────────────────

def _vector_search(
    query: str,
    thread_ids: list[str],
    scope: str,
    top_n: int = 20,
) -> list[tuple[int, float]]:
    """FAISS search with a text query (encodes via _embed_model)."""
    if _faiss_index is None or _embed_model is None:
        return []

    q_emb = _embed_model.encode(query, normalize_embeddings=True).reshape(1, -1).astype("float32")
    return _vector_search_with_emb(q_emb[0], thread_ids, scope, top_n)


def _vector_search_with_emb(
    q_emb: np.ndarray,
    thread_ids: list[str],
    scope: str,
    top_n: int = 20,
) -> list[tuple[int, float]]:
    """
    FAISS search with a pre-computed, unit-normalised embedding vector.

    Used for thread synopsis queries where the embedding is pre-computed
    at index time rather than derived from a text query at runtime.
    """
    if _faiss_index is None:
        return []

    q = q_emb.reshape(1, -1).astype("float32")
    scores_arr, indices_arr = _faiss_index.search(q, top_n * 3)

    results = []
    for score, idx in zip(scores_arr[0], indices_arr[0]):
        if idx < 0 or idx >= len(_faiss_chunks):
            continue
        chunk = _faiss_chunks[idx]
        if thread_ids and chunk["thread_id"] not in thread_ids:
            continue
        if scope != "both" and chunk.get("source") != scope:
            continue
        results.append((int(idx), float(score)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


# ─────────────────────────────────────────────────
# Reciprocal Rank Fusion
# ─────────────────────────────────────────────────

RRF_K = 60  # standard smoothing constant


def _rrf_fuse(
    ranked_lists: list[list[tuple[int, float]]],
    source_tag: list[str],
) -> list[tuple[int, float]]:
    """Fuse N ranked lists with RRF: score(d) = Σ_i  1 / (RRF_K + rank_i(d))."""
    rrf_scores: dict[int, float] = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, (chunk_idx, _) in enumerate(ranked_list, start=1):
            rrf_scores[chunk_idx] += 1.0 / (RRF_K + rank)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────────
# Maximum Marginal Relevance
# ─────────────────────────────────────────────────

def _mmr_select(
    query_emb: np.ndarray,
    candidates: list[dict],
    k: int = 10,
    lambda_: float = 0.5,
) -> list[dict]:
    """
    Maximum Marginal Relevance selection (Carbonell & Goldstein, 1998).

    Selects chunks that are both relevant to the query AND diverse from each
    other. Compared to pure reranker top-k (which clusters around the single
    most relevant topic), MMR ensures the answer draws from different parts
    of the thread — essential for comprehensive thread summaries.

    score(d) = lambda * sim(d, query) - (1 - lambda) * max_{s∈selected} sim(d, s)

    lambda_=0.5: equal weight on relevance and diversity.
    lambda_=0.6: slightly favours relevance (for specific-answer queries).

    Embeddings are retrieved from the FAISS index via reconstruct() — no
    re-encoding needed.
    """
    if not candidates or _faiss_index is None:
        return candidates[:k]

    # Retrieve embeddings for all candidates from FAISS
    embs: list[np.ndarray] = []
    for c in candidates:
        faiss_i = _doc_id_to_faiss_idx.get(c.get("doc_id", ""))
        if faiss_i is not None and 0 <= faiss_i < _faiss_index.ntotal:
            embs.append(_faiss_index.reconstruct(faiss_i).astype("float32"))
        elif _embed_model is not None:
            embs.append(_embed_model.encode(c["text"], normalize_embeddings=True).astype("float32"))
        else:
            embs.append(np.zeros_like(query_emb, dtype="float32"))

    emb_arr = np.array(embs, dtype="float32")
    # Re-normalise (FAISS reconstruct returns stored vectors, already normalised)
    norms = np.linalg.norm(emb_arr, axis=1, keepdims=True) + 1e-9
    emb_arr = emb_arr / norms

    q = query_emb.astype("float32")
    norm_q = np.linalg.norm(q) + 1e-9
    q = q / norm_q

    selected: list[dict] = []
    selected_embs: list[np.ndarray] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(k, len(candidates))):
        if not remaining:
            break

        if not selected_embs:
            # First pick: pure relevance
            sims = np.array([float(emb_arr[i] @ q) for i in remaining])
            best_i = remaining[int(np.argmax(sims))]
        else:
            sel_arr = np.array(selected_embs, dtype="float32")  # (s, D)
            scores: list[float] = []
            for i in remaining:
                rel      = float(emb_arr[i] @ q)
                max_sim  = float(np.max(sel_arr @ emb_arr[i]))
                scores.append(lambda_ * rel - (1.0 - lambda_) * max_sim)
            best_i = remaining[int(np.argmax(scores))]

        selected.append(candidates[best_i])
        selected_embs.append(emb_arr[best_i])
        remaining.remove(best_i)

    return selected


# ─────────────────────────────────────────────────
# Cross-encoder reranker
# ─────────────────────────────────────────────────

def _rerank(
    query: str,
    candidate_chunks: list[dict],
    top_k: int = 8,
) -> list[dict]:
    """Rerank with bge-reranker-base cross-encoder; falls back to RRF order."""
    if _reranker is None:
        return candidate_chunks[:top_k]

    pairs  = [(query, c["text"]) for c in candidate_chunks]
    scores = _reranker.predict(pairs)

    ranked = sorted(zip(candidate_chunks, scores), key=lambda x: x[1], reverse=True)
    result = []
    for chunk, score in ranked[:top_k]:
        chunk = dict(chunk)
        chunk["rerank_score"] = float(score)
        result.append(chunk)
    return result


# ─────────────────────────────────────────────────
# Main retrieval node
# ─────────────────────────────────────────────────

def retrieval_node(state: PipelineState) -> dict:
    """
    Retrieval node — BM25 + FAISS + thread synopsis query + RRF + reranker/MMR.

    For every query the temporally-diverse thread synopsis is added as an extra
    FAISS ranked list in RRF.  This ensures representative thread content always
    has a path into the candidate pool regardless of keyword match — replacing the
    previous hardcoded 'if summarize_thread: dump all chunks' bypass.

    Summary intent (cluster_label == "summary"):
      - wider top_k (10 vs 8)
      - MMR selection instead of reranker, queried with the synopsis embedding
        (temporal diversity emphasis, lambda=0.5)

    Specific queries:
      - cross-encoder reranker on top-20 candidates (precision emphasis)

    On retry (retrieval_attempt >= 2): top_k doubles.
    """
    sub_queries = state.get("sub_queries", [])
    thread_ids  = state.get("active_thread_ids", [state["thread_id"]])
    attempt     = state.get("retrieval_attempt", 1)
    is_summary  = state.get("cluster_label", "") == "summary"

    base_top_k = 10 if is_summary else 8
    top_k      = base_top_k * (2 if attempt >= 2 else 1)
    # Pool size sent to the cross-encoder. bge-reranker-base latency scales linearly
    # with candidate count; 3× top_k gives the reranker meaningful room to re-order
    # while keeping CPU latency roughly half of the old 5× pool.
    # BM25-pinned chunks (score ≥ 5.0) are appended after RRF and also reach the reranker.
    retrieve_n = top_k * 3

    with AgentTimer() as t:
        all_ranked_lists: list[list[tuple[int, float]]] = []
        all_chunk_pools: dict[int, dict] = {}

        for sq in sub_queries:
            scope   = sq.get("scope", "both")
            sq_text = sq.get("text", "")

            # BM25: resolved + step-back
            bm25_queries = [sq_text]
            if state.get("stepback_query"):
                bm25_queries.append(state["stepback_query"])

            for q in bm25_queries:
                bm25_results = _bm25_search(q, thread_ids, scope, retrieve_n)
                all_ranked_lists.append(bm25_results)
                for idx, score in bm25_results:
                    if idx not in all_chunk_pools:
                        c = dict(_bm25_chunks[idx])
                        c["bm25_score"] = score
                        all_chunk_pools[idx] = c

            # Vector: resolved + step-back + HyDE
            vector_queries = [sq_text]
            if state.get("stepback_query"):
                vector_queries.append(state["stepback_query"])
            if state.get("hyde_query"):
                vector_queries.append(state["hyde_query"])

            for q in vector_queries:
                vec_results = _vector_search(q, thread_ids, scope, retrieve_n)
                all_ranked_lists.append(vec_results)
                for idx, score in vec_results:
                    if idx not in all_chunk_pools:
                        c = dict(_faiss_chunks[idx])
                        c["vector_score"] = score
                        all_chunk_pools[idx] = c

        # ── Thread synopsis query ──────────────────────────────────────────
        # One extra FAISS ranked list per thread using the pre-computed synopsis
        # vector (temporally-diverse mean of 5 quantile-sampled CLS embeddings).
        # This surfaces representative content for any query — including meta-
        # queries like "summarise the thread" where BM25/vector have no match —
        # without any hardcoded keyword detection.
        synopsis_hit = False
        if _thread_synopses:
            for tid in thread_ids:
                if tid in _thread_synopses:
                    synopsis_results = _vector_search_with_emb(
                        _thread_synopses[tid], [tid], "email", retrieve_n
                    )
                    if synopsis_results:
                        all_ranked_lists.append(synopsis_results)
                        synopsis_hit = True
                        for idx, score in synopsis_results:
                            if idx not in all_chunk_pools:
                                c = dict(_faiss_chunks[idx])
                                c["vector_score"] = score
                                all_chunk_pools[idx] = c
        # ──────────────────────────────────────────────────────────────────

        # RRF fusion
        fused = _rrf_fuse(all_ranked_lists, source_tag=[])
        candidates = []
        candidate_idxs: set[int] = set()
        for idx, rrf_score in fused[:retrieve_n]:
            if idx in all_chunk_pools:
                chunk = all_chunk_pools[idx]
                chunk["rrf_score"] = rrf_score
                candidates.append(chunk)
                candidate_idxs.add(idx)

        # Guarantee high-BM25 chunks make it to the reranker — RRF can dilute
        # a strong BM25 hit (e.g. an exact keyword match in the document title)
        # if it doesn't appear in many vector lists.  We inject any chunk whose
        # raw BM25 score exceeds a threshold, even if it ranked low in RRF.
        BM25_PINNED_THRESHOLD = 5.0
        for idx, chunk in all_chunk_pools.items():
            if idx not in candidate_idxs and chunk.get("bm25_score", 0) >= BM25_PINNED_THRESHOLD:
                candidates.append(chunk)
                candidate_idxs.add(idx)

        primary_query = sub_queries[0]["text"] if sub_queries else state["resolved_query"]

        if is_summary:
            # For summary queries: use MMR with synopsis as the diversity anchor.
            # Synopsis embedding reflects the thread's temporal arc (not just its
            # semantic centroid), so MMR will pick chunks spread across the thread.
            primary_tid = thread_ids[0] if thread_ids else ""
            if primary_tid in _thread_synopses:
                mmr_query_emb = _thread_synopses[primary_tid]
            elif _embed_model is not None:
                mmr_query_emb = _embed_model.encode(
                    primary_query, normalize_embeddings=True
                ).astype("float32")
            else:
                mmr_query_emb = np.zeros(1, dtype="float32")
            final_chunks = _mmr_select(mmr_query_emb, candidates, k=top_k, lambda_=0.5)
        else:
            # Specific queries: cross-encoder reranker for precision
            final_chunks = _rerank(primary_query, candidates, top_k=top_k)

    log_agent(
        session_id=state["session_id"],
        agent="retrieval",
        turn=len(state.get("turn_history", [])),
        inputs={
            "sub_queries": [sq.get("text", "") for sq in sub_queries],
            "attempt":      attempt,
            "top_k":        top_k,
            "is_summary":   is_summary,
            "synopsis_hit": synopsis_hit,
        },
        outputs={
            "n_candidates_before_selection": len(candidates),
            "n_returned": len(final_chunks),
            "selection":  "mmr" if is_summary else "reranker",
            "top_scores": [round(c.get("rerank_score", c.get("rrf_score", 0)), 3)
                           for c in final_chunks[:3]],
        },
        routing="citation_validator",
        latency_ms=t.latency_ms,
        token_count=0,
    )

    return {
        "retrieved_chunks": [dict(c) for c in final_chunks],
        "retrieval_attempt": attempt + 1,
    }
