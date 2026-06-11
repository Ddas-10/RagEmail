"""
tests/test_retrieval.py — Retrieval unit tests.

Tests cover:
  - BM25 scoring and scope filtering
  - RRF fusion correctness
  - Reranker fallback when model is None
  - Thread isolation (only returns chunks from the correct thread)
  - Retry behaviour: top_k doubles on attempt 2
  - retrieval_node state field isolation

Run: pytest tests/test_retrieval.py -v
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_chunk(doc_id, thread_id, message_id, source="email", page_no=None, text="sample text"):
    return {
        "doc_id":     doc_id,
        "thread_id":  thread_id,
        "message_id": message_id,
        "source":     source,
        "text":       text,
        "page_no":    page_no,
        "rrf_score":  0.0,
        "rerank_score": 0.0,
    }


def _base_retrieval_state(**overrides):
    state = {
        "session_id":          "test-session",
        "thread_id":           "T-0001",
        "turn_history":        [],
        "search_outside_thread": False,
        "entity_register":     {},
        "raw_query":           "what was approved?",
        "resolved_query":      "what was approved?",
        "clarify_needed":      False,
        "clarify_text":        None,
        "active_thread_ids":   ["T-0001"],
        "date_filter":         None,
        "scope_hint":          "both",
        "stepback_query":      "what financial decisions were communicated",
        "hyde_query":          "Subject: Re: approval\n\nThe approval was granted for the storage vendor.",
        "cluster_label":       "approval",
        "expansion_queries":   ["what was approved?", "what financial decisions were communicated"],
        "sub_queries":         [{"id": "sq-0", "text": "what was approved?", "scope": "both", "priority": 1}],
        "retrieved_chunks":    [],
        "retrieval_attempt":   1,
        "draft_answer":        "",
        "citations":           [],
        "grounding_score":     0.0,
        "retrieval_insufficient": False,
        "is_temporal_query":   False,
        "timeline_events":     None,
        "final_answer":        "",
        "agent_trace":         [],
    }
    state.update(overrides)
    return state


# ──────────────────────────────────────────────
# RRF fusion
# ──────────────────────────────────────────────

class TestRRFFusion:

    def test_rrf_combines_two_lists(self):
        from retrieval.hybrid import _rrf_fuse
        # List A: [0, 1, 2], List B: [2, 0, 1]
        list_a = [(0, 0.9), (1, 0.8), (2, 0.7)]
        list_b = [(2, 0.95), (0, 0.85), (1, 0.75)]
        fused = _rrf_fuse([list_a, list_b], source_tag=[])
        ids = [idx for idx, _ in fused]
        # idx 0 appears at rank 1 in A and rank 2 in B — should be near top
        assert 0 in ids[:2]

    def test_rrf_handles_empty_list(self):
        from retrieval.hybrid import _rrf_fuse
        result = _rrf_fuse([], source_tag=[])
        assert result == []

    def test_rrf_single_list(self):
        from retrieval.hybrid import _rrf_fuse
        ranked = [(5, 0.9), (3, 0.7), (1, 0.5)]
        fused = _rrf_fuse([ranked], source_tag=[])
        ids = [idx for idx, _ in fused]
        # Order must be preserved for a single list
        assert ids[0] == 5

    def test_rrf_score_decreases_with_rank(self):
        from retrieval.hybrid import _rrf_fuse, RRF_K
        ranked = [(0, 1.0), (1, 0.9), (2, 0.8)]
        fused = dict(_rrf_fuse([ranked], source_tag=[]))
        # Score for rank-1 item > rank-2 item
        assert fused[0] > fused[1] > fused[2]

    def test_rrf_k_constant(self):
        from retrieval.hybrid import RRF_K
        assert RRF_K == 60


# ──────────────────────────────────────────────
# BM25 search with thread filtering
# ──────────────────────────────────────────────

class TestBM25Search:

    def _build_mock_bm25(self, scores):
        """Return a mock BM25 object that always returns 'scores'."""
        mock = MagicMock()
        mock.get_scores.return_value = np.array(scores, dtype=float)
        return mock

    def test_thread_filter_excludes_other_threads(self):
        from retrieval import hybrid
        chunks = [
            _make_chunk("c0", "T-0001", "m1", text="storage vendor approval"),
            _make_chunk("c1", "T-0002", "m2", text="unrelated thread content"),
        ]
        hybrid._bm25_index = self._build_mock_bm25([0.8, 0.9])
        hybrid._bm25_chunks = chunks

        results = hybrid._bm25_search("storage vendor", ["T-0001"], "both", top_n=10)
        returned_indices = [i for i, _ in results]
        assert 0 in returned_indices       # T-0001 chunk
        assert 1 not in returned_indices   # T-0002 chunk excluded

    def test_scope_filter_email_only(self):
        from retrieval import hybrid
        chunks = [
            _make_chunk("c0", "T-0001", "m1", source="email",      text="approval sent via email"),
            _make_chunk("c1", "T-0001", "m2", source="attachment", text="contract document pdf"),
        ]
        hybrid._bm25_index = self._build_mock_bm25([0.7, 0.8])
        hybrid._bm25_chunks = chunks

        results = hybrid._bm25_search("approval", ["T-0001"], "email", top_n=10)
        returned_indices = [i for i, _ in results]
        assert 0 in returned_indices       # email chunk
        assert 1 not in returned_indices   # attachment chunk excluded

    def test_empty_index_returns_empty(self):
        from retrieval import hybrid
        hybrid._bm25_index = None
        results = hybrid._bm25_search("anything", ["T-0001"], "both", top_n=10)
        assert results == []

    def test_results_sorted_descending(self):
        from retrieval import hybrid
        chunks = [
            _make_chunk("c0", "T-0001", "m1", text="low relevance text"),
            _make_chunk("c1", "T-0001", "m2", text="high relevance storage vendor approval"),
            _make_chunk("c2", "T-0001", "m3", text="medium relevance storage"),
        ]
        hybrid._bm25_index = self._build_mock_bm25([0.3, 0.9, 0.6])
        hybrid._bm25_chunks = chunks

        results = hybrid._bm25_search("storage vendor", ["T-0001"], "both", top_n=3)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


# ──────────────────────────────────────────────
# Reranker fallback
# ──────────────────────────────────────────────

class TestReranker:

    def test_reranker_none_returns_top_k(self):
        from retrieval import hybrid
        hybrid._reranker = None
        chunks = [_make_chunk(f"c{i}", "T-0001", f"m{i}") for i in range(10)]
        result = hybrid._rerank("query", chunks, top_k=5)
        assert len(result) == 5

    def test_reranker_called_with_pairs(self):
        from retrieval import hybrid
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.9, 0.7, 0.5]
        hybrid._reranker = mock_reranker

        chunks = [_make_chunk(f"c{i}", "T-0001", f"m{i}", text=f"text {i}") for i in range(3)]
        result = hybrid._rerank("my query", chunks, top_k=3)

        pairs_used = mock_reranker.predict.call_args[0][0]
        assert all(isinstance(p, tuple) and p[0] == "my query" for p in pairs_used)
        assert len(result) == 3
        # Results should be sorted by rerank score descending
        scores = [c["rerank_score"] for c in result]
        assert scores == sorted(scores, reverse=True)


# ──────────────────────────────────────────────
# retrieval_node state isolation + retry
# ──────────────────────────────────────────────

class TestRetrievalNode:

    def test_only_writes_designated_fields(self):
        from retrieval.hybrid import retrieval_node
        allowed = {"retrieved_chunks", "retrieval_attempt"}
        state = _base_retrieval_state()

        # Patch internals to avoid needing a real index
        with patch("retrieval.hybrid._bm25_search", return_value=[]):
            with patch("retrieval.hybrid._vector_search", return_value=[]):
                result = retrieval_node(state)

        assert set(result.keys()).issubset(allowed), \
            f"retrieval_node wrote unexpected fields: {set(result.keys()) - allowed}"

    def test_attempt_incremented(self):
        from retrieval.hybrid import retrieval_node
        state = _base_retrieval_state(retrieval_attempt=1)

        with patch("retrieval.hybrid._bm25_search", return_value=[]):
            with patch("retrieval.hybrid._vector_search", return_value=[]):
                result = retrieval_node(state)

        assert result["retrieval_attempt"] == 2

    def test_top_k_doubles_on_retry(self):
        """On attempt 2, top_k should be 16 (8 * 2)."""
        from retrieval import hybrid

        captured_top_k = {}

        original_rerank = hybrid._rerank
        def capture_rerank(query, candidates, top_k=8):
            captured_top_k["top_k"] = top_k
            return candidates[:top_k]

        state = _base_retrieval_state(retrieval_attempt=2)

        with patch("retrieval.hybrid._bm25_search", return_value=[]):
            with patch("retrieval.hybrid._vector_search", return_value=[]):
                with patch.object(hybrid, "_rerank", side_effect=capture_rerank):
                    hybrid.retrieval_node(state)

        assert captured_top_k.get("top_k") == 16

    def test_empty_sub_queries_returns_empty(self):
        from retrieval.hybrid import retrieval_node
        state = _base_retrieval_state(sub_queries=[])

        with patch("retrieval.hybrid._bm25_search", return_value=[]):
            with patch("retrieval.hybrid._vector_search", return_value=[]):
                result = retrieval_node(state)

        assert result["retrieved_chunks"] == []


# ──────────────────────────────────────────────
# Thread discipline
# ──────────────────────────────────────────────

class TestThreadDiscipline:

    def test_cross_thread_disabled_by_default(self):
        """search_outside_thread=False → active_thread_ids = [thread_id]."""
        from agents.agents import metadata_prefilter_node
        from tests.test_agents import _base_state

        state = _base_state(search_outside_thread=False, thread_id="T-ONLY")
        result = metadata_prefilter_node(state)
        assert result["active_thread_ids"] == ["T-ONLY"]

    def test_cross_thread_enabled_unlocks_all(self):
        """search_outside_thread=True → active_thread_ids = [] (all threads)."""
        from agents.agents import metadata_prefilter_node
        from tests.test_agents import _base_state

        state = _base_state(search_outside_thread=True, thread_id="T-ONLY")
        result = metadata_prefilter_node(state)
        assert result["active_thread_ids"] == []
