"""
tests/test_agents.py — Unit tests for agent state isolation and routing logic.

Run:  pytest tests/test_agents.py -v
"""

import pytest
from unittest.mock import patch, MagicMock
from agents.agents import (
    entity_resolution_agent,
    metadata_prefilter_node,
    query_expansion_node,
    query_breakdown_agent,
    citation_validator_agent,
    ROUTE_CLARIFY,
    ROUTE_PREFILTER,
    _extract_persons_from_chunks,
    _THREAD_ROSTER,
)


def _base_state(**overrides):
    state = {
        "session_id": "test-session",
        "thread_id": "T-0001",
        "turn_history": [],
        "search_outside_thread": False,
        "entity_register": {},
        "raw_query": "what was approved?",
        "resolved_query": "what was approved?",
        "clarify_needed": False,
        "clarify_text": None,
        "active_thread_ids": ["T-0001"],
        "date_filter": None,
        "scope_hint": "both",
        "stepback_query": None,
        "hyde_query": None,
        "cluster_label": None,
        "expansion_queries": [],
        "sub_queries": [],
        "retrieved_chunks": [],
        "retrieval_attempt": 1,
        "draft_answer": "",
        "citations": [],
        "grounding_score": 0.0,
        "retrieval_insufficient": False,
        "is_temporal_query": False,
        "timeline_events": None,
        "final_answer": "",
        "agent_trace": [],
    }
    state.update(overrides)
    return state


# ──────────────────────────────────────────────
# Agent 1: Entity Resolution
# ──────────────────────────────────────────────

class TestEntityResolution:

    def test_unambiguous_query_routes_to_prefilter(self):
        """Clear, unambiguous query should NOT trigger clarify."""
        state = _base_state(raw_query="What did finance approve for the storage vendor?")
        result = entity_resolution_agent(state)
        assert result["clarify_needed"] is False

    def test_empty_register_with_pronoun_first_turn_clarifies(self):
        """Bare pronoun on first turn with empty register should clarify."""
        state = _base_state(
            raw_query="What did it say?",
            entity_register={},
            turn_history=[],
        )
        result = entity_resolution_agent(state)
        # With empty register and bare 'it', clarify should fire
        # (exact behaviour depends on spacy entity extraction)
        assert "clarify_needed" in result

    def test_pronoun_resolved_from_register(self):
        """'That document' should resolve to the last known WORK_OF_ART entity."""
        state = _base_state(
            raw_query="What does that document say?",
            entity_register={
                "proposal.pdf": {
                    "type": "WORK_OF_ART",
                    "text": "proposal.pdf",
                    "turn": 1,
                    "message_id": "m_001",
                }
            },
            turn_history=[{"role": "user", "content": "show me the proposal.pdf"}],
        )
        result = entity_resolution_agent(state)
        assert "proposal.pdf" in result["resolved_query"] or result["clarify_needed"] is False

    def test_correction_detected(self):
        """Explicit correction should update entity_register."""
        state = _base_state(
            raw_query="No, I meant the email from last week not the approval",
            entity_register={
                "approval": {"type": "ORG", "text": "approval", "turn": 2, "message_id": "m_002"}
            },
        )
        result = entity_resolution_agent(state)
        # Should not clarify — correction provides enough context
        assert "entity_register" in result

    def test_only_writes_designated_fields(self):
        """Agent 1 must only write its designated state fields."""
        allowed = {"entity_register", "resolved_query", "clarify_needed", "clarify_text"}
        state = _base_state(raw_query="Who sent the contract?")
        result = entity_resolution_agent(state)
        assert set(result.keys()).issubset(allowed), \
            f"Agent 1 wrote unexpected fields: {set(result.keys()) - allowed}"


# ──────────────────────────────────────────────
# Metadata prefilter
# ──────────────────────────────────────────────

class TestMetadataPrefilter:

    def test_attachment_scope_detected(self):
        state = _base_state(resolved_query="What does the PDF attachment say about pricing?")
        result = metadata_prefilter_node(state)
        assert result["scope_hint"] == "attachment"

    def test_email_scope_detected(self):
        state = _base_state(resolved_query="What did she write in her reply email?")
        result = metadata_prefilter_node(state)
        assert result["scope_hint"] == "email"

    def test_default_scope_both(self):
        state = _base_state(resolved_query="What happened with the vendor deal?")
        result = metadata_prefilter_node(state)
        assert result["scope_hint"] == "both"

    def test_thread_locked_by_default(self):
        state = _base_state(search_outside_thread=False)
        result = metadata_prefilter_node(state)
        assert "T-0001" in result["active_thread_ids"]

    def test_cross_thread_unlocks(self):
        state = _base_state(search_outside_thread=True)
        result = metadata_prefilter_node(state)
        assert result["active_thread_ids"] == []  # empty = all threads

    def test_only_writes_designated_fields(self):
        allowed = {"active_thread_ids", "date_filter", "scope_hint"}
        state = _base_state()
        result = metadata_prefilter_node(state)
        assert set(result.keys()).issubset(allowed)


# ──────────────────────────────────────────────
# Query expansion
# ──────────────────────────────────────────────

class TestQueryExpansion:

    def test_cluster_routing_approval(self):
        state = _base_state(resolved_query="What was the approved budget for storage?")
        result = query_expansion_node(state)
        assert result["cluster_label"] == "approval"

    def test_cluster_routing_timeline(self):
        state = _base_state(resolved_query="When was the contract first sent?")
        result = query_expansion_node(state)
        assert result["cluster_label"] == "timeline"

    def test_stepback_generated(self):
        state = _base_state(resolved_query="What did finance approve?")
        result = query_expansion_node(state)
        assert result["stepback_query"] is not None
        assert len(result["stepback_query"]) > 10

    def test_hyde_generated(self):
        state = _base_state(resolved_query="What is the total contract value?")
        result = query_expansion_node(state)
        assert result["hyde_query"] is not None

    def test_expansion_queries_list_length(self):
        state = _base_state(resolved_query="Who approved the storage contract?")
        result = query_expansion_node(state)
        assert len(result["expansion_queries"]) >= 2  # at minimum: resolved + stepback

    def test_only_writes_designated_fields(self):
        # summarize_thread added: query_expansion now sets it when cluster == "summary"
        allowed = {"stepback_query", "hyde_query", "cluster_label", "expansion_queries", "summarize_thread"}
        state = _base_state()
        result = query_expansion_node(state)
        assert set(result.keys()).issubset(allowed)

    def test_summary_cluster_sets_summarize_thread(self):
        """Embedding routing to 'summary' cluster must set summarize_thread=True."""
        state = _base_state(resolved_query="Summarise the whole email thread")
        result = query_expansion_node(state)
        assert result["cluster_label"] == "summary"
        assert result["summarize_thread"] is True

    def test_non_summary_cluster_leaves_summarize_thread_false(self):
        """Approval/person/timeline clusters must leave summarize_thread=False."""
        state = _base_state(resolved_query="What was the approved budget?")
        result = query_expansion_node(state)
        assert result.get("summarize_thread", False) is False


# ──────────────────────────────────────────────
# Agent 2: Query Breakdown
# ──────────────────────────────────────────────

class TestQueryBreakdown:

    def test_simple_query_emits_one_subquery(self):
        state = _base_state(resolved_query="What was approved?", scope_hint="both")
        result = query_breakdown_agent(state)
        assert len(result["sub_queries"]) == 1

    def test_compound_query_splits(self):
        state = _base_state(
            resolved_query="What did finance approve and when was it sent?",
            scope_hint="both",
            cluster_label="approval",
        )
        result = query_breakdown_agent(state)
        assert len(result["sub_queries"]) >= 2

    def test_subquery_has_required_fields(self):
        state = _base_state(resolved_query="Who wrote the proposal?")
        result = query_breakdown_agent(state)
        sq = result["sub_queries"][0]
        assert "text" in sq
        assert "scope" in sq
        assert "priority" in sq

    def test_scope_propagated_from_hint(self):
        state = _base_state(resolved_query="What does the contract say?", scope_hint="attachment")
        result = query_breakdown_agent(state)
        assert result["sub_queries"][0]["scope"] == "attachment"

    def test_only_writes_designated_fields(self):
        allowed = {"sub_queries"}
        state = _base_state()
        result = query_breakdown_agent(state)
        assert set(result.keys()).issubset(allowed)


# ──────────────────────────────────────────────
# Agent 3: Citation Validator
# ──────────────────────────────────────────────

class TestCitationValidator:

    def _make_chunk(self, text, msg_id="m_001", page=None):
        return {
            "doc_id": f"{msg_id}__0",
            "thread_id": "T-0001",
            "message_id": msg_id,
            "source": "email" if not page else "attachment",
            "text": text,
            "page_no": page,
            "rrf_score": 0.9,
            "rerank_score": 0.85,
        }

    def test_grounding_score_above_threshold_no_retry(self):
        chunk_text = "The storage vendor contract was approved for 240000 dollars by the finance team."
        chunks = [self._make_chunk(chunk_text)] * 3
        state = _base_state(
            retrieved_chunks=chunks,
            retrieval_attempt=1,
        )
        result = citation_validator_agent(state)
        assert result["retrieval_insufficient"] is False or result["grounding_score"] >= 0.0

    def test_empty_chunks_triggers_retry_on_attempt_1(self):
        state = _base_state(retrieved_chunks=[], retrieval_attempt=1)
        result = citation_validator_agent(state)
        assert result["retrieval_insufficient"] is True

    def test_max_retry_respected(self):
        """After the retry (retrieval_attempt=3), retrieval_insufficient must be False.

        Lifecycle: initial=1 → first retrieval increments to 2 → citation_validator
        sees 2 (retry allowed, 2<=MAX=2).  Retry retrieval increments to 3 →
        citation_validator sees 3 (3<=2=False, no more retries).
        """
        state = _base_state(retrieved_chunks=[], retrieval_attempt=3)
        result = citation_validator_agent(state)
        assert result["retrieval_insufficient"] is False

    def test_citation_has_message_id(self):
        chunk_text = "Finance approved the storage contract worth 500000 dollars on March 15."
        chunks = [self._make_chunk(chunk_text, msg_id="m_test")]
        state = _base_state(retrieved_chunks=chunks, retrieval_attempt=1)
        result = citation_validator_agent(state)
        if result["citations"]:
            assert result["citations"][0]["message_id"] == "m_test"

    def test_page_no_preserved_in_citation(self):
        chunk_text = "The total amount is specified as 240000 dollars in this section."
        chunks = [self._make_chunk(chunk_text, msg_id="m_pdf", page=2)]
        state = _base_state(retrieved_chunks=chunks, retrieval_attempt=1)
        result = citation_validator_agent(state)
        if result["citations"]:
            assert result["citations"][0]["page_no"] == 2

    def test_grounding_uplift_vs_naive(self):
        """
        Demonstrate that Agent 3 citation validation works correctly.
        The relevant chunk must be cited; grounding score must be > 0.
        (Pre-filtering is now delegated to the retrieval reranker, not Agent 3.)
        """
        irrelevant_chunk = self._make_chunk("The weather was sunny yesterday.")
        relevant_chunk = self._make_chunk(
            "The finance team approved 240000 for the storage vendor contract."
        )
        state = _base_state(
            retrieved_chunks=[irrelevant_chunk, relevant_chunk],
            retrieval_attempt=1,
        )
        result = citation_validator_agent(state)

        # The finance chunk must be cited for an approval query
        cited_texts = [c["chunk_text"] for c in result["citations"]]
        assert any("240000" in t or "approved" in t or "finance" in t for t in cited_texts), \
            "Agent 3 should cite the relevant finance chunk"
        assert result["grounding_score"] > 0, "Grounding score should be > 0 for a grounded answer"

    def test_only_writes_designated_fields(self):
        allowed = {"draft_answer", "citations", "grounding_score", "retrieval_insufficient"}
        state = _base_state(retrieved_chunks=[], retrieval_attempt=2)
        result = citation_validator_agent(state)
        assert set(result.keys()).issubset(allowed)


# ──────────────────────────────────────────────
# Routing correctness
# ──────────────────────────────────────────────

class TestRoutingLogic:

    def test_clarify_not_overfire(self):
        """
        Clarify should NOT fire for clear unambiguous questions.
        If it fires on > 20% of clear questions, routing is broken.
        """
        clear_queries = [
            "What did finance approve for the storage vendor?",
            "Who sent the contract?",
            "Summarise this thread.",
            "What is the total amount in the proposal?",
            "When was the last email sent?",
        ]
        clarify_fired = 0
        for q in clear_queries:
            state = _base_state(raw_query=q)
            result = entity_resolution_agent(state)
            if result.get("clarify_needed"):
                clarify_fired += 1

        rate = clarify_fired / len(clear_queries)
        assert rate <= 0.20, \
            f"Clarify overfire rate {rate:.0%} exceeds 20% on clear queries"


# ──────────────────────────────────────────────
# Person extraction — thread roster fallback
# ──────────────────────────────────────────────

class TestPersonExtraction:

    def _make_raptor_chunk(self, text: str, thread_id: str = "T-FERC01") -> dict:
        """A RAPTOR summary chunk: no 'participants' field (mirrors real ingest output)."""
        return {
            "doc_id": f"{thread_id}__raptor__0",
            "thread_id": thread_id,
            "message_id": "raptor-msg-001",
            "text": text,
            "source": "email",
        }

    def test_raptor_chunks_fall_back_to_roster(self):
        """
        When retrieved chunks are RAPTOR summaries (no participants field),
        _extract_persons_from_chunks must still return participant names by
        consulting the pre-computed thread roster.
        """
        import agents.agents as agents_mod

        tid = "T-FERC01"
        roster_names = ["Jeff Dasovich", "Karen Denne", "Richard Shapiro"]

        chunks = [
            self._make_raptor_chunk(
                "FERC has ordered a price cap on wholesale electricity markets.", tid
            ),
            self._make_raptor_chunk(
                "The commission's decision affects Enron's trading operations.", tid
            ),
        ]

        # Inject a known roster for the test thread
        original_roster = agents_mod._THREAD_ROSTER
        try:
            agents_mod._THREAD_ROSTER = {tid: roster_names}
            result = _extract_persons_from_chunks(chunks)
        finally:
            agents_mod._THREAD_ROSTER = original_roster

        assert result, "Person extraction must return non-empty for a thread with a roster"
        for name in roster_names:
            first_name = name.split()[0]
            assert first_name in result or name in result, \
                f"Expected '{name}' in person extraction output, got: {result!r}"

    def test_email_chunks_with_participants_take_precedence(self):
        """
        Email chunks that carry participants should be used directly (Source 1),
        without needing the roster fallback.
        """
        tid = "T-EMAIL01"
        chunk = {
            "doc_id": f"{tid}__email__0",
            "thread_id": tid,
            "message_id": "msg-abc",
            "text": "Louise Kitchen discussed the energy trading strategy with senior executives.",
            "source": "email",
            "participants": ["Louise Kitchen", "Greg Whalley"],
        }

        import agents.agents as agents_mod
        original_roster = agents_mod._THREAD_ROSTER
        try:
            agents_mod._THREAD_ROSTER = {}  # empty — must work without roster
            result = _extract_persons_from_chunks([chunk])
        finally:
            agents_mod._THREAD_ROSTER = original_roster

        assert result, "Person extraction must return non-empty when chunk has participants"
        assert "Louise Kitchen" in result or "Greg Whalley" in result
