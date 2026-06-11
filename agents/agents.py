"""
agents/agents.py — All agent implementations.

Each function signature: (state: PipelineState) -> dict
Return value is a PARTIAL state update — LangGraph merges it.

State field ownership (each agent ONLY writes its designated fields):
  entity_resolution  → entity_register, resolved_query, clarify_needed, clarify_text
  metadata_prefilter → active_thread_ids, date_filter, scope_hint
  query_expansion    → stepback_query, hyde_query, cluster_label, expansion_queries
  query_breakdown    → sub_queries
  retrieval          → retrieved_chunks, retrieval_attempt
  citation_validator → draft_answer, citations, grounding_score, retrieval_insufficient
  timeline_synthesis → timeline_events, is_temporal_query (updates draft_answer)
  respond            → final_answer, agent_trace
"""

import json
import os
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# Disable TensorFlow in transformers to avoid Keras 3 conflict
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Suppress duplicate OpenMP DLL warning on Windows/Anaconda
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer

from utils.models import PipelineState, SubQuery, Citation
from utils.tracer import AgentTimer, log_agent

# ── Routing constants (imported by graph.py) ──
ROUTE_CLARIFY   = "clarify"
ROUTE_PREFILTER = "metadata_prefilter"
ROUTE_RESPOND   = "respond"
ROUTE_RETRIEVAL = "retrieval"
ROUTE_TIMELINE  = "timeline_synthesis"


# ════════════════════════════════════════════════════════════════════
# Shared model singletons
# ════════════════════════════════════════════════════════════════════

_nlp: spacy.Language | None = None
_embed_model: SentenceTransformer | None = None
_proto_embeddings: dict | None = None
_THREAD_ROSTER: dict[str, list[str]] | None = None


def _get_thread_roster() -> dict[str, list[str]]:
    """Lazy-load the pre-computed thread participant roster from disk."""
    global _THREAD_ROSTER
    if _THREAD_ROSTER is None:
        roster_path = Path("data/index/thread_roster.json")
        if roster_path.exists():
            with open(roster_path) as f:
                _THREAD_ROSTER = json.load(f)
        else:
            _THREAD_ROSTER = {}
    return _THREAD_ROSTER


def _get_nlp() -> spacy.Language:
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _embed_model


# ════════════════════════════════════════════════════════════════════
# Agent 1 — Entity Resolution
# ════════════════════════════════════════════════════════════════════

_REF_PATTERNS = re.compile(
    # Bare pronouns — always ambiguous
    r"\bit\b|\bits\b|"
    # Demonstratives that point back without a following content noun
    r"\b(that|this|those)\s+(?:email|file|attachment|document|pdf|thing|one)\b|"
    # Possessives used without a following noun
    r"\b(?:her|his|their)\s+(?:email|message|reply|response|attachment)\b",
    re.IGNORECASE,
)

_TEMPORAL_SIGNALS = re.compile(
    r"\b(when|before|after|last week|earlier|previously|yesterday)\b",
    re.IGNORECASE,
)

# Email header lines that appear verbatim in Enron body text (forwarded-message format).
# This is a format filter (RFC 2822 headers), not a semantic heuristic — appropriate to keep.
_HEADER_LINE_RE = re.compile(
    r"^(?:from|to|cc|bcc|subject|date|sent\s+by|message-id|received|"
    r"content-type|mime-version|x-\w)\s*:",
    re.IGNORECASE,
)


def entity_resolution_agent(state: PipelineState) -> dict:
    """
    Agent 1 — Entity Resolution.

    1. NER on raw_query to extract new entities.
    2. Update entity_register with any new entities found.
    3. Resolve pronoun/elliptical references using register (most-recent wins).
    4. Detect if query is ambiguous beyond what register can resolve.
    5. Route: clarify_needed=True → clarify, else → metadata_prefilter.
    """
    with AgentTimer() as t:
        raw = state["raw_query"]
        register = dict(state.get("entity_register", {}))
        turn = len(state.get("turn_history", []))

        # ── 1. NER extraction ──
        nlp = _get_nlp()
        doc = nlp(raw)
        for ent in doc.ents:
            key = ent.text.lower()
            if key not in register:
                register[key] = {
                    "type": ent.label_,
                    "text": ent.text,
                    "turn": turn,
                    "message_id": None,
                }

        # ── 2. Reference resolution ──
        resolved = raw
        refs_found = _REF_PATTERNS.findall(raw)

        for ref in refs_found:
            ref_lower = ref.lower()
            if ref_lower in {"it", "its", "that", "this", "the same"}:
                candidate = _most_recent_entity(register, turn,
                                                preferred_types={"ORG", "PRODUCT", "WORK_OF_ART"})
            elif ref_lower in {"her", "his", "their"}:
                candidate = _most_recent_entity(register, turn,
                                                preferred_types={"PERSON"})
            elif any(w in ref_lower for w in ("document", "pdf", "attachment", "file")):
                candidate = _most_recent_entity(register, turn,
                                                preferred_types={"WORK_OF_ART", "PRODUCT", "ORG"})
            elif any(w in ref_lower for w in ("amount", "approval")):
                candidate = _most_recent_entity(register, turn,
                                                preferred_types={"MONEY", "CARDINAL"})
            elif "date" in ref_lower:
                candidate = _most_recent_entity(register, turn,
                                                preferred_types={"DATE", "TIME"})
            else:
                candidate = _most_recent_entity(register, turn, preferred_types=None)

            if candidate:
                resolved = re.sub(
                    r"\b" + re.escape(ref) + r"\b",
                    candidate["text"],
                    resolved,
                    count=1,
                    flags=re.IGNORECASE,
                )

        # ── 3. Ambiguity check ──
        # Clarify only if unresolved references remain AND register is empty AND no temporal context
        still_refs = _REF_PATTERNS.findall(resolved)
        has_temporal = bool(_TEMPORAL_SIGNALS.search(raw))
        register_empty = len(register) == 0 and turn == 0

        # "this email thread", "this thread", "this conversation" always refer to
        # the active thread (thread_id is always set) — never genuinely ambiguous.
        _THREAD_SELF_RE = re.compile(
            r"\bthis\s+(?:email\s+)?(?:thread|conversation|discussion|exchange)\b",
            re.IGNORECASE,
        )
        if _THREAD_SELF_RE.search(raw):
            still_refs = [r for r in still_refs if r.lower() not in {"this", "that", "those"}]

        clarify_needed = bool(still_refs) and register_empty and not has_temporal

        clarify_text = None
        if clarify_needed:
            unresolved = still_refs[0] if still_refs else "that"
            clarify_text = (
                f"Could you clarify what \"{unresolved}\" refers to? "
                f"For example, which email or attachment are you asking about?"
            )

        routing = ROUTE_CLARIFY if clarify_needed else ROUTE_PREFILTER

    log_agent(
        session_id=state["session_id"],
        agent="entity_resolution",
        turn=turn,
        inputs={"raw_query": raw, "register_size": len(register)},
        outputs={
            "resolved_query": resolved,
            "clarify_needed": clarify_needed,
            "new_entities": len(doc.ents),
        },
        routing=routing,
        latency_ms=t.latency_ms,
        token_count=len(raw.split()) + len(resolved.split()),
    )

    return {
        "entity_register": register,
        "resolved_query":  resolved,
        "clarify_needed":  clarify_needed,
        "clarify_text":    clarify_text,
    }


def _most_recent_entity(
    register: dict,
    current_turn: int,
    preferred_types: set | None,
) -> dict | None:
    """Return the most recently mentioned entity of a preferred type."""
    candidates = list(register.values())
    if preferred_types:
        typed = [e for e in candidates if e["type"] in preferred_types]
        if typed:
            candidates = typed
    if not candidates:
        return None
    return max(candidates, key=lambda e: e.get("turn", 0))


# ════════════════════════════════════════════════════════════════════
# Metadata Prefilter (rule-based, no LLM)
# ════════════════════════════════════════════════════════════════════

_DATE_PATTERNS = [
    (re.compile(r"after\s+([\w\s,]+?\d{4})", re.I),  "after"),
    (re.compile(r"before\s+([\w\s,]+?\d{4})", re.I), "before"),
    (re.compile(
        r"in\s+(january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\s+(\d{4})", re.I), "month"),
]

_SCOPE_SIGNALS = {
    "attachment": ["pdf", "attachment", "document", "file", "page",
                   "proposal", "contract", "report"],
    "email":      ["wrote", "replied", "sent", "said", "email", "message", "thread"],
}


def metadata_prefilter_node(state: PipelineState) -> dict:
    """
    Lightweight rule-based node. No LLM call.
    Determines active_thread_ids, date_filter, and scope_hint.
    Summary/person intent is detected downstream in query_expansion_node
    via embedding-based cluster routing (no regex patterns here).
    """
    with AgentTimer() as t:
        resolved = state["resolved_query"]
        query    = resolved.lower()
        thread_id = state["thread_id"]

        active_thread_ids = [] if state.get("search_outside_thread", False) else [thread_id]

        date_filter = None
        for pattern, kind in _DATE_PATTERNS:
            m = pattern.search(query)
            if m:
                date_filter = {"kind": kind, "raw": m.group(0)}
                break

        att_signals   = sum(1 for w in _SCOPE_SIGNALS["attachment"] if w in query)
        email_signals = sum(1 for w in _SCOPE_SIGNALS["email"] if w in query)
        if att_signals > email_signals:
            scope_hint = "attachment"
        elif email_signals > att_signals:
            scope_hint = "email"
        else:
            scope_hint = "both"

    log_agent(
        session_id=state["session_id"],
        agent="metadata_prefilter",
        turn=len(state.get("turn_history", [])),
        inputs={"resolved_query": resolved},
        outputs={"active_thread_ids": active_thread_ids, "scope_hint": scope_hint,
                 "date_filter": date_filter},
        routing="query_expansion",
        latency_ms=t.latency_ms,
        token_count=0,
    )

    return {
        "active_thread_ids": active_thread_ids,
        "date_filter":       date_filter,
        "scope_hint":        scope_hint,
    }


# ════════════════════════════════════════════════════════════════════
# Query Expansion (step-back + HyDE + cluster routing)
# ════════════════════════════════════════════════════════════════════

_PROTOTYPES = {
    "approval": [
        "what was approved and for how much",
        "what did finance sign off on",
        "what is the approved budget",
        "was the contract approved",
    ],
    "timeline": [
        "when was this sent",
        "what happened first in this thread",
        "what is the sequence of events",
        "when did they reply",
    ],
    "person": [
        "who sent this email",
        "who is responsible for this",
        "who wrote to whom",
        "which person mentioned this",
        "who are the key people involved",
        "list all participants in this thread",
    ],
    "attachment": [
        "what does the attached document say",
        "what is on page 2 of the pdf",
        "what are the terms in the contract",
        "summarise the attached proposal",
    ],
    "status": [
        "what is the current status of this",
        "has this been resolved",
        "what is the latest update",
        "is this still pending",
    ],
    # Thread-summary intent: queries asking for comprehensive coverage of the whole thread.
    # Detected via CLS embedding cosine similarity — no regex patterns needed.
    "summary": [
        "summarise the whole email thread",
        "give me an overview of this conversation",
        "what happened in this thread",
        "recap the main points of this discussion",
        "describe the entire thread",
        "what is this thread about",
        "what were the key points discussed",
    ],
}


def _get_proto_embeddings() -> dict:
    global _proto_embeddings
    if _proto_embeddings is None:
        model = _get_embed_model()
        _proto_embeddings = {}
        for cluster, texts in _PROTOTYPES.items():
            embs = model.encode(texts, normalize_embeddings=True)
            _proto_embeddings[cluster] = embs.mean(axis=0)
    return _proto_embeddings


def _route_to_cluster(query: str) -> str:
    """
    Intent classification via SetFit (6 classes: approval, timeline, person,
    attachment, status, summary).

    SetFit (Tunstall et al., 2022 — arxiv:2209.11055) is retrained with a
    "summary" class so the old cosine-threshold pre-check is no longer needed.
    The centroid fallback is retained for resilience when the model file is absent.
    """
    try:
        from utils.intent_classifier import predict_label
        return predict_label(query)
    except Exception:
        model = _get_embed_model()
        q_emb = model.encode(query, normalize_embeddings=True)
        centroids = _get_proto_embeddings()
        scores = {c: float(np.dot(q_emb, emb)) for c, emb in centroids.items()}
        return max(scores, key=scores.get)


def _stepback_rule(query: str) -> str:
    """Rule-based fallback when LLM is unavailable."""
    q = query.lower()
    if any(w in q for w in ["approved", "approval", "budget", "amount", "sign off"]):
        return "what financial decisions or approvals were communicated in this thread"
    if any(w in q for w in ["when", "date", "sent", "timeline", "first", "last"]):
        return "what is the chronological sequence of communications in this thread"
    if any(w in q for w in ["who", "person", "from", "sender"]):
        return "who are the parties involved and what did each communicate"
    if any(w in q for w in ["pdf", "attachment", "document", "page", "contract", "proposal"]):
        return "what is the content and status of documents shared in this thread"
    if any(w in q for w in ["status", "update", "resolved", "pending", "latest"]):
        return "what is the current state of the matter discussed in this thread"
    return f"what information in this email thread is relevant to: {query}"


def _stepback(query: str) -> str:
    """LLM-based step-back abstraction (flan-t5-small) with rule-based fallback."""
    try:
        from utils.llm import stepback_query
        return stepback_query(query)
    except Exception:
        return _stepback_rule(query)


def _hyde_template(query: str) -> str:
    """Template fallback when LLM is unavailable."""
    return (
        f"Subject: Re: {query[:60]}\n\n"
        f"Hi team,\n\n"
        f"Following up on your question about {query.rstrip('?')}. "
        f"Based on our records, the relevant details are as follows. "
        f"Please see the attached document for full specifics. "
        f"Let me know if you need any further clarification.\n\n"
        f"Best regards"
    )


def _hyde(query: str) -> str:
    """LLM-based HyDE document generation (flan-t5-small) with template fallback."""
    try:
        from utils.llm import hyde_document
        return hyde_document(query)
    except Exception:
        return _hyde_template(query)


def query_expansion_node(state: PipelineState) -> dict:
    """
    Step-back + HyDE + cluster routing — all three feed into retrieval.

    Cluster routing uses CLS embedding cosine similarity against prototype
    centroids (no regex).  When the cluster is "summary", summarize_thread
    is set True so that retrieval uses MMR + synopsis query and the citation
    validator builds a full thread summary instead of a single focused answer.

    LLM calls are skipped for clusters where they don't improve retrieval:
      summary / person  — synopsis vector + direct query already covers them;
                          LLM expansion adds noise and costs 4-6 s on CPU.
      timeline          — step-back is useful but HyDE (fake email) is not.
      approval / attachment / status — benefit from both step-back and HyDE.
    """
    with AgentTimer() as t:
        resolved = state["resolved_query"]

        # Cluster first — determines which LLM calls are worth making.
        cluster    = _route_to_cluster(resolved)
        is_summary = (cluster == "summary")

        use_llm_stepback = cluster in {"approval", "attachment", "status", "timeline"}
        use_llm_hyde     = cluster in {"approval", "attachment", "status"}

        stepback = _stepback(resolved) if use_llm_stepback else _stepback_rule(resolved)
        hyde     = _hyde(resolved)     if use_llm_hyde     else _hyde_template(resolved)
        expansion_queries = [resolved, stepback, hyde]

    log_agent(
        session_id=state["session_id"],
        agent="query_expansion",
        turn=len(state.get("turn_history", [])),
        inputs={"resolved_query": resolved},
        outputs={"stepback_query": stepback, "cluster_label": cluster,
                 "summarize_thread": is_summary, "hyde_preview": hyde[:80]},
        routing="query_breakdown",
        latency_ms=t.latency_ms,
        token_count=len(resolved.split()) * 3,
    )

    return {
        "stepback_query":    stepback,
        "hyde_query":        hyde,
        "cluster_label":     cluster,
        "expansion_queries": expansion_queries,
        "summarize_thread":  is_summary,   # True when cluster routing says "summary"
    }


# ════════════════════════════════════════════════════════════════════
# Agent 2 — Query Breakdown
# ════════════════════════════════════════════════════════════════════

_COMPOUND_SIGNALS = re.compile(
    r"\b(and|also|as well as|in addition|furthermore|plus|both)\b",
    re.IGNORECASE,
)
_QUESTION_SPLITS = re.compile(r"\?\s+(?=[A-Z])")


def query_breakdown_agent(state: PipelineState) -> dict:
    """
    Agent 2 — Query Breakdown.
    Decomposes resolved query into 1-3 focused sub-queries.
    Simple queries emit exactly one sub-query — no over-splitting.
    """
    with AgentTimer() as t:
        resolved   = state["resolved_query"]
        scope_hint = state.get("scope_hint", "both")
        cluster    = state.get("cluster_label", "status")
        sub_queries = _decompose(resolved, scope_hint, cluster)

    log_agent(
        session_id=state["session_id"],
        agent="query_breakdown",
        turn=len(state.get("turn_history", [])),
        inputs={"resolved_query": resolved, "scope_hint": scope_hint},
        outputs={"sub_queries": [sq.model_dump() for sq in sub_queries]},
        routing="retrieval",
        latency_ms=t.latency_ms,
        token_count=len(resolved.split()),
    )

    return {"sub_queries": [sq.model_dump() for sq in sub_queries]}


def _decompose(query: str, scope_hint: str, cluster: str) -> list[SubQuery]:
    # Multiple explicit questions separated by '? '
    parts = _QUESTION_SPLITS.split(query)
    if len(parts) >= 2:
        sqs = []
        for i, part in enumerate(parts[:3]):
            part = part.strip().rstrip("?")
            if not part:
                continue
            sqs.append(SubQuery(
                text=part,
                scope=_infer_scope(part, scope_hint),
                priority=i + 1,
            ))
        return sqs

    # Compound within a single question (approval or attachment cluster)
    if _COMPOUND_SIGNALS.search(query) and cluster in {"approval", "attachment"}:
        return [
            SubQuery(text=query, scope=scope_hint, priority=1),
            SubQuery(
                text=f"when and how was this communicated or documented: {query}",
                scope="email",
                priority=2,
            ),
        ]

    return [SubQuery(text=query, scope=scope_hint, priority=1)]


def _infer_scope(text: str, hint: str) -> str:
    att_words   = {"pdf", "page", "attachment", "document", "contract", "proposal", "file"}
    email_words = {"sent", "wrote", "replied", "said", "email", "message"}
    t = text.lower()
    if any(w in t for w in att_words):
        return "attachment"
    if any(w in t for w in email_words):
        return "email"
    return hint


# ════════════════════════════════════════════════════════════════════
# Agent 3 — Citation Validator
# ════════════════════════════════════════════════════════════════════

GROUNDING_THRESHOLD    = 0.80
MAX_RETRIEVAL_ATTEMPTS = 2


def _token_f1(claim: str, chunk_text: str) -> float:
    """Token-level F1 overlap — used to find the best matching chunk."""
    def tokens(s: str) -> Counter:
        return Counter(re.sub(r"[^\w\s]", "", s.lower()).split())

    c_tok  = tokens(claim)
    ch_tok = tokens(chunk_text)
    if not c_tok or not ch_tok:
        return 0.0

    overlap   = sum((c_tok & ch_tok).values())
    precision = overlap / sum(c_tok.values())
    recall    = overlap / sum(ch_tok.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _claim_precision(claim: str, chunk_text: str) -> float:
    """Fraction of claim tokens that appear in chunk_text.

    F1 is the wrong metric for extractive grounding: a 10-word claim from a
    300-word chunk gets precision=1.0 but recall=0.03, so F1≈0.06 even though
    the claim is perfectly supported. Precision correctly answers 'is this
    claim present in the source?'

    Citation tags [msg: <id>] are stripped before calculation — the message-ID
    tokens (numeric IDs, 'JavaMail', 'evans', 'thyme') do not appear in chunk
    bodies and would otherwise inflate the denominator and drop precision to 0%.
    """
    def tokens(s: str) -> Counter:
        return Counter(re.sub(r"[^\w\s]", "", s.lower()).split())

    claim_clean = re.sub(r"\[msg:[^\]]*\]", "", claim).strip()
    c_tok  = tokens(claim_clean)
    ch_tok = tokens(chunk_text)
    if not c_tok:
        return 0.0
    overlap = sum((c_tok & ch_tok).values())
    return overlap / sum(c_tok.values())


_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "i", "you", "he", "she",
    "it", "we", "they", "what", "who", "which", "that", "this", "these",
    "those", "with", "of", "for", "in", "on", "at", "to", "from", "and",
    "or", "but", "not", "no", "any", "all", "also", "than", "then", "by",
    "as", "if", "so", "up", "out", "into", "about", "per", "its",
})


def _content_tokens(text: str) -> set:
    return {t.lower().strip(".,!?;:") for t in text.split()
            if t.lower().strip(".,!?;:") not in _STOPWORDS and len(t) > 2}


def _split_into_claims(answer: str) -> list[str]:
    """Split a draft answer into individual factual sentences, filtering filler."""
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    filler = {"sure", "certainly", "of course", "based on", "according to", "please note"}
    return [
        s for s in sentences
        if len(s.split()) > 4 and not any(f in s.lower() for f in filler)
    ]


_NAME_PILE_RE = re.compile(
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"     # Name token (e.g. "Jeff" or "Jeff Skilling")
    r"(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?){2,}",  # 2+ more comma-separated name tokens
)


def _is_coherent_chunk(text: str) -> bool:
    """Reject garbled text produced by distilbart RAPTOR summarization.

    Three signals for garbage:
    1. Average sentence length < 4 words — fragmented, not real sentences.
    2. A bigram repeats 3+ times — distilbart repetition loop.
    3. Name pile: 3+ comma-separated capitalised names in sequence — distilbart
       hallucinates entity lists when the source contains many proper nouns.
    """
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    if sentences:
        avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_words < 4:
            return False
    words = text.lower().split()
    if len(words) >= 4:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        if bigrams and max(Counter(bigrams).values()) >= 3:
            return False
    # Distilbart name-pile hallucination: "Jensen, Johnson, McGregory, vanished..."
    if _NAME_PILE_RE.search(text):
        return False
    return True


_GREETING_RE = re.compile(
    r"^(hi|hello|dear|hey|thanks|thank you|regards|sincerely|best|cheers|"
    r"all,|team,|everyone,|all —|jeff,|ken,|skilling|lay,)",
    re.IGNORECASE,
)

# Meta-openers: sentences that introduce content without carrying content themselves.
_META_OPENER_RE = re.compile(
    r"^(here are|here is|please see|see below|see attached|as follows|"
    r"the following|below is|below are|i've included|i have included|"
    r"following up|please find|i'm forwarding|i am forwarding|"
    r"fyi|for your information|per your request|as requested|"
    r"just a quick|quick note|wanted to|i wanted)\b",
    re.IGNORECASE,
)


def _is_list_like(text: str) -> bool:
    """
    Return True when text is a bullet/numbered list rather than a prose sentence.

    Signals:
      • 3+ occurrences of ' - ' (dash bullet separator common in Enron emails)
      • 3+ lines with average < 8 words (each line is a list item, not a sentence)
      • Any embedded header line mid-text (forwarded-message block inside body)
    """
    if text.count(" - ") >= 3:
        return True
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) >= 3:
        avg_words = sum(len(l.split()) for l in lines) / len(lines)
        if avg_words < 8:
            return True
    return any(_HEADER_LINE_RE.match(l) for l in lines)


_VERBS_RE = re.compile(
    r"\b(is|was|were|are|will|has|had|have|said|sent|wrote|became|took|"
    r"announced|approved|outlined|meets|met|joined|left|resigned|stated|"
    r"confirmed|noted|reported|presented|discussed|agreed|recommended|"
    r"hired|appointed|named|promoted|succeeded|replaced|signed|authorized)\b",
    re.IGNORECASE,
)

# Farewell / congratulatory platitudes — dense in threads triggered by promotions/announcements.
# Chunks where >60% of sentences match these patterns carry no factual content for RAG.
_FAREWELL_PHRASES_RE = re.compile(
    r"\b(pleasure (?:working|interacting|to have worked|to have been)|"
    r"paths cross|enjoyed working with|look forward to future|"
    r"look forward to (?:working with you|our future interactions)|"
    r"best (?:wishes|of luck)|well deserved (?:promotion|appointment|recognition)|"
    r"continued? success|hope (?:to work with|to see|our paths)|"
    r"it (?:has been|was) a pleasure|accept my (?:sincere|heartfelt)|"
    r"(?:my |sincere )?congratulations on (?:this|your|the)|"
    r"wishing you|all the best for|warm(?:est)? regards|"
    r"sincerely congratulate|what an achievement|so (?:very )?pleased for|"
    r"i (?:hope|wish) you (?:continued|all)|"
    r"travelling to (?:asia|europe)|old relic|bricks.*clicks)\b",
    re.IGNORECASE,
)


def _farewell_density(text: str) -> float:
    """Fraction of sentences that are farewell/congratulatory platitudes."""
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if len(s.split()) > 3]
    if not sentences:
        return 0.0
    farewell_count = sum(1 for s in sentences if _FAREWELL_PHRASES_RE.search(s))
    return farewell_count / len(sentences)


_FORWARDED_HEADER_LINE_RE = re.compile(
    r"^(?:from|to|cc|bcc|subject|date|sent\s+by|message-id|received|"
    r"content-type|mime-version|x-\w|forwarded\s+by|original\s+message|"
    r"-{3,}|={3,})\s*[:\-]?",
    re.IGNORECASE,
)


def _strip_forwarded_headers(text: str) -> str:
    """Remove leading email forwarding/header blocks from chunk text.

    Enron emails stored in .eml files often embed multiple forwarded message
    headers (Sent by:, To:, Subject:) directly in the chunk text without any
    sentence-ending punctuation.  These lines are not useful for RAG but
    confuse sentence splitting by merging with the first real sentence.

    Strategy: scan lines; once we see a header block followed by a blank line,
    drop everything up to and including that blank line.
    """
    lines = text.splitlines()
    i = 0
    last_header_end = -1
    in_header = False
    for j, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if in_header:
                last_header_end = j
            in_header = False
        elif _FORWARDED_HEADER_LINE_RE.match(stripped) or (stripped in ("cc:", "cc:", "--")):
            in_header = True
        else:
            in_header = False
    if last_header_end >= 0 and last_header_end < len(lines) - 2:
        return "\n".join(lines[last_header_end + 1:])
    return text


def _filter_sentence_candidates(sentences: list[str]) -> list[str]:
    """
    Shared pre-filter for both _best_extractive_sentence and _best_extractive_sentences.

    Hard filters (applied unconditionally, no fallback relaxation):
      • Email header lines (From:, Sent by:, etc.)
      • Bullet / numbered list blocks
      • Greeting openers (Hi, Dear, etc.)
      • Meta-openers ("Here are the ...", "Please see below", etc.)

    Soft filters (relaxed progressively so we always return something):
      Level 1: requires a content verb AND no farewell phrasing
      Level 2: allows lack-of-verb; still rejects farewell phrasing
      Level 3: allows farewell phrasing — last resort
    """
    # Hard filters
    sentences = [s for s in sentences if not _HEADER_LINE_RE.match(s)]
    sentences = [s for s in sentences if not _is_list_like(s)]
    sentences = [s for s in sentences
                 if not _GREETING_RE.match(s) and not _META_OPENER_RE.match(s)]
    if not sentences:
        return []  # nothing survived hard filters
    # Soft filters — progressive fallback
    candidates = [s for s in sentences
                  if _VERBS_RE.search(s) and not _FAREWELL_PHRASES_RE.search(s)]
    if not candidates:
        candidates = [s for s in sentences if not _FAREWELL_PHRASES_RE.search(s)]
    return candidates or sentences


def _split_into_sentences(text: str) -> list[str]:
    """
    Split chunk text into candidate sentences.

    Splits on BOTH sentence-ending punctuation AND paragraph breaks (\n\n).
    Splitting on paragraph breaks is important for Enron email text that uses
    multi-line "salutation\n\ncontent" blocks — without paragraph splitting,
    "Catherine/Roger:\n\nHere are some more details..." is treated as one sentence
    and the meta-opener filter only catches sentences that START with a meta-opener.

    Non-ASCII characters (ψ, ◆, etc.) from .eml encoding issues are stripped here
    so every downstream sentence is display-clean.
    """
    clean = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Strip Lotus Notes attribution fragments ("Name/Org@Domain wrote:") before
    # splitting so forwarded-email attribution lines don't bleed into sentences.
    clean = re.sub(r"\b\w+(?:\.\w+)*/\w+(?:/\w+)*@\w+(?:\s+wrote)?:?", "", clean)
    raw = re.split(r"(?<=[.!?])\s+|\n{2,}", clean)
    return [s.strip() for s in raw if len(s.split()) > 5]


def _best_extractive_sentence(chunk_text: str, query: str = "") -> str:
    """
    Pick the most informative sentence from a chunk.

    With a query: adaptive quality-threshold selection via _select_by_quality.
    Only sentences semantically close to the query are returned — filler sentences
    like "I agree with what Jeff wrote" are excluded when the query matches
    a more specific sentence like "FERC's final order will fix the price cap".

    Without a query: longest candidate wins (original behaviour).
    Returns "" when no sentence survives the hard pre-filters.
    """
    text = _strip_forwarded_headers(chunk_text)
    sentences = _split_into_sentences(text)
    if not sentences:
        return ""
    candidates = _filter_sentence_candidates(sentences)
    if not candidates:
        return ""
    if query:
        selected = _select_by_quality(candidates, query, n=1)
        return selected[0][:300] if selected else ""
    return max(candidates, key=len)[:300]


def _best_extractive_sentences(chunk_text: str, n: int = 2, query: str = "") -> list[str]:
    """
    Pick up to n informative, distinct sentences from a chunk.

    With a query: only sentences within the adaptive quality window of the best
    match are included (see _select_by_quality).  This prevents weak filler from
    being included as the 2nd-best pick when the best sentence is highly specific.

    Without a query: sorted by length (longest first).
    Returns [] when no sentence survives the hard pre-filters.
    """
    text = _strip_forwarded_headers(chunk_text)
    sentences = _split_into_sentences(text)
    if not sentences:
        return []
    candidates = _filter_sentence_candidates(sentences)
    if not candidates:
        return []
    if query:
        return [s[:250] for s in _select_by_quality(candidates, query, n=n)]
    candidates.sort(key=len, reverse=True)
    result: list[str] = []
    seen_tok: list[set] = []
    for s in candidates:
        tok = set(s.lower().split())
        if not any(len(tok & p) / max(len(tok), 1) > 0.70 for p in seen_tok):
            result.append(s[:250])
            seen_tok.append(tok)
        if len(result) >= n:
            break
    return result


def _extract_persons_from_chunks(chunks: list[dict]) -> str:
    """
    For person queries: return named participants with contextual citation sentences.

    Three-source strategy (in priority order):
      1. chunk["participants"] — MIME header names stored at ingest time (fast, authoritative)
      2. thread_roster.json   — pre-computed roster of all unique names per thread;
                                used when retrieved chunks are RAPTOR summaries with no
                                participants field (RAPTOR chunks are AI-generated, not emails)
      3. spaCy NER on body    — for context sentences and names not captured above

    This layered approach means person queries ALWAYS return a participant list,
    even when the retriever happens to rank only RAPTOR summary chunks first.
    """
    if not chunks:
        return ""

    nlp = _get_nlp()
    ref_msg_id = chunks[0]["message_id"]
    thread_id  = chunks[0].get("thread_id", "")

    # persons_dict: name_lower → {name, context, msg_id}
    persons_dict: dict[str, dict] = {}

    # ── Source 1: chunk metadata participants ────────────────────────────────
    for chunk in chunks[:12]:
        for name in chunk.get("participants") or []:
            key = name.lower()
            if key not in persons_dict:
                persons_dict[key] = {"name": name, "context": "", "msg_id": chunk["message_id"]}
        if len(persons_dict) >= 12:
            break

    # ── Source 2: thread roster (pre-computed from all emails at ingest) ─────
    # Always supplement — RAPTOR chunks have no participants so retrieved chunks
    # often yield nothing from Source 1.
    if thread_id:
        roster = _get_thread_roster().get(thread_id, [])
        for name in roster[:15]:
            key = name.lower()
            if key not in persons_dict:
                persons_dict[key] = {"name": name, "context": "", "msg_id": ref_msg_id}

    # ── Source 3: spaCy NER for context sentences ────────────────────────────
    for chunk in chunks[:12]:
        if len([v for v in persons_dict.values() if v["context"]]) >= 6:
            break
        body = _strip_forwarded_headers(chunk["text"])[:1000]
        doc  = nlp(body)
        for ent in doc.ents:
            if ent.label_ != "PERSON" or len(ent.text.split()) < 2:
                continue
            person = ent.text.strip()
            key = person.lower()
            if key in persons_dict and persons_dict[key]["context"]:
                continue
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if (person in sent_text
                        and len(sent_text.split()) > 6
                        and _VERBS_RE.search(sent_text)
                        and not _GREETING_RE.match(sent_text)
                        and not _FAREWELL_PHRASES_RE.search(sent_text)
                        and not _HEADER_LINE_RE.match(sent_text)
                        and not _META_OPENER_RE.match(sent_text)
                        and not _is_list_like(sent_text)):  # reject bullet-list blocks
                    if key not in persons_dict:
                        persons_dict[key] = {"name": person, "context": sent_text,
                                             "msg_id": chunk["message_id"]}
                    else:
                        persons_dict[key]["context"] = sent_text
                        persons_dict[key]["msg_id"]  = chunk["message_id"]
                    break

    # ── Build answer ─────────────────────────────────────────────────────────
    # Lead with the participant list — the direct answer to "who are the people?".
    # Context sentences (prose snippets mentioning a person) are appended after,
    # and only if they pass quality filters (non-list, non-opener prose).
    with_ctx = [v for v in persons_dict.values() if v["context"]]
    no_ctx   = [v for v in persons_dict.values() if not v["context"]]

    parts: list[str] = []
    if no_ctx:
        names_str = ", ".join(v["name"] for v in no_ctx[:8])
        ref_id = no_ctx[0]["msg_id"] if no_ctx else ref_msg_id
        parts.append(f"Participants: {names_str}. [msg: <{ref_id}>]")

    for val in with_ctx[:3]:
        parts.append(f"{val['context']} [msg: <{val['msg_id']}>]")

    return " ".join(parts)


_MAX_DRAFT_CHUNKS = 5  # maximum unique chunks included in any answer

# ── Embedding-based sentence selection ──────────────────────────────────────
# Cache query embeddings within the process so the same query isn't re-encoded
# across multiple chunk loops in a single turn.
_query_emb_cache: dict[str, "np.ndarray"] = {}


def _embed_query(query: str) -> "np.ndarray":
    if len(_query_emb_cache) > 200:  # prevent unbounded growth across a long session
        _query_emb_cache.clear()
    if query not in _query_emb_cache:
        _query_emb_cache[query] = _get_embed_model().encode(
            query, normalize_embeddings=True
        )
    return _query_emb_cache[query]


def _select_by_quality(
    sentences: list[str],
    query: str,
    n: int = 1,
    *,
    quality_gap: float = 0.18,
    abs_floor: float = 0.22,
) -> list[str]:
    """
    Select up to *n* sentences using adaptive semantic quality filtering.

    Algorithm:
    1. Score all candidates by cosine similarity to the query embedding.
    2. Compute a dynamic threshold = max(abs_floor, best_score - quality_gap).
       • abs_floor (0.22) removes sentences that are broadly irrelevant to the query.
       • quality_gap (0.18) means a second sentence must be within 0.18 cosine of
         the best: if the best is 0.70, only sentences ≥ 0.52 are eligible.
         This eliminates weak "I agree with what Jeff wrote" filler that would slip
         in as a 2nd-best pick when the first sentence is genuinely strong.
    3. Always return at least the best sentence (never empty).

    This is principled rather than pattern-based: the model itself decides what is
    "close enough" to the query without requiring hand-crafted regex filters for
    every email style variant.
    """
    if not sentences or not query:
        return sentences[:n]
    try:
        model  = _get_embed_model()
        q_emb  = _embed_query(query)
        s_embs = model.encode(sentences, normalize_embeddings=True)
        scores = [float(np.dot(q_emb, s)) for s in s_embs]

        best_score = max(scores)
        threshold  = max(abs_floor, best_score - quality_gap)

        ranked = sorted(zip(scores, sentences), reverse=True)
        selected = [s for score, s in ranked if score >= threshold]

        if not selected:
            selected = [ranked[0][1]]  # always include the best, even below floor

        # Dedup within the selected set
        result, seen = [], []
        for s in selected:
            tok = set(s.lower().split())
            if not any(len(tok & p) / max(len(tok), 1) > 0.70 for p in seen):
                result.append(s)
                seen.append(tok)
            if len(result) >= n:
                break
        return result

    except Exception:
        return sentences[:n]


def _rank_by_query(sentences: list[str], query: str) -> list[str]:
    """Simple cosine-rank without quality filtering — used when only ordering matters."""
    if not sentences or not query:
        return sentences
    try:
        model  = _get_embed_model()
        q_emb  = _embed_query(query)
        s_embs = model.encode(sentences, normalize_embeddings=True)
        scores = [float(np.dot(q_emb, s)) for s in s_embs]
        return [sentences[i] for i in np.argsort(scores)[::-1]]
    except Exception:
        return sentences


def _try_llm_ground(chunk_text: str, query: str) -> str:
    """
    Generate one grounded sentence from a chunk via flan-t5-base.
    Returns "" on any error so the caller can fall back to extractive selection.

    This is the Ground-Every-Sentence (GES) pattern: the sentence is generated
    FROM a specific chunk, so it is traceable to that source by construction —
    unlike post-hoc citation where a synthesised answer must be matched back to
    a source it may have paraphrased away from.

    Only called for the top 1-2 chunks per turn (not the full retrieved set)
    to keep the added latency under ~500ms on CPU.

    Hard rejections (fall back to extractive):
      • Output is shorter than 5 words — flan-t5 produced nothing useful.
      • Output starts with an email header line (From:, To:, Cc:, ...) — flan-t5
        copied the forwarding header block verbatim instead of answering.
      • Output starts with a greeting (Hi, Dear, Thanks, ...) — same problem.
      • Output is too similar to the query — flan-t5 parroted the question back
        (a common failure mode for very short or abstract queries).
    """
    if not query:
        return ""
    try:
        from utils.llm import ground_sentence
        # Remove forwarded-message header blocks before sending to flan-t5 so it
        # can't echo Lotus Notes address lines ("Name/Org@Domain cc: ...").
        stripped_text = _strip_forwarded_headers(chunk_text)
        result = ground_sentence(stripped_text, query)
        if not result or len(result.split()) < 5:
            return ""
        # Reject header/greeting copies (RFC 2822 header at start)
        if _HEADER_LINE_RE.match(result) or _GREETING_RE.match(result):
            return ""
        # Reject Lotus Notes address copies: "Name/Org@Domain" format
        if re.search(r"\b\w+/\w+@\w+", result):
            return ""
        # Reject query parroting: >65% of query words in first 10 output words
        q_words = set(query.lower().split())
        r_words = set(result.lower().split()[:10])
        if q_words and len(q_words & r_words) / len(q_words) > 0.65:
            return ""
        return result[:300]
    except Exception:
        pass
    return ""


def _build_thread_summary(chunks: list[dict], query: str = "") -> str:
    """
    Build a comprehensive multi-paragraph thread summary from ALL retrieved chunks.

    Called when cluster_label == "summary". We receive MMR-selected diverse chunks
    so we can produce a representative summary even when most emails are congratulatory.

    Strategy:
    1. Filter garbled RAPTOR chunks (distilbart repetition loops).
    2. Separate substantive chunks (low farewell density) from farewell-heavy ones.
    3. Dedup by content token overlap within each group.
    4. Extract 2 sentences from substantive chunks, 1 from farewell chunks.
       Sentences are ranked by semantic similarity to the query when supplied.
    5. Include up to 2 farewell entries so the summary reflects the thread's tone.
    """
    usable = [c for c in chunks if _is_coherent_chunk(c["text"])]
    if not usable:
        usable = chunks[:20]

    substantive = [c for c in usable if _farewell_density(c["text"]) < 0.75]
    farewells   = [c for c in usable if _farewell_density(c["text"]) >= 0.75]

    seen_tok: list[set] = []

    def _dedup_group(group: list[dict], limit: int) -> list[dict]:
        result: list[dict] = []
        for chunk in group:
            tokens = _content_tokens(chunk["text"])
            if not any(len(tokens & p) / max(len(tokens), 1) > 0.70 for p in seen_tok):
                seen_tok.append(tokens)
                result.append(chunk)
            if len(result) >= limit:
                break
        return result

    deduped_sub = _dedup_group(substantive, 7)
    deduped_far = _dedup_group(farewells,   2)
    ordered     = deduped_sub + deduped_far

    if not ordered:
        return "No relevant content found for this thread."

    parts: list[str] = []
    seen_sent_toks: list[set] = []  # per-sentence dedup across all chunks

    for chunk in ordered:
        msg_id = chunk["message_id"]
        page   = chunk.get("page_no")
        fd     = _farewell_density(chunk["text"])
        n_sents = 1 if fd >= 0.75 else 2
        sents  = _best_extractive_sentences(chunk["text"], n=n_sents, query=query)

        # Deduplicate each sentence individually against already-seen content.
        # This catches the case where two chunks share an opening sentence (e.g.,
        # "I agree with what Jeff wrote.") — chunk-level dedup misses it because
        # the second chunk also contains unique content, making its overall token
        # overlap with the first chunk fall below the threshold.
        fresh: list[str] = []
        for s in sents:
            tok = _content_tokens(s)
            if tok and any(len(tok & p) / max(len(tok), 1) > 0.65 for p in seen_sent_toks):
                continue
            seen_sent_toks.append(tok)
            # Strip non-ASCII encoding artifacts (ψ, ◆, etc.) before display.
            # These come from encoding issues in the original .eml files and are
            # not readable content — replace with a single space.
            clean_s = re.sub(r"[^\x00-\x7F]+", " ", s).strip()
            if clean_s:
                fresh.append(clean_s)

        if not fresh:
            continue
        text = " ".join(fresh)
        cite = f"[msg: <{msg_id}>, page: {page}]" if page else f"[msg: <{msg_id}>]"
        parts.append(f"{text} {cite}")

    return " ".join(parts) if parts else "No relevant content found for this thread."

def _dedup_chunks(
    chunks: list[dict],
    max_chunks: int = _MAX_DRAFT_CHUNKS,
    filter_farewells: bool = True,
    require_coherent: bool = True,
) -> list[dict]:
    """Return up to max_chunks unique chunks (>70% token overlap = duplicate).

    Coherence check (_is_coherent_chunk) is applied ONLY to RAPTOR chunks —
    it was designed to reject garbled distilbart output, not email body text.
    Email chunks may have bigram-repetitive forwarding headers or short lines,
    which would falsely fail the coherence check.

    Farewell-dense chunks (>60% platitude sentences) are skipped unless
    filter_farewells=False (used as a last-resort fallback).

    require_coherent=False skips the RAPTOR coherence check too — last resort
    when even incoherent RAPTOR chunks are all we have.
    """
    seen_token_sets: list[set] = []
    deduped: list[dict] = []
    for chunk in chunks:
        is_raptor = "__raptor__" in chunk.get("doc_id", "")
        if require_coherent and is_raptor and not _is_coherent_chunk(chunk["text"]):
            continue
        if filter_farewells and _farewell_density(chunk["text"]) > 0.6:
            continue
        tokens = _content_tokens(chunk["text"])
        is_dup = any(len(tokens & p) / max(len(tokens), 1) > 0.70 for p in seen_token_sets)
        if not is_dup:
            seen_token_sets.append(tokens)
            deduped.append(chunk)
        if len(deduped) >= max_chunks:
            break
    return deduped


def _build_draft_answer_extractive(
    chunks: list[dict],
    sub_queries: list[dict],
    sentences_per_chunk: int = 1,
    max_chunks: int = _MAX_DRAFT_CHUNKS,
    query: str = "",
) -> str:
    """Extractive draft: top unique chunks with inline citations.

    When `query` is provided, sentences are selected by semantic similarity to the
    query (embedding cosine) rather than by length.  This avoids pulling in
    bullet-list blocks or meta-opener sentences that happen to be long.

    Three-tier fallback so we never silently return empty:
    1. Normal: coherent RAPTOR + farewell filter
    2. Farewell bypass: still rejects incoherent RAPTOR
    3. Full bypass: accept any chunk (last resort — best_extractive_sentence
       still filters out header lines and greetings within the chunk text)
    """
    deduped = _dedup_chunks(chunks, max_chunks=max_chunks)
    if not deduped:
        deduped = _dedup_chunks(chunks, max_chunks=max_chunks, filter_farewells=False)
    if not deduped:
        deduped = _dedup_chunks(chunks, max_chunks=max_chunks,
                                filter_farewells=False, require_coherent=False)
    parts = []
    seen_sent_toks: list[set] = []  # sentence-level dedup across all parts
    for chunk_idx, chunk in enumerate(deduped):
        msg_id = chunk["message_id"]
        page   = chunk.get("page_no")
        if sentences_per_chunk > 1:
            sents = _best_extractive_sentences(chunk["text"], n=sentences_per_chunk, query=query)
            text  = " ".join(sents)
        else:
            # For the top 2 chunks, try LLM-grounded sentence generation first.
            # ground_sentence() produces one sentence derived directly from this
            # chunk, giving more coherent prose while keeping citation traceable.
            # Falls back to extractive if LLM fails or produces a short result.
            llm_text = _try_llm_ground(chunk["text"], query) if chunk_idx < 2 else ""
            text = llm_text or _best_extractive_sentence(chunk["text"], query=query)
        if not text.strip():
            continue
        tok = _content_tokens(text)
        if tok and any(len(tok & p) / max(len(tok), 1) > 0.65 for p in seen_sent_toks):
            continue
        seen_sent_toks.append(tok)
        if page:
            parts.append(f"{text} [msg: <{msg_id}>, page: {page}]")
        else:
            parts.append(f"{text} [msg: <{msg_id}>]")
    return " ".join(parts)


def _build_draft_answer(
    chunks: list[dict],
    sub_queries: list[dict],
    cluster: str = "",
    query: str = "",
) -> str:
    """
    Extractive answer: best unique coherent sentence per chunk, cited inline.
    Dispatches by cluster label (set via CLS embedding cosine similarity, not regex):

      "summary"    → _build_thread_summary: multi-paragraph extractive summary from
                     all retrieved chunks (MMR-selected by retrieval node).
      "person"     → _extract_persons_from_chunks: NER-extracted names with substantive
                     context sentences, one citation per unique person.
      "status"     → email chunks first (specific evidence), RAPTOR as fallback, 2 sentences/chunk for depth.
      everything else → single most informative sentence per chunk, ranked by query
                        embedding similarity when a query is available.
    """
    if cluster == "summary":
        result = _build_thread_summary(chunks, query=query)
        if result:
            return result

    if cluster == "person":
        result = _extract_persons_from_chunks(chunks)
        if result:
            return result

    if cluster == "status":
        email_chunks  = [c for c in chunks if "__raptor__" not in c.get("doc_id", "")]
        raptor_chunks = [c for c in chunks if "__raptor__"     in c.get("doc_id", "")]
        reordered = email_chunks + raptor_chunks
        return _build_draft_answer_extractive(
            reordered, sub_queries, sentences_per_chunk=2, max_chunks=6, query=query
        )

    return _build_draft_answer_extractive(chunks, sub_queries, query=query)


def citation_validator_agent(state: PipelineState) -> dict:
    """
    Agent 3 — Citation Validator.
    Builds draft → splits into claims → validates each claim against retrieved chunks.
    Triggers retry if grounding_score < threshold and attempt < max.
    """
    with AgentTimer() as t:
        chunks      = state.get("retrieved_chunks", [])
        sub_queries = state.get("sub_queries", [])
        attempt     = state.get("retrieval_attempt", 1)

        if not chunks:
            routing = "retrieval" if attempt <= MAX_RETRIEVAL_ATTEMPTS else ROUTE_TIMELINE
            log_agent(
                session_id=state["session_id"],
                agent="citation_validator",
                turn=len(state.get("turn_history", [])),
                inputs={"n_chunks": 0, "n_claims": 0, "attempt": attempt},
                outputs={"grounding_score": 0.0, "retrieval_insufficient": attempt <= MAX_RETRIEVAL_ATTEMPTS},
                routing=routing,
                latency_ms=0,
                token_count=0,
            )
            return {
                "draft_answer":           "No relevant information found in the selected thread.",
                "citations":              [],
                "grounding_score":        0.0,
                "retrieval_insufficient": attempt <= MAX_RETRIEVAL_ATTEMPTS,
            }

        # The retrieval pipeline (BM25 + FAISS + synopsis + RRF + reranker/MMR)
        # already selects the top-k relevant chunks. Trust the retrieval ranking;
        # pass all chunks to the draft builder without an extra keyword filter.
        cluster = state.get("cluster_label", "")
        query   = state.get("resolved_query", "")
        draft   = _build_draft_answer(chunks, sub_queries, cluster, query=query)
        claims  = _split_into_claims(draft)

        citations: list[Citation] = []
        grounded_count = 0

        for claim in claims:
            best_chunk     = None
            best_precision = 0.0
            best_f1        = 0.0
            for chunk in chunks:
                prec = _claim_precision(claim, chunk["text"])
                if prec > best_precision:
                    best_precision = prec
                    best_f1        = _token_f1(claim, chunk["text"])
                    best_chunk     = chunk

            # A claim is grounded if ≥60% of its content tokens appear in the source.
            # Select by precision (does the claim come from this chunk?) not F1
            # (F1 penalises long chunks unfairly: a 10-word claim in a 300-word chunk
            # gets F1≈6% even when precision=100%).
            if best_chunk and best_precision >= 0.60:
                grounded_count += 1
                citations.append(Citation(
                    claim_text=claim,
                    message_id=best_chunk["message_id"],
                    page_no=best_chunk.get("page_no"),
                    chunk_text=best_chunk["text"][:200],
                    token_f1=best_f1,
                    confidence=best_precision,
                ))

        grounding_score = grounded_count / len(claims) if claims else 0.0
        # Never retry for summary/person clusters:
        #   summary — MMR already ensures full temporal coverage.
        #   person  — answer is built from header metadata (roster/NER), not chunk body
        #             text, so token-precision grounding will always be low regardless
        #             of how many chunks are retrieved.
        # Retry logic: attempt starts at 1, retrieval increments it to 2 before we
        # reach here. Allow retry when attempt <= MAX (2<=2=True); on the retry the
        # incremented value is 3, so 3<=2=False stops the loop.
        insufficient    = (
            grounding_score < GROUNDING_THRESHOLD
            and attempt <= MAX_RETRIEVAL_ATTEMPTS
            and cluster not in {"summary", "person"}
        )

        routing = ROUTE_RETRIEVAL if insufficient else ROUTE_TIMELINE

    log_agent(
        session_id=state["session_id"],
        agent="citation_validator",
        turn=len(state.get("turn_history", [])),
        inputs={"n_chunks": len(chunks), "n_claims": len(claims), "attempt": attempt},
        outputs={
            "grounding_score":        round(grounding_score, 3),
            "grounded_claims":        grounded_count,
            "total_claims":           len(claims),
            "retrieval_insufficient": insufficient,
        },
        routing=routing,
        latency_ms=t.latency_ms,
        token_count=len(draft.split()),
    )

    return {
        "draft_answer":           draft,
        "citations":              [c.model_dump() for c in citations],
        "grounding_score":        grounding_score,
        "retrieval_insufficient": insufficient,
    }


# ════════════════════════════════════════════════════════════════════
# Timeline Synthesis — Bonus agent
# ════════════════════════════════════════════════════════════════════

_TEMPORAL_QUERY = re.compile(
    r"\b(when|timeline|chronolog|sequence|first|last|before|after|order)\b",
    re.IGNORECASE,
)

_DATE_IN_TEXT = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2},?\s+\d{4}|"
    r"\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)


def timeline_synthesis_agent(state: PipelineState) -> dict:
    """
    Bonus Agent — Timeline Synthesis.
    Fires only when the resolved query contains temporal signals.
    Extracts (event, date, message_id) tuples from retrieved chunks,
    sorts chronologically, and prepends a timeline to the draft answer.
    """
    with AgentTimer() as t:
        resolved   = state["resolved_query"]
        is_temporal = bool(_TEMPORAL_QUERY.search(resolved))

        if not is_temporal:
            log_agent(
                session_id=state["session_id"],
                agent="timeline_synthesis",
                turn=len(state.get("turn_history", [])),
                inputs={"resolved_query": resolved, "is_temporal": False},
                outputs={"n_events": 0},
                routing=ROUTE_RESPOND,
                latency_ms=t.latency_ms,
                token_count=0,
            )
            return {"is_temporal_query": False, "timeline_events": None}

        chunks = state.get("retrieved_chunks", [])
        events = []
        seen_event_keys: set = set()
        seen_event_tokens: list[set] = []

        for chunk in chunks:
            if _farewell_density(chunk["text"]) > 0.6:
                continue  # dates mentioned in farewell emails aren't informative events
            dates = _DATE_IN_TEXT.findall(chunk["text"])
            for date_str in dates:
                key = (date_str, chunk["message_id"])
                if key in seen_event_keys:
                    continue
                event_text = _best_extractive_sentence(chunk["text"])
                if not event_text.strip():
                    seen_event_keys.add(key)
                    continue  # no readable sentence — skip this date/chunk pair
                # Skip near-duplicate event text (same content, different msg IDs)
                ev_tokens = _content_tokens(event_text)
                if any(len(ev_tokens & prev) / max(len(ev_tokens), 1) > 0.70
                       for prev in seen_event_tokens):
                    seen_event_keys.add(key)  # mark as seen to avoid re-checking
                    continue
                seen_event_keys.add(key)
                seen_event_tokens.append(ev_tokens)
                events.append({
                    "event":      event_text,
                    "date_raw":   date_str,
                    "message_id": chunk["message_id"],
                    "page_no":    chunk.get("page_no"),
                })

        def _parse_date(e: dict):
            try:
                from dateutil import parser as dp
                return dp.parse(e["date_raw"])
            except Exception:
                return datetime.min

        events.sort(key=_parse_date)

        if events:
            lines = ["**Timeline:**"]
            for ev in events[:6]:
                cite = f"[msg: <{ev['message_id']}>]"
                if ev.get("page_no"):
                    cite = f"[msg: <{ev['message_id']}>, page: {ev['page_no']}]"
                lines.append(f"- {ev['date_raw']}: {ev['event'][:120]} {cite}")
            timeline_str = "\n".join(lines) + "\n\n"
        else:
            timeline_str = ""

        updated_answer = timeline_str + state.get("draft_answer", "")

    log_agent(
        session_id=state["session_id"],
        agent="timeline_synthesis",
        turn=len(state.get("turn_history", [])),
        inputs={"resolved_query": resolved, "is_temporal": is_temporal},
        outputs={"n_events": len(events)},
        routing=ROUTE_RESPOND,
        latency_ms=t.latency_ms,
        token_count=0,
    )

    return {
        "is_temporal_query": is_temporal,
        "timeline_events":   events[:6],
        "draft_answer":      updated_answer,
    }


# ════════════════════════════════════════════════════════════════════
# Clarify and Respond nodes
# ════════════════════════════════════════════════════════════════════

def clarify_node(state: PipelineState) -> dict:
    return {"final_answer": state.get("clarify_text") or "Could you clarify your question?"}


def respond_node(state: PipelineState) -> dict:
    """
    Assemble the final answer with inline citations.
    Also builds agent_trace list from the tracer for the current turn.
    """
    draft     = state.get("draft_answer", "") or ""
    citations = state.get("citations", [])
    # Guarantee a non-empty final answer — empty string causes "no output" in the UI
    if not draft.strip():
        draft = (
            "I could not find relevant information for that question in the selected thread. "
            "Try rephrasing, or enable 'Search outside thread' to look across all threads."
        )

    final = draft

    if citations:
        legend = ["\n\n**Sources:**"]
        seen: set = set()
        for c in citations:
            key = (c["message_id"], c.get("page_no"))
            if key in seen:
                continue
            seen.add(key)
            if c.get("page_no"):
                legend.append(
                    f"- [msg: <{c['message_id']}>, page: {c['page_no']}]  "
                    f"(confidence: {c['confidence']:.0%})"
                )
            else:
                legend.append(
                    f"- [msg: <{c['message_id']}>]  "
                    f"(confidence: {c['confidence']:.0%})"
                )
        final += "\n".join(legend)

    # Build agent_trace from tracer for this turn
    try:
        from utils.tracer import read_trace
        turn = len(state.get("turn_history", []))
        records = read_trace(session_id=state["session_id"])
        turn_records = [r for r in records if r.get("turn") == turn]
        agent_trace = [
            {
                "agent":      r["agent"],
                "routing":    r.get("routing", ""),
                "latency_ms": r.get("latency_ms", 0),
            }
            for r in turn_records
        ]
    except Exception:
        agent_trace = []

    return {"final_answer": final, "agent_trace": agent_trace}
