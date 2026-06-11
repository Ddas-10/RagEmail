"""
utils/llm.py — Lightweight LLM wrapper using flan-t5-base.

Provides three functions called at query time:
  stepback_query(query)         → broader abstract rewrite
  hyde_document(query)          → hypothetical answering email
  synthesize_answer(chunks, q)  → extractive→abstractive answer

Model: google/flan-t5-base (250 MB, CPU-feasible, ~3-5s per call)
Loaded once as a module-level singleton on first use.

To further upgrade: swap MODEL_NAME for "google/flan-t5-large" (800 MB)
with no other code changes.
"""

from __future__ import annotations
import os
import re

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

MODEL_NAME = "google/flan-t5-base"   # ~1 GB download on first use; falls back gracefully
MAX_INPUT_CHARS = 1200   # stay well inside the 512-token window
MAX_NEW_TOKENS  = 120

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline as hf_pipeline
        _pipeline = hf_pipeline(
            "text2text-generation",
            model=MODEL_NAME,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    return _pipeline


def _run(prompt: str) -> str:
    pipe = _get_pipeline()
    result = pipe(prompt[:MAX_INPUT_CHARS])
    return result[0]["generated_text"].strip()


# ──────────────────────────────────────────────────────────────────────
# Public functions
# ──────────────────────────────────────────────────────────────────────

def stepback_query(query: str) -> str:
    """
    Step-back abstraction: rewrite the specific query to a broader principle.

    Example:
      in : "What did Jeff Skilling approve for the storage vendor?"
      out: "What financial approvals were communicated in this thread?"
    """
    prompt = (
        "Rewrite the following question to be more general and abstract, "
        "removing specific names or amounts but keeping the core intent. "
        "Return only the rewritten question.\n\n"
        f"Question: {query}\n\n"
        "General version:"
    )
    result = _run(prompt)
    # Fallback: if model returns empty or repeats the prompt
    if not result or len(result) < 8 or result.lower() == query.lower():
        return f"what information in this thread is relevant to: {query}"
    return result


def hyde_document(query: str) -> str:
    """
    HyDE: generate a hypothetical email that would answer the query.
    The embedding of this fake email is then used for doc-to-doc search
    instead of query-to-doc search (Gao et al., 2022).

    Example:
      in : "What did finance approve for the storage vendor?"
      out: "Subject: Re: Storage Vendor Approval\n\nHi team,\nFinance has
            approved the storage vendor contract for $240,000..."
    """
    prompt = (
        "Write a short internal business email (3-4 sentences) that directly "
        "answers the following question. Write as if you are replying to a "
        "colleague. Be specific and include plausible details.\n\n"
        f"Question: {query}\n\n"
        "Email:"
    )
    result = _run(prompt)
    if not result or len(result) < 20:
        # Fallback: structured template (better than empty)
        return (
            f"Subject: Re: {query[:60]}\n\n"
            f"Hi team, following up on {query.rstrip('?')}. "
            f"Please see the details below and let me know if you have questions."
        )
    return result


def ground_sentence(chunk_text: str, query: str) -> str:
    """
    Ground-Every-Sentence (inspired by ResearchGate 2025):
    Generate exactly ONE sentence about `query` from `chunk_text`.

    Citation happens at generation time — each sentence is produced from a
    specific chunk and immediately attributed to it. This eliminates the
    post-hoc citation mismatch: the generated sentence cannot cite something
    it was not derived from.

    Contrast with the naive approach:
      Naive:  concatenate chunks → split into claims → find best matching chunk
      Here:   for each chunk → generate one sentence → cite that chunk

    flan-t5-small handles this well because the task is narrow:
    one chunk, one query, one sentence.
    """
    prompt = (
        "In one sentence, what does the following email excerpt say about "
        f"this topic: {query[:120]}\n\n"
        f"Excerpt: {chunk_text[:500]}\n\n"
        "One sentence answer:"
    )
    result = _run(prompt)
    if not result or len(result.split()) < 4:
        return ""
    # Strip any sentence beyond the first
    first = re.split(r"(?<=[.!?])\s", result)[0].strip()
    return first


def synthesize_answer(chunks: list[dict], query: str) -> str:
    """
    Synthesize a coherent answer from retrieved chunks rather than
    concatenating them verbatim.

    The synthesized text is used as the draft_answer in Agent 3.
    Citations are still validated against the original chunks.

    Example:
      chunks: ["Finance approved $240k on March 15...", "The vendor is Enserco..."]
      query : "What did finance approve and who is the vendor?"
      out   : "Finance approved $240,000 for Enserco on March 15."
    """
    if not chunks:
        return "No relevant information found in the selected thread."

    # Build context from top chunks (stay within token budget)
    context_parts = []
    char_budget = 900
    for c in chunks[:6]:
        text = c.get("text", "").strip()
        if len(text) + sum(len(p) for p in context_parts) > char_budget:
            break
        context_parts.append(text)

    context = " | ".join(context_parts)

    prompt = (
        "Answer the question using only the provided email excerpts. "
        "Be concise (2-3 sentences). If the answer is not in the excerpts, "
        "say 'The emails do not contain enough information to answer this.'\n\n"
        f"Excerpts: {context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    result = _run(prompt)
    if not result or len(result) < 10:
        # Fallback: extractive (concatenate top-3 chunks)
        return " ".join(c.get("text", "")[:200] for c in chunks[:3])
    return result
