"""
ingestion/chunker.py — Late Chunking (Günther et al., 2024) with naive fallback.

Late Chunking:
  1. Concatenate all text for a single document (email body or attachment).
  2. Run the FULL text through the embedding model in one forward pass.
  3. Apply chunk boundary splitting AFTER the transformer, BEFORE mean pooling.
  4. Mean-pool token embeddings within each chunk boundary.
  Result: each chunk embedding has seen the full document context.

Why this matters for email:
  Email bodies contain pronoun references ("it", "that contract", "her attachment")
  that only resolve with surrounding context. Naive chunking breaks these references.
  Late chunking preserves them at no training cost.

Fallback:
  If the document is longer than the model's context window (512 tokens for small models),
  we fall back to sliding-window chunking with 10% overlap and warn.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_MODEL_TOKENS = 512       # bge-small context window
ATTACHMENT_CHUNK_TOKENS = 300
ATTACHMENT_OVERLAP_TOKENS = 40
EMAIL_MAX_TOKENS = 512       # emails treated as single chunk if under this

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


# ──────────────────────────────────────────────
# Data class
# ──────────────────────────────────────────────

@dataclass
class TextChunk:
    text: str
    start_token: int
    end_token: int
    embedding: np.ndarray | None = None
    page_no: int | None = None


# ──────────────────────────────────────────────
# Tokeniser helper (uses the model's tokeniser)
# ──────────────────────────────────────────────

def tokenise(text: str) -> list[str]:
    model = get_model()
    return model.tokenizer.tokenize(text)


def token_count(text: str) -> int:
    return len(tokenise(text))


# ──────────────────────────────────────────────
# Late chunking
# ──────────────────────────────────────────────

def late_chunk_document(
    text: str,
    chunk_boundaries: list[tuple[int, int]],  # list of (start_char, end_char) per chunk
) -> list[TextChunk]:
    """
    Core late chunking implementation.

    Args:
        text: full document text
        chunk_boundaries: character-level boundaries for desired chunks

    Returns:
        TextChunks with contextualised embeddings via late pooling.
    """
    model = get_model()
    tokeniser = model.tokenizer

    # Tokenise the full document — single forward pass
    encoding = tokeniser(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_MODEL_TOKENS,
        return_offsets_mapping=True,
    )
    offset_mapping = encoding["offset_mapping"][0].tolist()  # (token_i, (start, end))

    # Get token-level embeddings from the model's last hidden state
    import torch
    with torch.no_grad():
        outputs = model._first_module().auto_model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            output_hidden_states=True,
        )
    # Shape: (1, seq_len, hidden_dim)
    token_embeddings = outputs.last_hidden_state[0].numpy()  # (seq_len, hidden_dim)

    chunks: list[TextChunk] = []
    for start_char, end_char in chunk_boundaries:
        chunk_text = text[start_char:end_char].strip()
        if not chunk_text:
            continue

        # Find which tokens correspond to this character span
        token_indices = [
            i for i, (ts, te) in enumerate(offset_mapping)
            if ts >= start_char and te <= end_char and ts < te
        ]

        if token_indices:
            # Mean pool token embeddings within this chunk's span
            chunk_emb = token_embeddings[token_indices].mean(axis=0)
        else:
            # Fallback: encode chunk independently
            chunk_emb = model.encode(chunk_text, normalize_embeddings=True)

        chunks.append(TextChunk(
            text=chunk_text,
            start_token=min(token_indices) if token_indices else 0,
            end_token=max(token_indices) if token_indices else 0,
            embedding=chunk_emb,
        ))

    return chunks


# ──────────────────────────────────────────────
# Email structure parsing
# ──────────────────────────────────────────────

# Patterns that mark the start of a forwarded/quoted chain in an email.
# Everything below the first match is quoted context already indexed elsewhere.
_FORWARD_BOUNDARY = re.compile(
    r"(?m)^[ \t]*("
    r"-{3,}\s*Forwarded\s+by"        # Enron / Lotus Notes
    r"|-{3,}\s*Original\s+Message"   # Outlook
    r"|On\s+.{5,80}wrote:"           # Gmail / standard
    r"|From:\s+\S.{0,80}\n"          # Outlook inline forward header block
    r"|>{3,}"                         # Heavy quoting prefix
    r")",
    re.IGNORECASE,
)

_SIGNATURE_BOUNDARY = re.compile(
    r"(?m)^[ \t]*(-{2,}|_{2,})\s*$",  # bare "-- " or "___" signature separator
)


def _strip_quoted_context(body: str) -> str:
    """
    Return only the NEW text the author wrote, discarding forwarded/quoted
    chains below the first forwarding marker.

    Keeps at least 80 chars so a bare forward (no personal text) still has
    something to index.
    """
    m = _FORWARD_BOUNDARY.search(body)
    if m and m.start() > 80:
        new_text = body[: m.start()].strip()
        return new_text if new_text else body

    # Also strip trailing signatures
    s = _SIGNATURE_BOUNDARY.search(body)
    if s and s.start() > 80:
        return body[: s.start()].strip()

    return body


# ──────────────────────────────────────────────
# Email chunking (one chunk per email message)
# ──────────────────────────────────────────────

def chunk_email(body: str) -> list[TextChunk]:
    """
    Structure-aware email chunking.

    1. Strip the forwarded/quoted chain — every reply repeats the full thread
       below the new text; indexing those duplicates degrades retrieval quality.
    2. Treat the remaining new text as one chunk if ≤ 512 tokens.
    3. For unusually long replies, split at paragraph boundaries with merging.
    """
    core = _strip_quoted_context(body)
    n_tokens = token_count(core)

    if n_tokens <= EMAIL_MAX_TOKENS:
        model = get_model()
        emb = model.encode(core, normalize_embeddings=True)
        return [TextChunk(text=core, start_token=0, end_token=n_tokens, embedding=emb)]

    # Long reply: split at blank-line paragraph boundaries
    paragraphs = re.split(r"\n\s*\n", core)
    boundaries = []
    pos = 0
    for para in paragraphs:
        start = core.find(para, pos)
        end = start + len(para)
        boundaries.append((start, end))
        pos = end

    merged = _merge_boundaries(core, boundaries, ATTACHMENT_CHUNK_TOKENS)
    return late_chunk_document(core, merged)


# ──────────────────────────────────────────────
# Attachment chunking (sliding window)
# ──────────────────────────────────────────────

def chunk_attachment(
    text: str,
    page_breaks: list[int] | None = None,  # char positions of page breaks
) -> list[TextChunk]:
    """
    Sliding-window chunking for PDFs and long attachments.
    If page_breaks are provided, hard-splits at page boundaries first,
    then slides within pages to respect page_no in citations.
    """
    if page_breaks:
        all_chunks: list[TextChunk] = []
        page_ranges = _page_break_ranges(text, page_breaks)
        for page_no, (ps, pe) in enumerate(page_ranges, start=1):
            page_text = text[ps:pe]
            page_chunks = _sliding_window_chunks(page_text, base_char_offset=ps)
            for c in page_chunks:
                c.page_no = page_no
            all_chunks.extend(page_chunks)
        return all_chunks

    return _sliding_window_chunks(text)


def _sliding_window_chunks(text: str, base_char_offset: int = 0) -> list[TextChunk]:
    """
    Splits text into overlapping character chunks approximating ATTACHMENT_CHUNK_TOKENS.
    Then applies late chunking over the full text to get contextualised embeddings.
    """
    words = text.split()
    # Approximate: avg 5 chars/word, 4 chars/token
    chars_per_chunk = ATTACHMENT_CHUNK_TOKENS * 4
    overlap_chars = ATTACHMENT_OVERLAP_TOKENS * 4

    boundaries = []
    start = 0
    while start < len(text):
        end = min(start + chars_per_chunk, len(text))
        # Snap to word boundary
        if end < len(text):
            snap = text.rfind(" ", start, end)
            if snap > start:
                end = snap
        boundaries.append((base_char_offset + start, base_char_offset + end))
        if end >= len(text):
            break  # covered the full text — do not create 1-char-shifted overflow chunks
        next_start = end - overlap_chars
        start = max(next_start, start + 1)  # always advance

    if token_count(text) <= MAX_MODEL_TOKENS:
        # Can do late chunking over the whole page/doc
        # Adjust boundaries to be relative to text
        local_bounds = [(s - base_char_offset, e - base_char_offset) for s, e in boundaries]
        chunks = late_chunk_document(text, local_bounds)
        # Restore absolute offsets
        for c, (s, e) in zip(chunks, boundaries):
            c.start_token += base_char_offset
            c.end_token += base_char_offset
        return chunks
    else:
        # Text too long for single pass — naive independent encoding with warning
        print(f"[WARN] Text ({token_count(text)} tokens) exceeds model window. "
              "Using independent chunk encoding (no late pooling).")
        model = get_model()
        result = []
        for s, e in boundaries:
            chunk_text = text[s - base_char_offset:e - base_char_offset].strip()
            if not chunk_text:
                continue
            emb = model.encode(chunk_text, normalize_embeddings=True)
            result.append(TextChunk(text=chunk_text, start_token=s, end_token=e, embedding=emb))
        return result


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _merge_boundaries(
    text: str,
    boundaries: list[tuple[int, int]],
    target_tokens: int,
) -> list[tuple[int, int]]:
    """Merge small adjacent boundaries until each bucket ≈ target_tokens."""
    merged = []
    buf_start, buf_end = boundaries[0]
    for s, e in boundaries[1:]:
        candidate = text[buf_start:e]
        if token_count(candidate) < target_tokens:
            buf_end = e
        else:
            merged.append((buf_start, buf_end))
            buf_start, buf_end = s, e
    merged.append((buf_start, buf_end))
    return merged


def _page_break_ranges(
    text: str,
    page_breaks: list[int],
) -> list[tuple[int, int]]:
    starts = [0] + [p + 1 for p in page_breaks]
    ends = page_breaks + [len(text)]
    return list(zip(starts, ends))
