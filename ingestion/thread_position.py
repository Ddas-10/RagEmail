"""
ingestion/thread_position.py — Temporal position encoding for email-thread chunks.

Each email in a thread is assigned a normalised temporal position ∈ [0.0, 1.0]:
  0.0 = earliest message (typically the original announcement)
  1.0 = most recent reply

The position is encoded as a sinusoidal signal (Vaswani et al., 2017, "Attention Is
All You Need") adapted for continuous document positions instead of integer token
positions, then soft-blended (alpha=0.05 by default) into the chunk's CLS-style
embedding so that:

  • Announcement emails  (pos ≈ 0.0) are semantically distinguishable from late replies
  • Thread synopses capture temporal spread rather than a farewell-biased centroid
  • "Summarise the thread" queries naturally match the synopsis without regex heuristics

Usage:
    positions = compute_thread_positions(records)          # {message_id: float}
    emb_with_pos = blend_position_into_embedding(emb, 0.0) # announcement email
    synopsis = build_temporally_diverse_synopsis(chunks, emb_matrix, n_reps=5)
"""

from __future__ import annotations
import math
from datetime import datetime

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 1. Temporal position assignment
# ──────────────────────────────────────────────────────────────────────────────

def compute_thread_positions(records: list) -> dict[str, float]:
    """
    Assign each email a normalised temporal position ∈ [0.0, 1.0] within its thread.
    Sorted by send date; ties broken by list order.

    Returns: {message_id: position_float}
    """
    from dateutil import parser as dp

    by_thread: dict[str, list] = {}
    for rec in records:
        by_thread.setdefault(rec.thread_id, []).append(rec)

    positions: dict[str, float] = {}
    for _, thread_records in by_thread.items():
        def _parse_date(r) -> datetime:
            try:
                return dp.parse(r.date)
            except Exception:
                return datetime.min

        ordered = sorted(thread_records, key=_parse_date)
        n = len(ordered)
        for i, rec in enumerate(ordered):
            # Single-email thread → position 0.0 (no denominator issue)
            positions[rec.message_id] = float(i) / max(n - 1, 1)

    return positions


# ──────────────────────────────────────────────────────────────────────────────
# 2. Sinusoidal positional encoding
# ──────────────────────────────────────────────────────────────────────────────

def sinusoidal_position_encoding(position: float, dim: int) -> np.ndarray:
    """
    Sinusoidal positional encoding for a continuous position ∈ [0, 1].

    Adapted from Vaswani et al. (2017) Eq. 1–2:
      pe[2i]   = sin(position / wavelength_i)
      pe[2i+1] = cos(position / wavelength_i)
      wavelength_i = 10000 ^ (2i / dim)

    Using a continuous position (rather than an integer token index) gives a
    smooth positional manifold: emails close in time produce similar pe vectors,
    while emails far apart in time produce orthogonal ones.
    """
    pe = np.zeros(dim, dtype="float32")
    for i in range(dim // 2):
        wavelength = 10000.0 ** (2.0 * i / max(dim, 1))
        pe[2 * i]     = math.sin(position / wavelength)
        pe[2 * i + 1] = math.cos(position / wavelength)
    return pe


def blend_position_into_embedding(
    embedding: np.ndarray,
    position: float,
    alpha: float = 0.05,
) -> np.ndarray:
    """
    Soft-blend a sinusoidal positional signal into a semantic CLS embedding.

    The blend is:   mixed = (1 - alpha) * semantic + alpha * pe
    then renormalised to the unit sphere so cosine similarity remains valid.

    alpha controls the positional influence:
      0.00 → pure semantic (unchanged)
      0.05 → 5 % positional (default; breaks cosine ties between equally relevant
              chunks from different positions without degrading retrieval quality)
      0.20 → stronger positional emphasis (use only for position-first ranking)
    """
    if alpha == 0.0 or embedding.size == 0:
        return embedding.astype("float32")
    pe    = sinusoidal_position_encoding(position, len(embedding))
    mixed = (1.0 - alpha) * embedding.astype("float32") + alpha * pe
    norm  = np.linalg.norm(mixed)
    return mixed / (norm + 1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Temporally-diverse thread synopsis
# ──────────────────────────────────────────────────────────────────────────────

def build_temporally_diverse_synopsis(
    thread_chunk_indices: list[int],
    thread_positions: list[float],
    emb_matrix: np.ndarray,
    n_reps: int = 5,
) -> np.ndarray:
    """
    Compute a thread synopsis embedding that captures temporal diversity.

    Unlike a simple mean (which is biased toward whichever email type dominates
    a thread — usually farewell/congratulatory replies in announcement threads),
    this samples one representative from each temporal quantile and averages them.

    For a thread with 20 emails (5 substantive, 15 farewells):
      • Simple mean ≈ 75 % farewell — biased against the announcement content
      • Temporal synopsis (5 quantiles) ≈ equal weight across the thread arc

    Args:
        thread_chunk_indices: indices into emb_matrix for this thread's chunks
        thread_positions: parallel list of thread_position floats (one per chunk)
        emb_matrix: (N, D) float32 array of all chunk embeddings
        n_reps: number of temporal quantile representatives

    Returns: unit-normalised synopsis vector of shape (D,)
    """
    if not thread_chunk_indices:
        return np.zeros(emb_matrix.shape[1], dtype="float32")

    # Sort indices by temporal position
    paired = sorted(zip(thread_positions, thread_chunk_indices))
    n = len(paired)
    n_reps = min(n_reps, n)

    # Pick one representative from each quantile (the middle element)
    step = max(n // n_reps, 1)
    rep_indices: list[int] = []
    for q in range(n_reps):
        start = q * step
        end   = min(start + step, n)
        mid   = (start + end) // 2
        _, chunk_idx = paired[mid]
        rep_indices.append(chunk_idx)

    rep_embs = emb_matrix[rep_indices].astype("float32")
    synopsis  = rep_embs.mean(axis=0)
    norm      = np.linalg.norm(synopsis)
    return synopsis / (norm + 1e-9)
