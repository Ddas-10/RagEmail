"""
ingest.py — Full indexing pipeline entry point.

Run:
    python ingest.py --data-dir data/slice/

Steps:
  1. Parse all .eml files → EmailRecord list
  2. Parse all attachments → Chunk list with page_no
  3. Late-chunk email bodies
  4. Build RAPTOR tree per thread
  5. Build BM25 index (rank_bm25)
  6. Build FAISS index (flat IP, pre-normalised)
  7. Persist both indices + chunk metadata to disk
  8. Write index manifest (thread_ids, counts)
"""

from __future__ import annotations
import argparse
import json
import os
import pickle
from pathlib import Path

# Disable TensorFlow integration in transformers/sentence-transformers.
# Keras 3 (installed by TF) breaks the import chain — PyTorch path works fine.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from ingestion.parse_eml import load_eml_directory
from ingestion.parse_attachments import parse_attachments_for_thread
from ingestion.chunker import chunk_email, chunk_attachment
from ingestion.raptor import build_raptor_tree
from ingestion.thread_position import (
    compute_thread_positions,
    blend_position_into_embedding,
    build_temporally_diverse_synopsis,
)

INDEX_DIR = Path("data/index")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"


def run_ingest(data_dir: Path):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 1/6  Parsing emails…")
    records = load_eml_directory(data_dir)

    # Compute temporal position of each email within its thread (0.0 = earliest, 1.0 = latest).
    # Used to blend sinusoidal positional encoding into CLS embeddings so that announcement
    # emails and late replies are distinguishable in vector space without regex heuristics.
    positions = compute_thread_positions(records)

    print("Step 2/6  Parsing attachments…")
    attachment_chunks: list[dict] = []
    for rec in records:
        att_chunks = parse_attachments_for_thread(data_dir, rec)
        attachment_chunks.extend(att_chunks)

    print("Step 3/6  Chunking email bodies (late chunking)…")
    embed_model = SentenceTransformer(EMBED_MODEL)
    all_chunks: list[dict] = []

    for rec in records:
        if not rec.body.strip():
            continue
        text_chunks = chunk_email(rec.body)
        pos = positions.get(rec.message_id, 0.0)
        for i, tc in enumerate(text_chunks):
            all_chunks.append({
                "doc_id":          f"{rec.message_id}__email__{i}",
                "thread_id":       rec.thread_id,
                "message_id":      rec.message_id,
                "source":          "email",
                "text":            tc.text,
                "page_no":         None,
                "thread_position": pos,   # normalised [0.0, 1.0] within the thread
                "participants":    rec.participants,  # display names from From/To/Cc headers
                "embedding":       tc.embedding.tolist() if tc.embedding is not None else None,
            })

    for ac in attachment_chunks:
        all_chunks.append(ac)

    print(f"         Total chunks: {len(all_chunks)}")

    print("Step 4/6  Building RAPTOR trees…")
    thread_ids = list({c["thread_id"] for c in all_chunks})
    raptor_chunks: list[dict] = []
    for tid in thread_ids:
        tid_chunks = [c for c in all_chunks if c["thread_id"] == tid]
        if len(tid_chunks) >= 5:
            tree_chunks = build_raptor_tree(tid, tid_chunks, embed_model)
            raptor_chunks.extend(tree_chunks)
    all_chunks.extend(raptor_chunks)
    print(f"         RAPTOR summary chunks added: {len(raptor_chunks)}")

    print("Step 5/6  Building BM25 index…")
    tokenised_corpus = [c["text"].lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenised_corpus)

    print("Step 6/6  Building FAISS index with positional embeddings…")
    # Build embeddings.  For email chunks we soft-blend a sinusoidal positional
    # encoding (alpha=0.05) derived from each email's temporal position in its
    # thread.  This makes announcement emails (pos≈0) and late farewell replies
    # (pos≈1) distinguishable in cosine space — no keyword heuristics needed.
    # RAPTOR and attachment chunks have no thread position; they are left as-is.
    embeddings = []
    for chunk in all_chunks:
        if chunk.get("embedding"):
            raw_emb = np.array(chunk["embedding"], dtype="float32")
        else:
            raw_emb = embed_model.encode(chunk["text"], normalize_embeddings=True).astype("float32")
            chunk["embedding"] = raw_emb.tolist()

        pos = chunk.get("thread_position")
        if pos is not None and chunk.get("source") == "email":
            # Blend positional signal: 95% semantic CLS + 5% sinusoidal PE
            emb = blend_position_into_embedding(raw_emb, pos, alpha=0.05)
        else:
            emb = raw_emb
        embeddings.append(emb)

    emb_matrix = np.stack(embeddings, axis=0).astype("float32")  # (N, D)
    faiss.normalize_L2(emb_matrix)

    dim = emb_matrix.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(emb_matrix)

    print(f"         FAISS index: {faiss_index.ntotal} vectors, dim={dim}")

    # ── Thread synopsis vectors ──────────────────────────────────────────────
    # For each thread, compute a temporally-diverse synopsis: mean of embeddings
    # sampled from 5 temporal quantiles.  Unlike a simple mean (biased toward the
    # most common email type — usually farewells in announcement threads), this
    # represents the full arc of the thread from first announcement to last reply.
    # The synopsis is used in retrieval as an extra FAISS query vector so that
    # summary-intent queries get representative content without a BM25 keyword match.
    print("         Computing temporally-diverse thread synopsis vectors…")
    thread_synopses: dict[str, list] = {}
    for tid in thread_ids:
        # Index positions and temporal positions for email chunks in this thread
        tid_indices: list[int] = []
        tid_positions: list[float] = []
        for i, c in enumerate(all_chunks):
            if (c["thread_id"] == tid
                    and c.get("source") == "email"
                    and "__raptor__" not in c.get("doc_id", "")
                    and "thread_position" in c):
                tid_indices.append(i)
                tid_positions.append(c["thread_position"])

        if not tid_indices:
            continue

        synopsis = build_temporally_diverse_synopsis(
            tid_indices, tid_positions, emb_matrix, n_reps=5
        )
        thread_synopses[tid] = synopsis.tolist()

    with open(INDEX_DIR / "thread_synopses.json", "w") as f:
        json.dump(thread_synopses, f)
    print(f"         Thread synopses: {len(thread_synopses)} threads")

    # ── Thread participant roster ────────────────────────────────────────────
    # Pre-compute {thread_id: [name, ...]} from email chunks (not RAPTOR).
    # RAPTOR chunks have no participants field; this roster lets person queries
    # find participants regardless of which chunks the retriever happens to rank first.
    print("         Building thread participant rosters…")
    thread_roster: dict[str, list[str]] = {}
    for tid in thread_ids:
        freq: dict[str, int] = {}
        canonical: dict[str, str] = {}
        for c in all_chunks:
            if c["thread_id"] != tid or "__raptor__" in c.get("doc_id", ""):
                continue
            for name in c.get("participants") or []:
                key = name.lower()
                freq[key] = freq.get(key, 0) + 1
                canonical.setdefault(key, name)
        sorted_keys = sorted(freq, key=lambda k: -freq[k])
        thread_roster[tid] = [canonical[k] for k in sorted_keys]

    with open(INDEX_DIR / "thread_roster.json", "w") as f:
        json.dump(thread_roster, f, indent=2)
    print(f"         Thread rosters: {len(thread_roster)} threads, "
          f"{sum(len(v) for v in thread_roster.values())} total names")

    # ── Persist ──
    chunks_no_emb = [{k: v for k, v in c.items() if k != "embedding"} for c in all_chunks]

    faiss.write_index(faiss_index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(INDEX_DIR / "chunks.json", "w") as f:
        json.dump(chunks_no_emb, f)
    np.save(str(INDEX_DIR / "embeddings.npy"), emb_matrix)

    manifest = {
        "total_chunks": len(all_chunks),
        "raptor_chunks": len(raptor_chunks),
        "thread_ids": thread_ids,
        "embed_model": EMBED_MODEL,
        "dim": dim,
    }
    with open(INDEX_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 60)
    print(f"[OK] Index written to {INDEX_DIR}/")
    print(f"  threads:       {len(thread_ids)}")
    print(f"  total chunks:  {len(all_chunks)}")
    print(f"  RAPTOR chunks: {len(raptor_chunks)}")
    print(f"  synopses:      {len(thread_synopses)}")


def load_index():
    """Load all index artefacts. Called at API startup."""
    from retrieval.hybrid import load_indices, load_thread_synopses

    with open(INDEX_DIR / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(INDEX_DIR / "chunks.json") as f:
        chunks = json.load(f)

    faiss_idx = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    embed_model = SentenceTransformer(EMBED_MODEL)

    try:
        reranker = CrossEncoder(RERANKER_MODEL)
    except Exception:
        print("[WARN] Reranker not loaded — falling back to RRF order")
        reranker = None

    load_indices(bm25, chunks, faiss_idx, chunks, embed_model, reranker)
    print(f"[INFO] Index loaded: {len(chunks)} chunks")

    # Load temporally-diverse thread synopsis vectors (built during ingest).
    # These allow retrieval to use a representative thread embedding as an
    # additional FAISS query — no BM25 keyword match required for summary intent.
    synopsis_path = INDEX_DIR / "thread_synopses.json"
    if synopsis_path.exists():
        with open(synopsis_path) as f:
            synopses = json.load(f)
        load_thread_synopses(synopses)
        print(f"[INFO] Thread synopses loaded: {len(synopses)} threads")


def get_all_thread_ids() -> list[str]:
    try:
        with open(INDEX_DIR / "manifest.json") as f:
            return json.load(f)["thread_ids"]
    except Exception:
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/slice/", type=Path)
    args = parser.parse_args()
    run_ingest(args.data_dir)
