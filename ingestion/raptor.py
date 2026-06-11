"""
ingestion/raptor.py — RAPTOR tree (Sarthi et al., 2024) per thread.

Algorithm:
  1. Take all leaf chunks for a thread.
  2. Embed them (already done by chunker).
  3. Reduce to 2D with UMAP for clustering.
  4. Cluster with Gaussian Mixture Model.
  5. For each cluster: generate a summary chunk using TF-IDF extractive selection.
  6. Recurse: treat summaries as new leaves until ≤ 1 cluster remains.
  7. Return all intermediate + root summary chunks.

Minimum thread size: 5 chunks. Smaller threads are skipped.
These summary chunks are added to both BM25 and FAISS indices.

Summarisation strategy: TF-IDF extractive (sklearn, already a dependency).
Every word in a RAPTOR chunk is sourced directly from the original emails —
zero hallucination risk. distilbart-cnn-12-6 was removed because it was
fine-tuned on news and garbled conversational email text into nonsense portmanteau
words ("gouproformative", "ipient") on non-news inputs.
"""

from __future__ import annotations
import re
import numpy as np


MIN_CLUSTER_SIZE = 5  # UMAP needs at least a few more points than n_components
MAX_TREE_LEVELS = 3


def _cluster_chunks(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """Gaussian Mixture Model clustering on UMAP-reduced embeddings."""
    n_clusters = max(2, min(n_clusters, len(embeddings) // MIN_CLUSTER_SIZE))
    try:
        import umap
        n_components = min(2, embeddings.shape[1] - 1)
        # n_neighbors must be < n_samples; UMAP requires at least n_components+2 samples
        n_neighbors = min(15, max(2, len(embeddings) - 1))
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            random_state=42,
            min_dist=0.0,
        )
        reduced = reducer.fit_transform(embeddings)
    except Exception:
        # UMAP failed (too few points or scipy error) — fall back to PCA-2d slice
        reduced = embeddings[:, :2]

    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=n_clusters, random_state=42)
    labels = gm.fit_predict(reduced)
    return labels


def _summarise(texts: list[str]) -> str:
    """
    TF-IDF extractive summary: pick the 4 highest-scoring sentences from the
    cluster using sklearn TfidfVectorizer (already a project dependency).

    Every word in the result comes directly from the source emails, so RAPTOR
    summary chunks cannot contain hallucinated or garbled words.
    """
    combined = " ".join(texts)[:4000]

    # Split into candidate sentences
    raw = re.split(r"(?<=[.!?])\s+|\n{2,}", combined)
    sentences = [s.strip() for s in raw if len(s.split()) >= 6]
    if not sentences:
        return combined[:500]
    if len(sentences) <= 4:
        return " ".join(sentences)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(max_features=400, stop_words="english", sublinear_tf=True)
        mat = tfidf.fit_transform(sentences)
        scores = np.asarray(mat.sum(axis=1)).flatten()
        # Keep top-4, preserving original order for readability
        top_idx = sorted(np.argsort(scores)[-4:].tolist())
        return " ".join(sentences[i] for i in top_idx)
    except Exception:
        return " ".join(sentences[:4])


def build_raptor_tree(
    thread_id: str,
    leaf_chunks: list[dict],
    embed_model,
    level: int = 1,
) -> list[dict]:
    """
    Recursively build a RAPTOR summary tree for one thread.
    Returns a flat list of all intermediate summary chunks to add to the index.
    """
    if len(leaf_chunks) < MIN_CLUSTER_SIZE or level > MAX_TREE_LEVELS:
        return []

    # Get embeddings for all chunks
    embeddings = []
    for c in leaf_chunks:
        if c.get("embedding"):
            embeddings.append(np.array(c["embedding"], dtype="float32"))
        else:
            emb = embed_model.encode(c["text"], normalize_embeddings=True)
            embeddings.append(emb)
    emb_matrix = np.stack(embeddings)

    n_clusters = max(2, len(leaf_chunks) // MIN_CLUSTER_SIZE)
    labels = _cluster_chunks(emb_matrix, n_clusters)

    summary_chunks = []

    for cluster_id in np.unique(labels):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_texts = [leaf_chunks[i]["text"] for i in cluster_indices]
        cluster_msg_ids = list({leaf_chunks[i]["message_id"] for i in cluster_indices})

        summary_text = _summarise(cluster_texts)
        summary_emb = embed_model.encode(summary_text, normalize_embeddings=True)

        summary_chunk = {
            "doc_id":     f"{thread_id}__raptor__L{level}__C{cluster_id}",
            "thread_id":  thread_id,
            "message_id": cluster_msg_ids[0],  # representative message
            "source":     "email",
            "text":       summary_text,
            "page_no":    None,
            "raptor_level": level,
            "raptor_source_ids": [leaf_chunks[i]["doc_id"] for i in cluster_indices],
            "embedding":  summary_emb.tolist(),
        }
        summary_chunks.append(summary_chunk)

    # Recurse: summaries become new leaves
    deeper = build_raptor_tree(thread_id, summary_chunks, embed_model, level + 1)
    return summary_chunks + deeper
