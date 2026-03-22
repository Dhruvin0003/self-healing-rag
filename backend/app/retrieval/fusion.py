"""
Merges ranked result lists from Dense, BM25, and (optionally) Graph chunk
retrievers using Reciprocal Rank Fusion (RRF).

Algorithm:
    RRF_score(doc) = Σ  1 / (k + rank_i)   for each retriever i that found the doc

where `k = 60` is the standard smoothing constant recommended in the original
Cormack, Clarke & Buettcher (2009) paper.

Deduplication key: (document_id, chunk_index)
  - If two results share the same key, their RRF scores are summed and the
    entry with the higher individual score is kept as the canonical record.
  - Falls back to text-based deduplication when document_id/chunk_index are
    not available (e.g. synthetic chunks).

Usage:
    from app.retrieval.fusion import fuse
    top_chunks = fuse(dense_results, bm25_results, top_k=5)
"""

from typing import Sequence

# Standard RRF smoothing constant
RRF_K: int = 60

def dedup_key(chunk: dict) -> str:
    """
    Stable deduplication key for a chunk.

    Prefers (document_id, chunk_index) when available; falls back to the
    first 200 characters of the chunk text.
    """
    doc_id = chunk.get("document_id", "")
    chunk_idx = chunk.get("chunk_index", -1)
    if doc_id:
        return f"{doc_id}::{chunk_idx}"
    return chunk.get("text", "")[:200]


def fuse(*ranked_lists: Sequence[dict],top_k: int = 5,) -> list[dict]:
    """
    Fuse multiple ranked retrieval result lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    *ranked_lists:
        One or more lists of chunk dicts, each already sorted by their
        retriever's score (highest first).  Each dict must contain at least:
            - "text"         : str
            - "score"        : float  (original retriever score)
            - "retriever"    : str    (e.g. "dense", "bm25")
            - "document_id"  : str    (used for dedup; can be empty)
            - "chunk_index"  : int    (used for dedup; can be -1)
    top_k:
        Maximum number of chunks to return.

    Returns
    -------
    list[dict]
        Top-K unique chunks, sorted by descending RRF score.
        Each chunk gains two extra fields:
            - "rrf_score"  : float  — the combined RRF score
            - "retriever"  : str    — updated to reflect all retrievers that
                                      contributed (e.g. "dense+bm25")
    """
    # Map from dedup_key → best chunk record
    best: dict[str, dict] = {}
    # Map from dedup_key → accumulated RRF score
    rrf_scores: dict[str, float] = {}
    # Map from dedup_key → set of retriever names that found this chunk
    retrievers_seen: dict[str, set[str]] = {}

    for ranked_list in ranked_lists:
        if not ranked_list:
            continue
        for rank, chunk in enumerate(ranked_list, start=1):
            key = dedup_key(chunk)
            contribution = 1.0 / (RRF_K + rank)

            if key not in rrf_scores:
                rrf_scores[key] = 0.0
                best[key] = dict(chunk)
                retrievers_seen[key] = set()

            rrf_scores[key] += contribution

            # Keep the version of the chunk with the highest raw score
            if chunk.get("score", 0.0) > best[key].get("score", 0.0):
                best[key] = dict(chunk)

            # Track which retrievers contributed
            retrievers_seen[key].add(chunk.get("retriever", "unknown"))

    if not best:
        return []

    # Attach RRF score and merged retriever label
    results: list[dict] = []
    for key, chunk in best.items():
        chunk["rrf_score"] = round(rrf_scores[key], 6)
        chunk["retriever"] = "+".join(sorted(retrievers_seen[key]))
        results.append(chunk)

    # Sort by RRF score descending
    results.sort(key=lambda c: c["rrf_score"], reverse=True)
    top = results[:top_k]

    return top
