"""
BM25 keyword retrieval.
On first call, scrolls all chunk payloads from Qdrant and builds an in-memory
BM25Okapi index.  Subsequent calls reuse the cached index.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Tuple

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

import os


def _tokenize(text: str) -> List[str]:
    """Lowercase + simple whitespace tokenisation."""
    return re.findall(r"\w+", text.lower())


@lru_cache(maxsize=1)
def _build_index() -> Tuple[BM25Okapi, List[dict]]:
    """
    Scroll every point from Qdrant, collect their payloads, build a BM25 index.
    Cached so this expensive operation only runs once per server lifetime.
    """
    client = QdrantClient(
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        api_key=os.environ.get("QDRANT_API_KEY") or None,
    )

    all_chunks: List[dict] = []
    offset = None

    while True:
        response, next_offset = client.scroll(
            collection_name=os.environ.get("COLLECTION_NAME", "rag_documents"),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in response:
            all_chunks.append(point.payload)

        if next_offset is None:
            break
        offset = next_offset

    if not all_chunks:
        raise RuntimeError(
            "BM25: no chunks found in Qdrant — run data_ingestion.py first."
        )

    tokenized_corpus = [_tokenize(chunk.get("text", "")) for chunk in all_chunks]
    index = BM25Okapi(tokenized_corpus)
    return index, all_chunks


def search(query: str, top_k: int = None) -> List[dict]:
    """
    Run BM25 keyword search over the corpus.

    Returns a list of dicts:
        {
            "text": str,
            "source_file": str,
            "document_id": str,
            "chunk_index": int,
            "score": float,
            "retriever": "bm25"
        }
    """
    k = top_k or int(os.environ.get("TOP_K", 5))
    index, all_chunks = _build_index()

    tokenized_query = _tokenize(query)
    scores = index.get_scores(tokenized_query)

    # Get indices of top-K highest scores
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    results = []
    for idx in top_indices:
        if scores[idx] == 0.0:
            # BM25 score of 0 means no keyword overlap — skip
            continue
        chunk = all_chunks[idx]
        results.append(
            {
                "text": chunk.get("text", ""),
                "source_file": chunk.get("source_file", ""),
                "document_id": chunk.get("document_id", ""),
                "chunk_index": chunk.get("chunk_index", -1),
                "score": float(scores[idx]),
                "retriever": "bm25",
            }
        )
    return results
