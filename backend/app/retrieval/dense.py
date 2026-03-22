"""
Dense retrieval using Qdrant vector similarity search.
Embeds the query with the same SentenceTransformer model used during ingestion,
then performs cosine similarity search against the 'rag_documents' collection.
"""

from functools import lru_cache
from typing import List

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

import os


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    device = "cuda" if int(os.environ.get("USE_GPU", 0)) else "cpu"
    return SentenceTransformer(os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"), device=device)


@lru_cache(maxsize=1)
def _get_client() -> QdrantClient:
    return QdrantClient(
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        api_key=os.environ.get("QDRANT_API_KEY") or None,
    )


def search(query: str, top_k: int = None) -> List[dict]:
    """
    Embed the query and return the top-K most similar chunks from Qdrant.

    Returns a list of dicts:
        {
            "text": str,
            "source_file": str,
            "document_id": str,
            "chunk_index": int,
            "score": float,
            "retriever": "dense"
        }
    """
    k = top_k or int(os.environ.get("TOP_K", 5))
    model = _get_model()
    client = _get_client()

    query_vector = model.encode(query).tolist()

    results = client.search(
        collection_name=os.environ.get("COLLECTION_NAME", "rag_documents"),
        query_vector=query_vector,
        limit=k,
        with_payload=True,
    )

    return [
        {
            "text": hit.payload.get("text", ""),
            "source_file": hit.payload.get("source_file", ""),
            "document_id": hit.payload.get("document_id", ""),
            "chunk_index": hit.payload.get("chunk_index", -1),
            "score": hit.score,
            "retriever": "dense",
        }
        for hit in results
    ]
