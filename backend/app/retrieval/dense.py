"""
Dense retrieval using Qdrant vector similarity search.
Embeds the query with the same SentenceTransformer model used during ingestion,
then performs cosine similarity search against the 'rag_documents' collection.
"""

from functools import lru_cache
from typing import List

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.core.config import settings


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    device = "cuda" if settings.use_gpu else "cpu"
    return SentenceTransformer(settings.embedding_model, device=device)


@lru_cache(maxsize=1)
def _get_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
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
    k = top_k or settings.top_k
    model = _get_model()
    client = _get_client()

    query_vector = model.encode(query).tolist()

    results = client.search(
        collection_name=settings.collection_name,
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
