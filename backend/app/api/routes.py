"""
POST /query endpoint — the core RAG pipeline.

Flow:
  1. Embed query → Dense retrieval from Qdrant
  2. BM25 keyword retrieval from the cached corpus
  3. Simple score-normalised fusion → deduplicated top-K chunks
  4. Generation via LLM
  5. Return answer + sources
"""

from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.config import settings
from app.retrieval import dense, bm25
from app.generation import controller

router = APIRouter()

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Chunks to retrieve (default from settings)")

class SourceInfo(BaseModel):
    source_file: str
    chunk_index: int
    score: float
    retriever: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceInfo]
    num_chunks_used: int


def _fuse_results(dense_results: List[dict], bm25_results: List[dict], top_k: int) -> List[dict]:
    """
    Normalise scores within each retriever to [0, 1], combine by chunk text key,
    sum the normalised scores, and return the top-K unique chunks.
    """
    def _normalize(results: List[dict]) -> List[dict]:
        if not results:
            return []
        max_score = max(r["score"] for r in results)
        if max_score == 0:
            return results
        return [{**r, "score": r["score"] / max_score} for r in results]

    merged: dict[str, dict] = {}

    for chunk in _normalize(dense_results) + _normalize(bm25_results):
        key = chunk["text"]
        if key in merged:
            merged[key]["score"] += chunk["score"]
            # Keep the retriever label informative
            if merged[key]["retriever"] != chunk["retriever"]:
                merged[key]["retriever"] = "dense+bm25"
        else:
            merged[key] = dict(chunk)

    ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


@router.post("/query", response_model=QueryResponse, summary="RAG Query")
async def query(request: QueryRequest):
    """
    Submit a question. The system retrieves the most relevant
    document chunks (dense vector search + BM25 keyword search), fuses them,
    and asks LLM to generate a grounded answer.
    """
    k = request.top_k or settings.top_k

    dense_results = dense.search(request.query, top_k=k)
    bm25_results  = bm25.search(request.query,  top_k=k)

    fused_chunks = _fuse_results(dense_results, bm25_results, top_k=k)

    answer = controller.generate(request.query, fused_chunks)

    sources = [
        SourceInfo(
            source_file=c["source_file"],
            chunk_index=c["chunk_index"],
            score=round(c["score"], 4),
            retriever=c["retriever"],
        )
        for c in fused_chunks
    ]

    return QueryResponse(
        query=request.query,
        answer=answer,
        sources=sources,
        num_chunks_used=len(fused_chunks),
    )
