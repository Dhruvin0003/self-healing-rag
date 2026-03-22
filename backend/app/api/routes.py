"""
POST /query        — Dense + BM25 + Graph retrieval RAG pipeline.
POST /ingest/graph — Extract entities from a chunk and store in Neo4j.

Query flow:
  1. Embed query → Dense retrieval (Qdrant)
  2. BM25 keyword retrieval
  3. Score-normalized fusion → deduplicated top-K chunks
  4. Graph retrieval → entity sub-graph + concept/domain hierarchy
  5. Generation via Gemini (vector context + graph knowledge)
  6. Return answer + sources + graph_context + concept_context
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import os
from app.retrieval import dense, bm25
from app.generation import controller
from app.graph import extractor, builder, retriever

logger = logging.getLogger(__name__)

router = APIRouter()

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Chunks to retrieve (default from settings)")
    use_graph: bool = Field(True, description="Whether to include graph context in the answer")

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
    graph_context: List[str] = []
    concept_context: List[str] = []

class ChunkInput(BaseModel):
    text: str = Field(..., min_length=10, description="Raw text of the chunk")
    chunk_id: str = Field(..., description="Unique identifier for this chunk")

class GraphBatchIngestRequest(BaseModel):
    chunks: List[ChunkInput] = Field(..., description="List of chunks to ingest")

class GraphBatchIngestResponse(BaseModel):
    chunks_processed: int
    entities_extracted: int
    relations_extracted: int
    entities: list
    relations: list


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
            if merged[key]["retriever"] != chunk["retriever"]:
                merged[key]["retriever"] = "dense+bm25"
        else:
            merged[key] = dict(chunk)

    ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


@router.post("/query", response_model=QueryResponse, summary="RAG Query")
async def query(request: QueryRequest):
    """
    Submit a question. The system retrieves the most relevant document chunks
    (dense vector search + BM25 keyword search) and sub-graph from Neo4j,
    fuses them, and asks Gemini to generate a grounded answer.
    """
    k = request.top_k or int(os.environ.get("TOP_K", 5))

    # Vector + keyword retrieval
    dense_results = dense.search(request.query, top_k=k)
    bm25_results = bm25.search(request.query, top_k=k)
    fused_chunks = _fuse_results(dense_results, bm25_results, top_k=k)

    # Graph retrieval
    graph_triples: List[str] = []
    concept_triples: List[str] = []

    if request.use_graph:
        try:
            graph_result = retriever.search(request.query)
            graph_triples = graph_result.get("graph_triples", [])
            concept_triples = graph_result.get("concept_triples", [])
        except Exception as exc:
            # Graph retrieval is best-effort; never block the answer
            logger.warning("Graph retrieval failed (non-blocking): %s", exc)

    #Generation
    answer = controller.generate(
        query=request.query,
        chunks=fused_chunks,
        graph_triples=graph_triples,
        concept_triples=concept_triples,
    )

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
        graph_context=graph_triples,
        concept_context=concept_triples,
    )


@router.post("/ingest/graph", response_model=GraphBatchIngestResponse, summary="Graph Batch Ingestion")
async def ingest_graph(request: GraphBatchIngestRequest):
    """
    Extract entities, concepts, domains and relationships from a batch of text chunks,
    then upsert them into the Neo4j knowledge graph.
    """
    try:
        chunks_dicts = [{"text": c.text, "chunk_id": c.chunk_id} for c in request.chunks]
        extraction = extractor.extract(chunks=chunks_dicts)
        builder.build(extraction=extraction, chunks=chunks_dicts)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return GraphBatchIngestResponse(
        chunks_processed=len(request.chunks),
        entities_extracted=len(extraction.get("entities", [])),
        relations_extracted=len(extraction.get("relations", [])),
        entities=extraction.get("entities", []),
        relations=extraction.get("relations", []),
    )
