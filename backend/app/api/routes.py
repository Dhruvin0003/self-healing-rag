"""
POST /query        — Intelligent RAG pipeline.
POST /ingest/graph — Extract entities from a chunk and store in Neo4j.

Query flow:
  1. Query Intelligence  — classify query; decide if graph is needed
  2. Retriever Router    — dispatch Dense + BM25 + (optional) Graph
  3. RRF Fusion          — merge & deduplicate with Reciprocal Rank Fusion
  4. Generation          — Gemini grounded answer from fused context
  5. Return              — answer + sources + graph_context + metadata
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import os
from app.retrieval import query_intelligence, router as retriever_router, fusion
from app.generation import controller
from app.graph import extractor, builder

router = APIRouter()

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Chunks to retrieve (default from env TOP_K)")
    use_graph: bool = Field(True, description="Allow graph retrieval when query warrants it")

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
    query_type: str = "factual"
    needs_graph: bool = False
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


@router.post("/query", response_model=QueryResponse, summary="RAG Query")
async def query(request: QueryRequest):
    """
    Submit a question.

    pipeline:
      1. **Query Intelligence** classifies the query and sets `needs_graph`.
      2. **Retriever Router** dispatches Dense + BM25 always; Graph only when needed.
      3. **RRF Fusion** merges all results into a single ranked context block.
      4. **Generation Controller** produces a grounded answer via Gemini.
    """
    k = request.top_k or int(os.environ.get("TOP_K", 5))

    # Classify the query
    profile = query_intelligence.classify(request.query)

    if not request.use_graph:
        profile.needs_graph = False

    # Route to appropriate retrievers
    route_result = retriever_router.route(
        query=request.query,
        profile=profile,
        top_k=k,
    )

    # RRF Fusion of Dense + BM25 results
    fused_chunks = fusion.fuse(
        route_result["dense"],
        route_result["bm25"],
        top_k=k,
    )

    graph_triples = route_result["graph_triples"]
    concept_triples = route_result["concept_triples"]

    # Generate grounded answer
    answer = controller.generate(
        query=request.query,
        chunks=fused_chunks,
        graph_triples=graph_triples,
        concept_triples=concept_triples,
    )

    sources = [
        SourceInfo(
            source_file=c.get("source_file", ""),
            chunk_index=c.get("chunk_index", -1),
            score=round(c.get("rrf_score", c.get("score", 0.0)), 6),
            retriever=c.get("retriever", "unknown"),
        )
        for c in fused_chunks
    ]

    return QueryResponse(
        query=request.query,
        answer=answer,
        sources=sources,
        num_chunks_used=len(fused_chunks),
        query_type=profile.query_type,
        needs_graph=profile.needs_graph,
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
