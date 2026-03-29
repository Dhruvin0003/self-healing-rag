"""
Evaluation Runner — Phase 4.

Loads the Golden Dataset from `backend/Data/golden_dataset.json`,
runs each entry through the live RAG pipeline,
and persists query traces + evaluation scores to PostgreSQL.

Usage (from backend/ directory):
    python -m app.evaluation.runner
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dotenv import load_dotenv
load_dotenv()

from app.db.database import get_db, init_db
from app.db.models import EvaluationMetric, QueryTrace
from app.evaluation.engine import evaluate_sample

from app.retrieval import query_intelligence, router as retriever_router, fusion
from app.generation import controller


GOLDEN_DATASET_PATH = Path(__file__).resolve().parents[3] / "Data" / "golden_dataset.json"
CHUNK_SEPARATOR = " ||| "


def _load_golden_dataset() -> list[dict]:
    if not GOLDEN_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Golden Dataset not found at {GOLDEN_DATASET_PATH}. "
            "Please create backend/Data/golden_dataset.json."
        )
    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("golden_dataset.json must be a non-empty JSON array.")
    return data


async def _run_pipeline(query: str, top_k: int = 5) -> dict:
    """Run the full RAG pipeline and return a result dict."""
    profile = query_intelligence.classify(query)
    route_result = retriever_router.route(query=query, profile=profile, top_k=top_k)
    fused_chunks = fusion.fuse(route_result["dense"], route_result["bm25"], top_k=top_k)
    graph_triples = route_result.get("graph_triples", [])
    concept_triples = route_result.get("concept_triples", [])

    answer = controller.generate(query=query, chunks=fused_chunks, graph_triples=graph_triples, concept_triples=concept_triples)

    context_texts = [c.get("text", "") for c in fused_chunks]

    return {
        "answer": answer,
        "contexts": context_texts,
        "query_type": profile.query_type,
        "needs_graph": str(profile.needs_graph).lower(),
        "num_chunks": len(fused_chunks),
    }


async def run_evaluation():
    """Main entry-point: run the full golden-dataset evaluation."""
    print("Initialising database tables...")
    await init_db()

    dataset = _load_golden_dataset()
    print(f"Loaded {len(dataset)} golden samples.")

    for i, sample in enumerate(dataset, start=1):
        query = sample.get("query", "").strip()
        ideal_answer = sample.get("ideal_answer", "").strip()

        if not query or not ideal_answer:
            print(f"[{i}/{len(dataset)}] Skipping — missing query or ideal_answer.")
            continue

        print(f"[{i}/{len(dataset)}] Running pipeline for: {query[:60]!r}")

        try:
            result = await _run_pipeline(query)
        except Exception as exc:
            print(f"  ⚠  Pipeline error: {exc}")
            continue

        # Compute RAGAS scores
        try:
            scores = evaluate_sample(
                query=query,
                retrieved_contexts=result["contexts"],
                answer=result["answer"],
                ideal_answer=ideal_answer,
            )
        except Exception as exc:
            print(f"Evaluation error: {exc}")
            scores = {"answer_similarity": None, "context_recall": None, "context_precision": None}

        # Persist to Postgres
        async with get_db() as db:
            trace = QueryTrace(
                query=query,
                retrieved_contexts=CHUNK_SEPARATOR.join(result["contexts"]),
                answer=result["answer"],
                query_type=result["query_type"],
                needs_graph=result["needs_graph"],
                num_chunks_used=result["num_chunks"],
            )
            db.add(trace)
            await db.flush()  # get the generated ID

            metric = EvaluationMetric(
                trace_id=trace.id,
                ideal_answer=ideal_answer,
                **scores,
            )
            db.add(metric)

        print(
            f"similarity={scores['answer_similarity']}, "
            f"recall={scores['context_recall']}, "
            f"precision={scores['context_precision']}"
        )

    print("Evaluation complete.")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
