"""
Evaluation API routes — Phase 4.

Endpoints
---------
POST /api/v1/evaluations/run
    Trigger a batch evaluation run over the Golden Dataset.
    Runs the full pipeline for each sample and stores scores in Postgres.

GET /api/v1/evaluations
    Return stored evaluation metrics from Postgres (most recent first).

GET /api/v1/evaluations/{trace_id}
    Return full trace + metrics for a specific query trace.
"""

import asyncio
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, desc

from app.db.database import get_db
from app.db.models import EvaluationMetric, QueryTrace
from app.evaluation.runner import run_evaluation

router = APIRouter(prefix="/evaluations", tags=["Evaluation"])

# Pydantic response schemas
class MetricOut(BaseModel):
    trace_id: UUID
    query: str
    answer: Optional[str]
    ideal_answer: Optional[str]
    answer_similarity: Optional[float]
    context_recall: Optional[float]
    context_precision: Optional[float]
    evaluated_at: str

    class Config:
        from_attributes = True


class RunResponse(BaseModel):
    status: str
    message: str


# Background task
def _run_evaluation_bg():
    """Kick off the async runner inside a background thread."""
    asyncio.run(run_evaluation())


# Routes
@router.post("/run", response_model=RunResponse, summary="Run Golden Dataset Evaluation")
async def run_evaluation_endpoint(background_tasks: BackgroundTasks):
    """
    Trigger a full batch evaluation against the Golden Dataset.

    The evaluation runs in the background — returns immediately.
    Check `GET /evaluations` to see results once complete.
    """
    background_tasks.add_task(_run_evaluation_bg)
    return RunResponse(
        status="accepted",
        message="Evaluation started in the background. Check GET /evaluations for results."
    )


@router.get("", response_model=List[MetricOut], summary="List Evaluation Results")
async def list_evaluations(limit: int = 50, offset: int = 0):
    """Return stored RAGAS evaluation metrics."""
    async with get_db() as db:
        stmt = (
            select(EvaluationMetric, QueryTrace)
            .join(QueryTrace, EvaluationMetric.trace_id == QueryTrace.id)
            .order_by(desc(EvaluationMetric.evaluated_at))
            .limit(limit)
            .offset(offset)
        )
        rows = (await db.execute(stmt)).all()

    results = []
    for metric, trace in rows:
        results.append(
            MetricOut(
                trace_id=metric.trace_id,
                query=trace.query,
                answer=trace.answer,
                ideal_answer=metric.ideal_answer,
                answer_similarity=metric.answer_similarity,
                context_recall=metric.context_recall,
                context_precision=metric.context_precision,
                evaluated_at=metric.evaluated_at.isoformat(),
            )
        )
    return results


@router.get("/{trace_id}", response_model=MetricOut, summary="Get Evaluation by Trace ID")
async def get_evaluation(trace_id: UUID):
    """Return a specific query trace and its evaluation scores."""
    async with get_db() as db:
        stmt = (
            select(EvaluationMetric, QueryTrace)
            .join(QueryTrace, EvaluationMetric.trace_id == QueryTrace.id)
            .where(EvaluationMetric.trace_id == trace_id)
        )
        row = (await db.execute(stmt)).first()

    if not row:
        raise HTTPException(status_code=404, detail=f"No evaluation found for trace_id={trace_id}")

    metric, trace = row
    return MetricOut(
        trace_id=metric.trace_id,
        query=trace.query,
        answer=trace.answer,
        ideal_answer=metric.ideal_answer,
        answer_similarity=metric.answer_similarity,
        context_recall=metric.context_recall,
        context_precision=metric.context_precision,
        evaluated_at=metric.evaluated_at.isoformat(),
    )
