"""
SQLAlchemy ORM models for Phase 4 — Evaluation & Metrics Storage.

Tables
------
query_traces       — Every request that passes through the RAG pipeline.
evaluation_metrics — RAGAS scores linked to a query trace.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.database import Base


def now() -> datetime:
    return datetime.now(timezone.utc)


class QueryTrace(Base):
    """Records every RAG query with the retrieved context and generated answer."""

    __tablename__ = "query_traces"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query = Column(Text, nullable=False)
    retrieved_contexts = Column(Text, nullable=True)
    answer = Column(Text, nullable=True)
    query_type = Column(String(50), nullable=True)
    needs_graph = Column(String(5), nullable=True)
    num_chunks_used = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=now, nullable=False)
    metrics = relationship("EvaluationMetric", back_populates="trace", uselist=False, cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<QueryTrace id={self.id} query={self.query[:40]!r}>"


class EvaluationMetric(Base):
    """Stores RAGAS non-LLM evaluation scores for a given query trace."""

    __tablename__ = "evaluation_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id = Column(UUID(as_uuid=True), ForeignKey("query_traces.id", ondelete="CASCADE"), nullable=False, unique=True)
    answer_similarity = Column(Float, nullable=True)    # cosine sim to ideal answer
    context_recall    = Column(Float, nullable=True)    # ROUGE-L coverage
    context_precision = Column(Float, nullable=True)    # embedding relevance
    ideal_answer = Column(Text, nullable=True)
    evaluated_at = Column(DateTime(timezone=True), default=now, nullable=False)
    trace = relationship("QueryTrace", back_populates="metrics")

    def __repr__(self) -> str:
        return (
            f"<EvaluationMetric trace_id={self.trace_id} "
            f"similarity={self.answer_similarity:.3f}>"
        )
