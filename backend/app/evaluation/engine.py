"""
RAGAS Evaluation Engine — Phase 4.

Uses ONLY non-LLM metrics so no API calls are made during evaluation:
  - AnswerSimilarity  : cosine similarity via SentenceTransformers
  - ContextRecall     : ROUGE-L coverage of ideal answer in retrieved chunks
  - ContextPrecision  : embedding similarity of chunks to ideal answer

Usage:
    from app.evaluation.engine import evaluate_sample
    scores = evaluate_sample(
        query="What is probation?",
        retrieved_contexts=["Probation is a 6-month period..."],
        answer="Probation lasts 6 months.",
        ideal_answer="The probation period is 6 months for all new hires.",
    )
    # scores = {"answer_similarity": 0.94, "context_recall": 0.80, "context_precision": 0.76}
"""

from __future__ import annotations

import os
from typing import List

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_similarity, context_recall, context_precision
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L12-v2")

ragas_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

answer_similarity.embeddings = ragas_embeddings
context_precision.embeddings = ragas_embeddings

METRICS = [answer_similarity, context_recall, context_precision]


def evaluate_sample(query: str, retrieved_contexts: List[str], answer: str, ideal_answer: str,) -> dict:
    """
    Score a single RAG response against an ideal answer.

    Parameters
    ----------
    query               : The original user question.
    retrieved_contexts  : List of text chunks returned by the retriever.
    answer              : The generated answer from the LLM.
    ideal_answer        : Ground-truth answer from the Golden Dataset.

    Returns
    -------
    dict with keys: answer_similarity, context_recall, context_precision.
    All values are floats in [0, 1]. Missing scores are None.
    """
    dataset = Dataset.from_dict({"question": [query],"answer": [answer],"contexts": [retrieved_contexts],"ground_truth": [ideal_answer]})

    result = evaluate(dataset, metrics=METRICS)
    result_df = result.to_pandas()
    row = result_df.iloc[0]

    return {
        "answer_similarity": _safe_float(row.get("answer_similarity")),
        "context_recall": _safe_float(row.get("context_recall")),
        "context_precision": _safe_float(row.get("context_precision")),
    }


def _safe_float(value) -> "float | None":
    """Return a Python float or None if the value is NaN / missing."""
    try:
        v = float(value)
        import math
        if math.isnan(v):
            return None
        return float(round(v, 6))
    except (TypeError, ValueError):
        return None
