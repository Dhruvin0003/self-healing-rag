"""
Dispatches retrieval calls to the right retrievers based on a QueryProfile.

Rules:
  - Dense  : always invoked
  - BM25   : always invoked
  - Graph  : only invoked when QueryProfile.needs_graph is True

Returns a RouterResult dict with raw results from each retriever, ready to
be passed to fusion.fuse().
"""

import logging
from typing import TypedDict
from app.retrieval import dense, bm25
from app.retrieval.query_intelligence import QueryProfile
from app.graph import retriever as graph_retriever

logger = logging.getLogger(__name__)

# Return type
class RouterResult(TypedDict):
    """Raw retrieval results from each retriever."""

    dense: list[dict]
    bm25: list[dict]
    graph_triples: list[str]
    concept_triples: list[str]
    graph_used: bool

# Public API
def route(query: str, profile: QueryProfile, top_k: int = 5) -> RouterResult:
    """
    Execute retrieval based on *profile* and return a :class:`RouterResult`.

    Parameters
    ----------
    query:
        The original user question.
    profile:
        The :class:`QueryProfile` produced by :mod:`query_intelligence`.
    top_k:
        Number of results to fetch from each vector/keyword retriever.

    Returns
    -------
    RouterResult
        Dict containing raw lists from each retriever.
    """
    # Dense retrieval (always)
    dense_results = dense.search(query, top_k=top_k)

    # BM25 retrieval (always)
    bm25_results = bm25.search(query, top_k=top_k)

    # Graph retrieval (conditional)
    graph_triples: list[str] = []
    concept_triples: list[str] = []
    graph_used = False

    if profile.needs_graph:
        try:
            graph_result = graph_retriever.search(query)
            graph_triples = graph_result.get("graph_triples", [])
            concept_triples = graph_result.get("concept_triples", [])
            graph_used = True

        except Exception as exc:
            logger.warning("Router: Graph retrieval skipped due to error: %s", exc)
    else:
        logger.info("Router: query_type=%r, needs_graph=False → skipping Graph retriever",profile.query_type,)

    return RouterResult(
        dense=dense_results,
        bm25=bm25_results,
        graph_triples=graph_triples,
        concept_triples=concept_triples,
        graph_used=graph_used,
    )
