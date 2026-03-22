"""
Graph Retriever
===============
Retrieves a relevant sub-graph from Neo4j given a user query.

Flow:
  1. Use Gemini to extract key entity names from the query.
  2. Run a Cypher multi-hop query to find related nodes (Entity, Concept, Domain).
  3. Format results into two human-readable lists:
     - graph_triples  : "A -[REL]-> B"
     - concept_triples: "A belongs_to C (Concept) · C part_of D (Domain)"
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import google.generativeai as genai

import os
from app.graph.client import get_session

logger = logging.getLogger(__name__)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

# ---------------------------------------------------------------------------
# Entity extraction from query
# ---------------------------------------------------------------------------

_ENTITY_PROMPT = """\
Extract the key named entities from the question below.
Return ONLY a JSON array of entity name strings — no markdown, no explanation.
Limit to the 5 most important entities.

Question: {query}

Example output: ["Neo4j", "FastAPI", "Python"]
"""


def _extract_query_entities(query: str) -> list[str]:
    try:
        model = genai.GenerativeModel(model_name=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))
        response = model.generate_content(_ENTITY_PROMPT.format(query=query))
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        entities: list[str] = json.loads(raw)
        return [e for e in entities if isinstance(e, str)][:5]
    except Exception as exc:
        logger.warning("Query entity extraction failed: %s", exc)
        return []


# Cypher sub-graph retrieval
# Primary: full-text index search (tokenized Lucene — handles partial keyword matches)
_CYPHER_FULLTEXT = """
CALL db.index.fulltext.queryNodes('entity_fulltext', $lucene_query)
YIELD node AS e, score
WHERE score > 0.1

OPTIONAL MATCH (e)-[r]-(n:Entity)
OPTIONAL MATCH (e)-[:BELONGS_TO]->(c:Concept)
OPTIONAL MATCH (c)-[:PART_OF]->(d:Domain)

RETURN
    e.name        AS entity,
    e.type        AS entity_type,
    type(r)       AS rel_type,
    n.name        AS neighbor,
    c.name        AS concept,
    d.name        AS domain

ORDER BY score DESC
LIMIT 50
"""

# Fallback: CONTAINS-based search (catches cases where full-text index misses)
_CYPHER_CONTAINS = """
MATCH (e:Entity)
WHERE ANY(kw IN $entities WHERE toLower(e.name) CONTAINS toLower(kw))
   OR ANY(kw IN $entities WHERE toLower(kw) CONTAINS toLower(e.name))

OPTIONAL MATCH (e)-[r]-(n:Entity)
OPTIONAL MATCH (e)-[:BELONGS_TO]->(c:Concept)
OPTIONAL MATCH (c)-[:PART_OF]->(d:Domain)

RETURN
    e.name        AS entity,
    e.type        AS entity_type,
    type(r)       AS rel_type,
    n.name        AS neighbor,
    c.name        AS concept,
    d.name        AS domain

LIMIT 50
"""


def _build_lucene_query(entities: list[str]) -> str:
    """
    Build a Lucene OR query from extracted entity keywords.
    Each entity is split into individual tokens and OR-joined so that
    e.g. ["MSME", "food processing"] becomes "MSME OR food OR processing".
    Special Lucene characters are escaped.
    """
    _LUCENE_SPECIAL = r'[+\-!(){}\[\]^"~*?:\\/]'
    tokens: set[str] = set()
    for entity in entities:
        for token in entity.split():
            clean = re.sub(_LUCENE_SPECIAL, " ", token).strip()
            if clean and len(clean) > 2:
                tokens.add(clean)
    return " OR ".join(tokens) if tokens else ""


def _run_graph_query(entities: list[str]) -> list[dict[str, Any]]:
    """Two-stage retrieval: full-text index first, CONTAINS fallback."""
    if not entities:
        return []

    lucene_query = _build_lucene_query(entities)
    logger.info("Full-text Lucene query: %s", lucene_query)

    with get_session() as session:
        # Stage 1 — full-text index search
        if lucene_query:
            try:
                result = session.run(_CYPHER_FULLTEXT, lucene_query=lucene_query)
                records = [dict(record) for record in result]
                if records:
                    logger.info("Full-text search returned %d records", len(records))
                    return records
                logger.info("Full-text search returned 0 records, falling back to CONTAINS")
            except Exception as exc:
                logger.warning("Full-text search failed (index may not exist yet): %s", exc)

        # Stage 2 — CONTAINS fallback
        result = session.run(_CYPHER_CONTAINS, entities=entities)
        records = [dict(record) for record in result]
        logger.info("CONTAINS fallback returned %d records", len(records))
        return records


# Result formatter
def _format_results(
    records: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """
    Returns
    -------
    graph_triples   : ["Neo4j -[STORES]-> Graph Data", ...]
    concept_triples : ["Neo4j belongs_to Graph Database · Graph Database part_of Databases", ...]
    """
    graph_set: set[str] = set()
    concept_set: set[str] = set()

    for rec in records:
        entity = rec.get("entity") or ""
        rel_type = rec.get("rel_type") or ""
        neighbor = rec.get("neighbor") or ""
        concept = rec.get("concept") or ""
        domain = rec.get("domain") or ""

        if entity and rel_type and neighbor:
            graph_set.add(f"{entity} -[{rel_type}]-> {neighbor}")

        if entity and concept:
            line = f"{entity} belongs_to {concept} (Concept)"
            if domain:
                line += f" · {concept} part_of {domain} (Domain)"
            concept_set.add(line)

    return sorted(graph_set), sorted(concept_set)


# Public API
def search(query: str) -> dict[str, list[str]]:
    """
    Retrieve graph knowledge relevant to *query*.

    Returns
    -------
    dict with:
        "graph_triples"   : list of "A -[REL]-> B" strings
        "concept_triples" : list of concept/domain hierarchy strings
    """
    entities = _extract_query_entities(query)
    logger.info("Graph retriever extracted query entities: %s", entities)

    if not entities:
        logger.warning("No entities extracted from query — graph context will be empty.")
        return {"graph_triples": [], "concept_triples": []}

    records = _run_graph_query(entities)
    logger.info("Graph query returned %d records for entities %s", len(records), entities)
    graph_triples, concept_triples = _format_results(records)

    return {
        "graph_triples": graph_triples,
        "concept_triples": concept_triples,
    }
