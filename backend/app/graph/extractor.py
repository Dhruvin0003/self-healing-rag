"""
Entity & Relationship Extractor
================================
Uses LLM to extract a structured knowledge graph from a text chunk.

Output schema:
{
  "entities": [
    {"name": "Neo4j", "type": "Technology", "concept": "Graph Database", "domain": "Databases", "source_chunk_id": "chunk_123"}
  ],
  "relations": [
    {"subject": "Neo4j", "predicate": "STORES", "object": "Graph Data"}
  ]
}
"""

from __future__ import annotations

import json
import re
import logging
from typing import Any

import google.generativeai as genai

import os

logger = logging.getLogger(__name__)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))


# Noise-filter lists
_GENERIC_ENTITIES: set[str] = {
    "system", "data", "process", "information", "entity", "thing",
    "item", "element", "object", "concept", "structure", "component",
    "module", "function", "value", "type", "example", "result", "output",
    "input", "user", "service", "server", "client", "model", "instance",
}

_WEAK_PREDICATES: set[str] = {
    "related_to", "related to", "associated_with", "associated with",
    "connected_to", "connected to", "has", "have", "is", "are",
}


# Extraction prompt
_EXTRACT_PROMPT = """\
You are a knowledge graph extraction expert. Analyze the text chunks below and extract entities and relationships to build a structured knowledge graph.

{chunks_text}

OUTPUT RULES:
1. Return ONLY valid JSON — no markdown fences, no extra text.
2. Extract at most {max_entities} entities and {max_relations} relations in total.
3. For each entity provide:
   - "name": canonical name (proper noun or technical term, e.g. "Apache Kafka")
   - "type": fine-grained type (e.g. "Technology", "Person", "Organization", "Algorithm", "Concept", "Location")
   - "concept": the higher-level category this entity belongs to (e.g. "Message Broker", "Machine Learning Model")
   - "domain": the top-level domain of the concept (e.g. "Distributed Systems", "Artificial Intelligence")
   - "source_chunk_id": the ID of the chunk where this entity was found.
4. For each relation provide:
   - "subject": entity name
   - "predicate": SHORT verb phrase in UPPER_SNAKE_CASE (e.g. "STORES", "RUNS_ON", "DEVELOPED_BY")
   - "object": entity name
5. The concept MUST be a higher abstraction than the entity.
6. The domain MUST be a higher abstraction than the concept.
7. Skip entities shorter than 3 characters.
8. Skip generic words like "system", "data", "process", "information".
9. Skip weak predicates like "related_to" or "associated_with".

Required JSON format:
{{
  "entities": [
    {{"name": "...", "type": "...", "concept": "...", "domain": "...", "source_chunk_id": "..."}}
  ],
  "relations": [
    {{"subject": "...", "predicate": "...", "object": "..."}}
  ]
}}
"""


def extract(chunks: list[dict[str, str]]) -> dict[str, Any]:
    """
    Extract entities and relations from a list of chunks using Gemini.

    Parameters
    ----------
    chunks : list of dicts with 'text' and 'chunk_id' keys

    Returns
    -------
    dict with keys ``entities`` and ``relations`` (both lists).
    Falls back to empty lists on any error.
    """
    
    chunks_text = "\n\n".join([f"--- CHUNK ID: {c.get('chunk_id', 'unknown')} ---\n{c.get('text', '')}" for c in chunks])
    
    prompt = _EXTRACT_PROMPT.format(
        chunks_text=chunks_text,
        max_entities=int(os.environ.get("MAX_ENTITIES_PER_CHUNK", 15)) * len(chunks),
        max_relations=int(os.environ.get("MAX_RELATIONS_PER_CHUNK", 25)) * len(chunks),
    )

    try:
        model = genai.GenerativeModel(model_name=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))
        response = model.generate_content(prompt)
        raw = response.text.strip()

        # Strip accidental markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)

        parsed: dict[str, Any] = json.loads(raw)
    except Exception as exc:
        logger.error("Extraction failed for batch: %s", exc)
        return {"entities": [], "relations": []}

    entities = _filter_entities(parsed.get("entities", []))
    relations = _filter_relations(parsed.get("relations", []))

    return {"entities": entities, "relations": relations}


# Filtering helpers

def _normalize_name(name: str) -> str:
    """Strip whitespace; canonical casing kept as-is for storage."""
    return name.strip()


def _filter_entities(entities: list[dict]) -> list[dict]:
    seen: set[str] = set()
    filtered: list[dict] = []

    for e in entities:
        name = _normalize_name(e.get("name", ""))
        lower = name.lower()
        chunk_id = e.get("source_chunk_id", "")
        dedup_key = f"{lower}::{chunk_id}"

        if len(name) < 3:
            continue
        if lower in _GENERIC_ENTITIES:
            continue
        if dedup_key in seen:
            continue

        seen.add(dedup_key)
        filtered.append({
            "name": name,
            "type": e.get("type", "Unknown"),
            "concept": e.get("concept", "General"),
            "domain": e.get("domain", "General"),
            "source_chunk_id": e.get("source_chunk_id", ""),
        })

    return filtered


def _filter_relations(relations: list[dict]) -> list[dict]:
    filtered: list[dict] = []

    for r in relations:
        predicate = r.get("predicate", "").strip().lower()
        if predicate in _WEAK_PREDICATES:
            continue
        subject = _normalize_name(r.get("subject", ""))
        obj = _normalize_name(r.get("object", ""))
        if not subject or not obj:
            continue

        filtered.append({
            "subject": subject,
            "predicate": r.get("predicate", "").strip(),
            "object": obj,
        })

    return filtered
