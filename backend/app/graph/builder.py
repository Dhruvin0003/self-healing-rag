"""
Graph Builder
=============
Upserts the extracted knowledge graph into Neo4j.

Schema created:
  (:Entity  {name, type, created_at})
  (:Concept {name})
  (:Domain  {name})

  (Entity)-[:BELONGS_TO]->(Concept)
  (Concept)-[:PART_OF]->(Domain)
  (Entity)-[:<PREDICATE>]->(Entity)
  (Entity)-[:SOURCE]->(Chunk)       where Chunk is a virtual node keyed by chunk_id
  (:Chunk {id})
"""

from __future__ import annotations

import re
import logging
from datetime import datetime, timezone
from typing import Any

from app.graph.client import get_session

logger = logging.getLogger(__name__)


def _sanitize_predicate(raw: str) -> str:
    """
    Convert a raw predicate string into a valid Neo4j relationship type.

    Examples:
        "runs on"  → RUNS_ON
        "STORES"   → STORES
        "part-of"  → PART_OF
    """
    upper = raw.upper()
    # Replace any non-alphanumeric character with underscore
    sanitized = re.sub(r"[^A-Z0-9]+", "_", upper)
    # Strip leading/trailing underscores
    sanitized = sanitized.strip("_")
    return sanitized or "RELATION"


def build(extraction: dict[str, Any], chunks: list[dict[str, str]] = None) -> None:
    """
    Upsert a full extraction result into Neo4j.

    Parameters
    ----------
    extraction : dict returned by ``extractor.extract()``
        Keys: ``entities``, ``relations``
    chunks : list of dicts with 'text' and 'chunk_id' keys
        The source chunks that generated this extraction.
    """
    entities: list[dict] = extraction.get("entities", [])
    relations: list[dict] = extraction.get("relations", [])

    if not entities and not relations and not chunks:
        return

    now = datetime.now(timezone.utc).isoformat()

    with get_session() as session:
        # Upsert Chunk node (for traceability)
        if chunks:
            for c in chunks:
                chunk_id = c.get("chunk_id")
                if chunk_id:
                    session.run("MERGE (:Chunk {id: $id})", id=chunk_id)

        # Upsert Entity → Concept → Domain hierarchy
        for e in entities:
            name = e["name"]
            etype = e.get("type", "Unknown")
            concept = e.get("concept", "General")
            domain = e.get("domain", "General")

            session.run(
                """
                MERGE (d:Domain {name: $domain})
                MERGE (c:Concept {name: $concept})
                MERGE (c)-[:PART_OF]->(d)
                MERGE (e:Entity {name: $name})
                  ON CREATE SET e.type = $type, e.created_at = $created_at
                  ON MATCH  SET e.type = $type
                MERGE (e)-[:BELONGS_TO]->(c)
                """,
                domain=domain,
                concept=concept,
                name=name,
                type=etype,
                created_at=now,
            )

            # Traceability: Entity sourced from Chunk
            source_chunk_id = e.get("source_chunk_id")
            if source_chunk_id:
                session.run(
                    """
                    MATCH (e:Entity {name: $name})
                    MATCH (ch:Chunk  {id: $chunk_id})
                    MERGE (e)-[:SOURCE]->(ch)
                    """,
                    name=name,
                    chunk_id=source_chunk_id,
                )


        # Upsert dynamic typed Entity→Entity relationships
        for r in relations:
            subject = r["subject"]
            predicate = _sanitize_predicate(r["predicate"])
            obj = r["object"]

            # Ensure both endpoint entities exist (may not have been in the
            # entities list due to noise-filtering or LLM inconsistency)
            session.run(
                "MERGE (:Entity {name: $name})", name=subject
            )
            session.run(
                "MERGE (:Entity {name: $name})", name=obj
            )

            # Dynamic relationship type via f-string in Cypher
            session.run(
                f"""
                MATCH (a:Entity {{name: $subject}})
                MATCH (b:Entity {{name: $object}})
                MERGE (a)-[:`{predicate}`]->(b)
                """,
                subject=subject,
                object=obj,
            )
