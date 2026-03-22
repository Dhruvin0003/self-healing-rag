"""
Neo4j driver singleton and schema bootstrap.

Call `setup_constraints()` once at application startup to ensure
uniqueness constraints exist for Entity, Concept, and Domain nodes.
"""

from contextlib import contextmanager
from typing import Generator
from neo4j import GraphDatabase, Session
import os

_driver = GraphDatabase.driver(
    os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
    auth=(
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "password")
    )
)

_CONSTRAINTS = [
    # Entity uniqueness
    """
    CREATE CONSTRAINT entity_unique IF NOT EXISTS
    FOR (e:Entity) REQUIRE e.name IS UNIQUE
    """,
    # Concept uniqueness
    """
    CREATE CONSTRAINT concept_unique IF NOT EXISTS
    FOR (c:Concept) REQUIRE c.name IS UNIQUE
    """,
    # Domain uniqueness
    """
    CREATE CONSTRAINT domain_unique IF NOT EXISTS
    FOR (d:Domain) REQUIRE d.name IS UNIQUE
    """
]

_FULLTEXT_INDEX = """
CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
FOR (e:Entity) ON EACH [e.name]
"""

@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a Neo4j session and close it afterwards."""
    session = _driver.session()
    try:
        yield session
    finally:
        session.close()

def setup_constraints() -> None:
    """Create uniqueness constraints and full-text index for Entity, Concept, and Domain nodes."""
    with get_session() as session:
        for cypher in _CONSTRAINTS:
            session.run(cypher)
        session.run(_FULLTEXT_INDEX)
