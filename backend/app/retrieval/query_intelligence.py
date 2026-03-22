"""
Classifies an incoming user query into a QueryProfile.

QueryProfile fields:
    query_type  : "factual" | "relational" | "analytical"
    needs_graph : bool  — True when multi-hop graph traversal is needed
    entities    : list[str] — key entities spotted in the query

This is the entry-point for intelligent routing. The QueryProfile
is consumed by router.py to decide which retrievers to invoke.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Literal
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

QueryType = Literal["factual", "relational", "analytical"]

@dataclass
class QueryProfile:
    """Structured classification result for a single user query."""

    query_type: QueryType = "factual"
    needs_graph: bool = False
    entities: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"QueryProfile(type={self.query_type!r}, "
            f"needs_graph={self.needs_graph}, "
            f"entities={self.entities})"
        )


# Classification prompt
CLASSIFY_PROMPT = """\
You are a query classification expert for a Retrieval-Augmented Generation system.
Analyse the user question below and return a single JSON object — no markdown, no explanation.

Definitions:
  factual    — asks for a specific fact, definition, or number. Graph reasoning NOT needed.
  relational — asks how things relate, compare, depend-on, or flow between entities. Graph reasoning IS needed.
  analytical — asks for summaries, overviews, pros/cons, or complex explanations. Graph MAY help.

Rules for needs_graph:
  - true  when the query involves relationships, dependencies, flows, or multi-hop paths between entities
  - false when the query is a simple look-up or definition

Few-shot examples:
Q: "What is BM25?"
{{"query_type": "factual", "needs_graph": false, "entities": ["BM25"]}}

Q: "How does FastAPI depend on Uvicorn and Starlette?"
{{"query_type": "relational", "needs_graph": true, "entities": ["FastAPI", "Uvicorn", "Starlette"]}}

Q: "Summarise the advantages of vector databases over traditional databases."
{{"query_type": "analytical", "needs_graph": false, "entities": ["vector databases", "traditional databases"]}}

Q: "What are the relationships between MSME schemes and the Ministry of Finance?"
{{"query_type": "relational", "needs_graph": true, "entities": ["MSME schemes", "Ministry of Finance"]}}

Now classify:
Q: "{query}"
"""

def classify(query: str) -> QueryProfile:
    """
    Classify *query* and return a :class:`QueryProfile`.

    Falls back to a safe default (factual, no graph) on any error so the
    pipeline always continues even if the LLM call fails.
    """
    try:
        model = genai.GenerativeModel(model_name=os.environ.get("GEMINI_MODEL"))
        response = model.generate_content(CLASSIFY_PROMPT.format(query=query))
        raw = response.text.strip()

        # Strip accidental markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)

        parsed: dict = json.loads(raw)

        query_type: QueryType = parsed.get("query_type", "factual")
        if query_type not in ("factual", "relational", "analytical"):
            query_type = "factual"

        entities: list[str] = [e for e in parsed.get("entities", []) if isinstance(e, str)][:5]

        profile = QueryProfile(query_type=query_type,needs_graph=bool(parsed.get("needs_graph", False)),entities=entities,)
        
        return profile

    except Exception as exc:
        return QueryProfile(query_type="factual", needs_graph=False, entities=[])
