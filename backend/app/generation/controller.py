"""
Generation Controller — builds a grounded prompt and calls Google Gemini.
"""

from typing import List
import os

import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY")

_SYSTEM_PROMPT = """You are a precise, trustworthy question-answering assistant.
Your ONLY source of information is the context provided below.
If the context does not contain enough information to answer, respond with:
"I don't have enough information in the provided context to answer this question."
Do NOT use any external knowledge or make up facts."""

_USER_TEMPLATE = """Context:
{context}

{graph_knowledge}

{concept_hierarchy}

Question: {question}

Answer:"""


def _build_context(chunks: List[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source_file", "unknown")
        text = chunk.get("text", "").strip()
        parts.append(f"[{i}] (source: {source})\n{text}")
    return "\n\n---\n\n".join(parts)


def generate(query: str,chunks: List[dict],graph_triples: List[str] = None,concept_triples: List[str] = None,) -> str:
    """
    Generate a grounded answer from the provided chunks and graph context using Gemini.

    Args:
        query:  The original user question.
        chunks: Fused list of retrieved chunks (from dense + BM25).
        graph_triples: List of extracted Entity-Relation-Entity triples.
        concept_triples: List of extracted Entity-Concept-Domain hierarchy strings.

    Returns:
        The generated answer as a plain string.
    """
    if not chunks and not graph_triples:
        return "No relevant context was retrieved for your query."

    context = _build_context(chunks) if chunks else "No document chunks retrieved."
    
    graph_knowledge = ""
    if graph_triples:
        graph_knowledge = "GRAPH KNOWLEDGE:\n" + "\n".join(graph_triples)
        
    concept_hierarchy = ""
    if concept_triples:
        concept_hierarchy = "CONCEPT HIERARCHY:\n" + "\n".join(concept_triples)

    user_message = _USER_TEMPLATE.format(
        context=context,
        graph_knowledge=graph_knowledge,
        concept_hierarchy=concept_hierarchy,
        question=query
    )

    model = genai.GenerativeModel(
        model_name=os.environ.get("GEMINI_MODEL"),
        system_instruction=_SYSTEM_PROMPT,
    )

    response = model.generate_content(user_message)
    return response.text.strip()
