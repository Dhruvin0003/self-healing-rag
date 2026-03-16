"""
Generation Controller — builds a grounded prompt and calls Google Gemini.
"""

from typing import List

import google.generativeai as genai

from app.core.config import settings

genai.configure(api_key=settings.gemini_api_key)

_SYSTEM_PROMPT = """You are a precise, trustworthy question-answering assistant.
Your ONLY source of information is the context provided below.
If the context does not contain enough information to answer, respond with:
"I don't have enough information in the provided context to answer this question."
Do NOT use any external knowledge or make up facts."""

_USER_TEMPLATE = """Context:
{context}

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


def generate(query: str, chunks: List[dict]) -> str:
    """
    Generate a grounded answer from the provided chunks using Gemini.

    Args:
        query:  The original user question.
        chunks: Fused list of retrieved chunks (from dense + BM25).

    Returns:
        The generated answer as a plain string.
    """
    if not chunks:
        return "No relevant context was retrieved for your query."

    context = _build_context(chunks)
    user_message = _USER_TEMPLATE.format(context=context, question=query)

    model = genai.GenerativeModel(
        model_name=settings.gemini_model,
        system_instruction=_SYSTEM_PROMPT,
    )

    response = model.generate_content(user_message)
    return response.text.strip()
