from typing import List
from sentence_transformers import SentenceTransformer
from langchain_experimental.text_splitter import SemanticChunker

class LangChainEmbeddingsWrapper:
    def __init__(self, model: SentenceTransformer, batch_size: int = 16):
        self.model = model
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

def get_semantic_chunker(model: SentenceTransformer, batch_size: int = 16) -> SemanticChunker:
    embeddings_wrapper = LangChainEmbeddingsWrapper(model, batch_size)
    
    return SemanticChunker(embeddings_wrapper, breakpoint_threshold_type="percentile", min_chunk_size=300)

def extract_chunks(semantic_chunker: SemanticChunker, text: str | list) -> list[str]:
    """Given raw text or a list of pages, returns a list of chunk strings."""
    chunk_texts = []
    if isinstance(text, list):
        for page in text:
            docs = semantic_chunker.create_documents([page])
            chunk_texts.extend([doc.page_content for doc in docs if len(doc.page_content) > 50])
    else:
        docs = semantic_chunker.create_documents([text])
        chunk_texts = [doc.page_content for doc in docs if len(doc.page_content) > 50]
    return chunk_texts
