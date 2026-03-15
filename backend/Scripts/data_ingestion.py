import os
import hashlib
import uuid
import json
import datetime
from typing import List
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from langchain_experimental.text_splitter import SemanticChunker
import fitz
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat

load_dotenv()

COLLECTION_NAME = "rag_documents"
DATA_DIR = os.path.join(os.path.dirname(__file__),"..","Data")
BATCH_SIZE = 16
SUPPORTED_EXTENSIONS = [".txt",".md",".pdf",".docx",".json"]

qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"),api_key=os.getenv("QDRANT_API_KEY"))

model = SentenceTransformer(model_name_or_path=os.getenv("EMBEDDING_MODEL"),device="cuda" if os.getenv("USE_GPU") else "cpu")

doc_converter = DocumentConverter()

class LangChainEmbeddingsWrapper:
    def embed_documents(self,texts: List[str]) -> List[List[float]]:
        return model.encode(texts,batch_size=BATCH_SIZE,show_progress_bar=True).tolist()

    def embed_query(self,text: str) -> List[float]:
        return model.encode([text])[0].tolist()

embeddings_wrapper = LangChainEmbeddingsWrapper()

semantic_chunker = SemanticChunker(embeddings_wrapper,breakpoint_threshold_type="percentile",min_chunk_size=300)

def create_collection():
    if qdrant_client.collection_exists(COLLECTION_NAME):
        print("Collection exists")
        return

    print("Creating collection...")
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=model.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE
        )
    )
    print("Collection created")

def load_txt(path):
    with open(path,"r",encoding="utf-8") as f:
        return f.read()

def load_pdf(path):
    doc = fitz.open(path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


def load_docx(path):
    result = doc_converter.convert(path)
    return result.document.export_to_markdown()

def load_json(path):
    with open(path) as f:
        data = json.load(f)

    return json.dumps(data,indent=2)

def load_file(path):
    ext = os.path.splitext(path)[1].lower()

    if ext in (".txt", ".md"):
        return load_txt(path)
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".docx":
        return load_docx(path)
    if ext == ".json":
        return load_json(path)

    return None

def chunk_hash(text: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_OID, text))

def ingest_data():
    if not os.path.exists(DATA_DIR):
        print("Data directory missing")
        return

    files = [f for f in os.listdir(DATA_DIR) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]

    for filename in tqdm(files,desc="Processing files"):
        path = os.path.join(DATA_DIR,filename)

        try:
            text = load_file(path)
            if not text:
                continue

            document_id = hashlib.md5(filename.encode()).hexdigest()

            chunk_texts = []

            if isinstance(text, list):
                for page in text:
                    docs = semantic_chunker.create_documents([page])
                    chunk_texts.extend(
                        [doc.page_content for doc in docs if len(doc.page_content) > 50]
                    )
            else:
                docs = semantic_chunker.create_documents([text])
                chunk_texts = [doc.page_content for doc in docs if len(doc.page_content) > 50]

            if not chunk_texts:
                continue

            embeddings = model.encode(chunk_texts,batch_size=BATCH_SIZE,show_progress_bar=False)

            points = []
            for i, (chunk, vector) in enumerate(zip(chunk_texts, embeddings)):
                chunk_id = chunk_hash(chunk)

                payload = {
                    "text": chunk,
                    "source_file": filename,
                    "document_id": document_id,
                    "chunk_index": i,
                    "chunk_size": len(chunk),
                    "ingested_at": str(datetime.datetime.utcnow())
                }

                points.append(
                    models.PointStruct(
                        id=chunk_id,
                        vector=vector.tolist(),
                        payload=payload
                    )
                )

            if points:
                qdrant_client.upsert(collection_name=COLLECTION_NAME,points=points)

        except Exception as e:
            print(f"Error processing {filename} : {e}")

if __name__ == "__main__":
    create_collection()
    ingest_data()
    print("Ingestion complete")