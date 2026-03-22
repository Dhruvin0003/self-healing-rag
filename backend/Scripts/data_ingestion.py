import os
import hashlib
import uuid
import sys
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ingest.parsers import load_file, SUPPORTED_EXTENSIONS
from ingest.chunkers import get_semantic_chunker, extract_chunks
from ingest.indexers import create_qdrant_collection, ingest_to_qdrant, ingest_to_graph

load_dotenv()

COLLECTION_NAME = "rag_documents"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")
BATCH_SIZE = 16

qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

model = SentenceTransformer(model_name_or_path=os.getenv("EMBEDDING_MODEL"),device="cuda" if os.getenv("USE_GPU") else "cpu")

semantic_chunker = get_semantic_chunker(model, batch_size=BATCH_SIZE)

def chunk_hash(text: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_OID, text))

def ingest_data():
    if not os.path.exists(DATA_DIR):
        print(f"Data directory missing at: {DATA_DIR}")
        return

    files = [f for f in os.listdir(DATA_DIR) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]

    for filename in tqdm(files, desc="Processing files"):
        path = os.path.join(DATA_DIR, filename)

        try:
            # 1. Parse File
            text = load_file(path)
            if not text:
                continue

            document_id = hashlib.md5(filename.encode()).hexdigest()

            # 2. Chunking
            chunk_texts = extract_chunks(semantic_chunker, text)
            if not chunk_texts:
                continue

            # 3. Embedding
            embeddings = model.encode(chunk_texts, batch_size=BATCH_SIZE, show_progress_bar=False)

            # 4. Ingest into Qdrant
            ingest_to_qdrant(
                client=qdrant_client,
                collection_name=COLLECTION_NAME,
                chunk_texts=chunk_texts,
                embeddings=embeddings,
                document_id=document_id,
                filename=filename,
                chunk_hash_fn=chunk_hash
            )

            # 5. Ingest into Neo4j
            ingest_to_graph(
                chunk_texts=chunk_texts,
                filename=filename,
                chunk_hash_fn=chunk_hash
            )

        except Exception as e:
            print(f"Error processing {filename} : {e}")

if __name__ == "__main__":
    create_qdrant_collection(qdrant_client, COLLECTION_NAME, model.get_sentence_embedding_dimension())
    ingest_data()
    print("Ingestion complete")