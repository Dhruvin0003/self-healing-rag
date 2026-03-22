import datetime
import requests
import time
import os
import json
from typing import List

from qdrant_client import QdrantClient, models

def create_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    if client.collection_exists(collection_name):
        print("Qdrant collection exists")
        return

    print("Creating Qdrant collection...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        )
    )
    print("Qdrant collection created")


def ingest_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    chunk_texts: List[str],
    embeddings,
    document_id: str,
    filename: str,
    chunk_hash_fn
):
    points = []
    for i, (chunk, vector) in enumerate(zip(chunk_texts, embeddings)):
        chunk_id = chunk_hash_fn(chunk)
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
        client.upsert(collection_name=collection_name, points=points)


def ingest_to_graph(chunk_texts: List[str], filename: str, chunk_hash_fn, graph_api_url: str = "http://localhost:8000/api/v1/ingest/graph", batch_size: int = 10):
    chunks_data = [{"text": chunk, "chunk_id": chunk_hash_fn(chunk)} for chunk in chunk_texts]
    
    state_file = "graph_ingest_state.json"
    processed_batches = set()
    
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                state_data = json.load(f)
                processed_batches = set(state_data.get(filename, []))
        except Exception:
            pass
            
    # Load entire state structure for updates
    full_state = {}
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                full_state = json.load(f)
        except Exception:
            pass
            
    if filename not in full_state:
        full_state[filename] = []
    
    for i in range(0, len(chunks_data), batch_size):
        batch_index = i // batch_size
        
        # Skip if already processed
        if batch_index in processed_batches:
            continue
            
        batch = chunks_data[i:i + batch_size]
        try:
            graph_resp = requests.post(
                graph_api_url,
                json={"chunks": batch},
                timeout=120
            )
            # Check for Rate Limits explicitly
            if graph_resp.status_code == 429:
                print(f"Rate limit hit! Sleeping for 60 seconds before retrying...")
                time.sleep(60)
                # Retry once
                graph_resp = requests.post(graph_api_url, json={"chunks": batch}, timeout=120)
                
            graph_resp.raise_for_status()
            
            # Save progress immediately
            processed_batches.add(batch_index)
            full_state[filename].append(batch_index)
            with open(state_file, "w") as f:
                json.dump(full_state, f)
                
            # Sleep 4 seconds to maintain ~15 requests per minute
            time.sleep(4) 
        except Exception as ex:
            print(f"Graph ingestion failed for batch {batch_index} in {filename}: {ex}")
            # If it fails, break the loop so we don't spam the API and can resume later
            break
