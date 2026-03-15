# Self-Healing RAG

A **Retrieval-Augmented Generation (RAG)** system with self-healing capabilities — automatically detects and corrects failures in the retrieval and generation pipeline.

## Architecture

```
Documents (PDF, DOCX, TXT, MD, JSON)
        │
        ▼
  Data Ingestion
  (PyMuPDF + Docling)
        │
  Semantic Chunking
  (LangChain SemanticChunker)
        │
  Embedding Generation
  (SentenceTransformers)
        │
        ▼
   Qdrant Vector DB  ◄──── RAG Query Pipeline
```

## Project Structure

```
Self_Healing_RAG/
├── backend/
│   ├── Scripts/
│   │   └── data_ingestion.py   # Ingestion pipeline
│   ├── Data/                   # Drop documents here (gitignored)
│   ├── requirements.txt
│   └── .env.example
├── frontend/                   # (Coming soon)
├── docker-compose.yml          # Qdrant local setup
└── README.md
```

## Setup

### 1. Prerequisites

- Python 3.10+
- [Docker](https://www.docker.com/) (for local Qdrant)
- CUDA-capable GPU (optional but recommended for embeddings)

### 2. Start Qdrant

```bash
docker compose up -d
```

### 3. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 5. Add documents & ingest

Drop your documents into `backend/Data/` (supports `.pdf`, `.docx`, `.txt`, `.md`, `.json`), then run:

```bash
python backend/Scripts/data_ingestion.py
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `QDRANT_API_KEY` | API key (leave empty for local) | `` |
| `EMBEDDING_MODEL` | HuggingFace embedding model name | `all-MiniLM-L12-v2` |
| `USE_GPU` | Use CUDA for embeddings (`1`/`0`) | `1` |

## Tech Stack

| Component | Library |
|---|---|
| PDF parsing | PyMuPDF |
| DOCX parsing | Docling |
| Semantic chunking | LangChain `SemanticChunker` |
| Embeddings | SentenceTransformers |
| Vector store | Qdrant |
