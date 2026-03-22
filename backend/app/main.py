import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import router
from app.graph.client import setup_constraints

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure Neo4j constraints are created
    try:
        setup_constraints()
    except Exception as e:
        print(f"Warning: Could not setup Neo4j constraints: {e}")
    yield
    # Shutdown logic (if any) goes here

app = FastAPI(
    title="Self-Healing RAG",
    description=(
        "Self-Healing RAG."
        "Submit a question and receive a Gemini-generated answer grounded in your document corpus."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}
