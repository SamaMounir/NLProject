"""
Phase 1 — Step 3: Embedder + Indexer
Embeds TextChunks using Ollama (nomic-embed-text) and stores them in Qdrant.

Prerequisites:
    ollama pull nomic-embed-text
    ollama serve   (must be running before the pipeline starts)
"""

import os
import sys
import pathlib
import argparse
import time

import requests
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from controllers.parser import parse_directory
from controllers.chunker import chunk_all
from stores.vectordb.provider.qdrant_provider import QdrantProvider

# ── Configuration — all values read from .env ──────────────────────────────────
OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL",     "http://localhost:11434")
EMBEDDING_MODEL     = os.getenv("EMBEDDINGS_MODEL",    "nomic-embed-text")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))
EMBED_BATCH_SIZE    = 20

QDRANT_DB_PATH  = os.getenv("VECTOR_DB_PATH",  "assets/db/qdrant_data")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "job_documents")
DOCUMENTS_DIR   = os.getenv("DOCUMENTS_DIR",   "assets/documents/job_pdfs")


# ── Ollama helpers ─────────────────────────────────────────────────────────────

def check_ollama_running():
    """Confirm Ollama is reachable and the embedding model is available."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}.\n"
            f"Make sure Ollama is running:  ollama serve\n"
            f"Original error: {e}"
        )

    models = [m["name"] for m in r.json().get("models", [])]
    if not any(EMBEDDING_MODEL in m for m in models):
        raise RuntimeError(
            f"Model '{EMBEDDING_MODEL}' is not pulled.\n"
            f"Run:  ollama pull {EMBEDDING_MODEL}"
        )

    print(f"[indexer] Ollama is running. Using model '{EMBEDDING_MODEL}' "
          f"(dim={EMBEDDING_DIMENSION}).")


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Send a batch of texts to Ollama and return their embedding vectors."""
    # Replace empty strings so Ollama doesn't reject the request
    texts = [t if t.strip() else "empty document" for t in texts]
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": texts},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embeddings"]


def embed_all_chunks(chunks: list) -> list[list[float]]:
    """
    Embed all chunks in batches of EMBED_BATCH_SIZE.
    Retries each failed batch once before filling with zero vectors.
    """
    all_vectors: list[list[float]] = []
    total = len(chunks)
    print(f"[indexer] Embedding {total} chunks with '{EMBEDDING_MODEL}' via Ollama...")

    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[i: i + EMBED_BATCH_SIZE]
        texts = [c.text for c in batch]

        try:
            vectors = embed_batch(texts)
            all_vectors.extend(vectors)
        except Exception as e:
            print(f"  Batch {i // EMBED_BATCH_SIZE + 1} failed: {e} — retrying in 3s...")
            time.sleep(3)
            try:
                vectors = embed_batch(texts)
                all_vectors.extend(vectors)
            except Exception as e2:
                print(f"  Failed again: {e2}. Filling with zero vectors.")
                all_vectors.extend([[0.0] * EMBEDDING_DIMENSION] * len(batch))

        done = min(i + EMBED_BATCH_SIZE, total)
        if done % 100 == 0 or done == total:
            print(f"  Embedded {done}/{total} chunks...")

    return all_vectors


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(
    documents_dir: str = DOCUMENTS_DIR,
    db_path: str = QDRANT_DB_PATH,
    collection_name: str = COLLECTION_NAME,
    reset: bool = False,
):
    print("=" * 60)
    print("Phase 1: Data Processing & Vectorization Pipeline")
    print("=" * 60)

    # Step 0 — confirm Ollama is up before doing any work
    check_ollama_running()

    # Step 1 — parse all PDFs into page objects
    print("\n[Step 1/4] Parsing PDFs...")
    parsed_pages = parse_directory(documents_dir)
    if not parsed_pages:
        print("No pages parsed. Check your documents directory.")
        return
    print(f"  Parsed {len(parsed_pages)} pages from "
          f"{len(set(p.source_file for p in parsed_pages))} files")

    # Step 2 — split pages into overlapping text chunks
    print("\n[Step 2/4] Chunking pages...")
    chunks = chunk_all(parsed_pages)
    print(f"  Created {len(chunks)} chunks")

    # Step 3 — convert each chunk's text into a 768-dim embedding vector
    print("\n[Step 3/4] Embedding chunks with Ollama...")
    vectors = embed_all_chunks(chunks)
    print(f"  Embedded {len(vectors)} chunks (dim={EMBEDDING_DIMENSION})")

    # Step 4 — store chunks + vectors in Qdrant for later retrieval
    print("\n[Step 4/4] Storing in Qdrant...")
    db = QdrantProvider(
        db_path=db_path,
        collection_name=collection_name,
        vector_size=EMBEDDING_DIMENSION,
    )
    if reset:
        db.reset_collection()

    db.upsert_chunks(chunks, vectors)

    print("\n" + "=" * 60)
    print("Phase 1 Complete!")
    print(f"   Documents parsed : {len(set(p.source_file for p in parsed_pages))}")
    print(f"   Chunks created   : {len(chunks)}")
    print(f"   Vectors stored   : {len(vectors)}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Delete and recreate the Qdrant collection before indexing")
    args = parser.parse_args()
    run_pipeline(reset=args.reset)