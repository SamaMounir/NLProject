"""
Test Retrieval - Check if semantic search works
"""

import os
import sys
import pathlib
import requests
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from stores.vectordb.provider.qdrant_provider import QdrantProvider

# ── Config from .env ───────────────────────────────────────────────────────────
OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL",     "http://localhost:11434")
EMBEDDING_MODEL     = os.getenv("EMBEDDINGS_MODEL",    "nomic-embed-text")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))
QDRANT_DB_PATH      = os.getenv("VECTOR_DB_PATH",     "assets/db/qdrant_data")
COLLECTION_NAME     = os.getenv("COLLECTION_NAME",    "job_documents")

# ── Setup ──────────────────────────────────────────────────────────────────────
db = QdrantProvider(
    db_path=QDRANT_DB_PATH,
    collection_name=COLLECTION_NAME,
    vector_size=EMBEDDING_DIMENSION,
)

print("🔍 Job Matching RAG Test\n")

test_queries = [
    "Python Django FastAPI backend developer REST API",
    "machine learning data science pandas scikit-learn",
    "AutoCAD Revit civil structural site engineer BOQ",
    "Google Ads SEO content marketing social media",
    "SAP Oracle financial analyst IFRS tax accountant"
]

for query in test_queries:
    print(f"Query: {query}")

    # Create query embedding via Ollama
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": [query]},
        timeout=30,
    )
    response.raise_for_status()
    query_vector = response.json()["embeddings"][0]

    # Search
    results = db.search(query_vector, top_k=3)

    for i, r in enumerate(results, 1):
        print(f"  {i}. Score: {r['score']:.4f} | {r['source_file']}")
        print(f"     {r['text'][:120]}...\n")

    print("-" * 80)

print("Test completed!")