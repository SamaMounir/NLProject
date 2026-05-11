"""
Phase 1 Runner — ONE command to do everything.

Run from the src/ directory:
    python phase1_pipeline.py

What it does:
    1. Checks your OpenAI API key is set
    2. Parses all PDFs in assets/documents/job_pdfs/
    3. Chunks the text (500-token chunks, 50-token overlap)
    4. Embeds each chunk with text-embedding-3-small
    5. Stores everything in Qdrant at assets/db/qdrant_data/

Prerequisites:
    - pip install -r requirements.txt
    - Set OPENAI_API_KEY in your .env file
"""

import os
import sys
import argparse
import pathlib

# Make sure we're running from src/
src_dir = pathlib.Path(__file__).parent
os.chdir(src_dir)
sys.path.insert(0, str(src_dir))

from controllers.indexer import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 1 RAG pipeline")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe and re-index the Qdrant collection from scratch",
    )
    args = parser.parse_args()

    docs_dir = "assets/documents/job_pdfs"

    if not os.path.exists(docs_dir) or not any(
        f.endswith(".pdf") for f in os.listdir(docs_dir)
    ):
        print("No PDFs found in", docs_dir)
        sys.exit(1)

    run_pipeline(
        documents_dir=docs_dir,
        db_path="assets/db/qdrant_data",
        collection_name="job_documents",
        reset=args.reset,  # now correctly reads --reset from command line
    )