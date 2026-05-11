"""
Phase 1 — Step 2: Chunking Strategy
Splits cleaned page text into overlapping chunks suitable for embedding.
"""

import os
import re
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Configuration — read from .env, fall back to sensible defaults ─────────────
# FILE_CHUNK_SIZE in .env is in characters (~600 chars ≈ 150 tokens for English)
CHARS_PER_CHUNK = int(os.getenv("FILE_CHUNK_SIZE", "600"))
CHARS_OVERLAP   = CHARS_PER_CHUNK // 10    # 10% overlap


@dataclass
class TextChunk:
    """One embeddable chunk of text with provenance metadata."""
    chunk_id:    str
    source_file: str
    page:        int
    language:    str
    chunk_index: int    # 0-based index within the parent page
    text:        str
    metadata:    dict = field(default_factory=dict)


def extract_job_title(source_file: str) -> str:
    """
    Derive a human-readable job title from the PDF filename.

    Examples:
      job_0000_Senior_Python_Developer.pdf  →  Senior Python Developer
      job_0042_Financial_Analyst.pdf        →  Financial Analyst
    """
    name = os.path.splitext(os.path.basename(source_file))[0]  # strip .pdf
    name = re.sub(r'^job_\d+_', '', name)            # remove "job_0000_" prefix
    name = name.replace('_', ' ').replace('-', ' ')  # underscores/hyphens → spaces
    name = re.sub(r'\s+', ' ', name).strip()          # collapse extra spaces
    return name


def build_splitter() -> RecursiveCharacterTextSplitter:
    """
    Build a RecursiveCharacterTextSplitter for English job description text.

    Separator priority (tried in order until a split fits within chunk_size):
      1. Paragraph break (\\n\\n) — most natural boundary in job descriptions
      2. Line break (\\n)         — section items / bullet points
      3. Period + space (". ")   — sentence boundary
      4. Space (" ")             — word boundary (last resort)
      5. "" (empty string)       — hard character cut (absolute fallback)
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHARS_PER_CHUNK,
        chunk_overlap=CHARS_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )


def chunk_parsed_page(parsed_chunk) -> list[TextChunk]:
    """
    Split one ParsedChunk (a parsed PDF page) into a list of TextChunks.

    Each output TextChunk is independent and self-contained — it carries its own
    source provenance (file, page, language) so retrieval results can always be
    traced back to the original document without needing the parent object.

    A job title prefix is prepended to every chunk so the embedding model always
    has role context, even when the chunk content is a generic section like
    requirements or responsibilities that looks similar across many job types.
    """
    splitter   = build_splitter()
    raw_splits = splitter.split_text(parsed_chunk.cleaned_text)
    job_title  = extract_job_title(parsed_chunk.source_file)

    chunks: list[TextChunk] = []
    for idx, text in enumerate(raw_splits):
        text = text.strip()
        if not text:
            continue

        # Prepend job title so every chunk is self-contained for retrieval.
        # Format: "Senior Python Developer: <chunk text>"
        prefixed_text = f"{job_title}: {text}"

        chunks.append(TextChunk(
            chunk_id=parsed_chunk.chunk_id,
            source_file=parsed_chunk.source_file,
            page=parsed_chunk.page,
            language="en",
            chunk_index=idx,
            text=prefixed_text,
            metadata={
                **parsed_chunk.metadata,
                "chunk_index":      idx,
                "chunk_size_chars": len(prefixed_text),
                "approx_tokens":    len(prefixed_text) // 4,
                "job_title":        job_title,
            }
        ))

    return chunks


def chunk_all(parsed_pages: list) -> list[TextChunk]:
    """
    Chunk a full list of ParsedChunk objects.
    Returns a flat list of TextChunk objects ready for embedding.
    """
    all_chunks: list[TextChunk] = []
    for page in parsed_pages:
        all_chunks.extend(chunk_parsed_page(page))

    print(f"[chunker] {len(parsed_pages)} pages → {len(all_chunks)} chunks")
    avg = len(all_chunks) / max(len(parsed_pages), 1)
    print(f"[chunker] Average {avg:.1f} chunks per page")

    return all_chunks