"""
Phase 1 — Step 1: Document Parser
Reads PDFs from assets/documents/job_pdfs/, extracts text per page,
cleans messy PyMuPDF output, and attaches metadata.

"""

import os
import re
import uuid
import fitz  # PyMuPDF
from dataclasses import dataclass, field


@dataclass
class ParsedChunk:
    """Represents one page/section extracted from a document."""
    chunk_id:     str
    source_file:  str
    page:         int
    language:     str
    raw_text:     str
    cleaned_text: str
    metadata:     dict = field(default_factory=dict)


def _fix_spacing(text: str) -> str:
    """Fix PyMuPDF's common broken-spacing issue: 'P y t h o n' → 'Python'."""
    if re.search(r'(?<!\w)(\w ){4,}\w(?!\w)', text):
        text = re.sub(r'(?<!\w)((?:\w ){3,}\w)(?!\w)',
                      lambda m: m.group().replace(' ', ''), text)
    return text


def _clean_text(text: str) -> str:
    """Clean extracted text: fix spacing, remove control characters, normalise whitespace."""
    text = _fix_spacing(text)
    # Remove null bytes and control characters (except newline/tab)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Collapse 3+ consecutive newlines → 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip trailing spaces from each line
    text = '\n'.join(line.rstrip() for line in text.splitlines())
    return text.strip()


def parse_pdf(file_path: str) -> list[ParsedChunk]:
    """
    Parse a PDF file into a list of ParsedChunk objects (one per page).
    Extracts text, cleans the content, and attaches metadata.
    Returns an empty list if the file cannot be read.
    """
    if not os.path.exists(file_path):
        print(f"  [parser] File not found: {file_path}")
        return []

    filename = os.path.basename(file_path)
    chunks: list[ParsedChunk] = []

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"  [parser] Cannot open {filename}: {e}")
        return []

    for page_num in range(len(doc)):
        page     = doc[page_num]
        raw_text = page.get_text("text")

        # Skip empty pages
        if not raw_text or not raw_text.strip():
            continue

        cleaned = _clean_text(raw_text)

        # Skip trivially short pages
        if len(cleaned) < 30:
            continue

        chunks.append(ParsedChunk(
            chunk_id=str(uuid.uuid4()),
            source_file=filename,
            page=page_num + 1,
            language="en",
            raw_text=raw_text,
            cleaned_text=cleaned,
            metadata={
                "source_file": filename,
                "page":        page_num + 1,
                "language":    "en",
                "total_pages": len(doc),
            }
        ))

    doc.close()
    return chunks


def parse_directory(documents_dir: str) -> list[ParsedChunk]:
    """
    Walk a directory and parse all PDF files found.
    Returns a flat list of ParsedChunk objects from all documents.
    """
    all_chunks: list[ParsedChunk] = []
    pdf_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(documents_dir)
        for f in files
        if f.lower().endswith(".pdf")
    ]

    print(f"[parser] Found {len(pdf_files)} PDF files in {documents_dir}")

    for i, pdf_path in enumerate(pdf_files, 1):
        chunks = parse_pdf(pdf_path)
        all_chunks.extend(chunks)
        if i % 20 == 0 or i == len(pdf_files):
            print(f"  [parser] Parsed {i}/{len(pdf_files)} files → {len(all_chunks)} pages total")

    return all_chunks