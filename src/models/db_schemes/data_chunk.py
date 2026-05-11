"""
Phase 1 — Data Models
Pydantic schemas for document chunks and search results.
Used for validation and API response typing in later phases.
"""

from pydantic import BaseModel, Field
from typing import Optional


class DataChunk(BaseModel):
    """Schema for a single text chunk stored in the vector DB."""
    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    source_file: str = Field(..., description="Name of the source PDF file")
    page: int = Field(..., description="Page number this chunk came from")
    language: str = Field(..., description="Detected language: 'en' or 'ar'")
    chunk_index: int = Field(..., description="Index of this chunk within its page")
    text: str = Field(..., description="The actual text content")
    chunk_size_chars: Optional[int] = Field(None, description="Length of text in characters")


class RetrievalDocument(BaseModel):
    """Schema for a single vector search result."""
    score: float = Field(..., description="Cosine similarity score (0-1, higher = more relevant)")
    text: str = Field(..., description="The retrieved chunk text")
    source_file: str = Field(..., description="Source PDF filename")
    page: Optional[int] = Field(None, description="Page number")
    language: str = Field(..., description="Language of the chunk")
    chunk_index: Optional[int] = Field(None, description="Chunk index within the page")


class IndexingResponseSchema(BaseModel):
    """Schema for the Phase 1 indexing summary."""
    documents_parsed: int
    pages_extracted: int
    chunks_created: int
    vectors_stored: int
    collection_name: str
    status: str = "success"