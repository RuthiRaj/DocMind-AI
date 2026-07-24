"""
Text Chunking Pydantic Schemas.

Defines response models for enriched chunk items and chunking operation summaries.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ChunkItem(BaseModel):
    """
    Production-grade representation of an individual text chunk with rich source location
    and embedding metadata.
    """
    chunk_id: str = Field(
        ...,
        description="Stable unique chunk identifier (e.g. 550e8400_chunk_000001)",
        examples=["550e8400_chunk_000001"]
    )
    chunk_index: int = Field(
        ...,
        description="Sequential 1-based index of the chunk",
        examples=[1]
    )
    document_id: str = Field(
        ...,
        description="Unique UUID document identifier",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )
    start_character: int = Field(
        ...,
        description="Starting character offset within extracted_text.txt",
        examples=[0]
    )
    end_character: int = Field(
        ...,
        description="Ending character offset within extracted_text.txt",
        examples=[798]
    )
    start_page: int = Field(
        ...,
        description="Starting PDF page number (1-based)",
        examples=[1]
    )
    end_page: int = Field(
        ...,
        description="Ending PDF page number (1-based)",
        examples=[2]
    )
    character_count: int = Field(
        ...,
        description="Character length of the chunk text",
        examples=[798]
    )
    word_count: int = Field(
        ...,
        description="Word count of the chunk text",
        examples=[132]
    )
    sentence_count: int = Field(
        ...,
        description="Sentence count within the chunk text",
        examples=[9]
    )
    estimated_tokens: int = Field(
        ...,
        description="Estimated token count using token estimation ratio",
        examples=[200]
    )
    embedding_status: str = Field(
        default="pending",
        description="Embedding generation pipeline status placeholder",
        examples=["pending"]
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description="Embedding model name placeholder",
        examples=[None]
    )
    vector_dimension: Optional[int] = Field(
        default=None,
        description="Embedding vector dimension size placeholder",
        examples=[None]
    )
    created_at: str = Field(
        ...,
        description="ISO 8601 UTC timestamp of chunk creation",
        examples=["2026-07-22T10:00:00Z"]
    )
    text: str = Field(
        ...,
        description="Extracted chunk text content",
        examples=["Machine Learning is a subfield of artificial intelligence..."]
    )


class ChunkingResponse(BaseModel):
    """
    Response model for chunking operation summaries.
    """
    success: bool = Field(
        default=True,
        description="Indicates successful chunking operation",
        examples=[True]
    )
    document_id: str = Field(
        ...,
        description="Unique UUID document identifier",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )
    total_chunks: int = Field(
        ...,
        description="Total number of chunks generated",
        examples=[154]
    )
    average_chunk_size: int = Field(
        ...,
        description="Average character length per chunk",
        examples=[796]
    )
    average_tokens: int = Field(
        ...,
        description="Average estimated token count per chunk",
        examples=[199]
    )
    processing_time_ms: int = Field(
        ...,
        description="Chunking execution time in milliseconds",
        examples=[82]
    )
    chunk_version: str = Field(
        default="1.0",
        description="Chunking engine schema version",
        examples=["1.0"]
    )
    message: str = Field(
        default="Chunking completed successfully.",
        description="Summary status message",
        examples=["Chunking completed successfully."]
    )
