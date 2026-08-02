"""
Semantic Retrieval Pydantic Schemas.

Defines Pydantic request, result item, and response models.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from app.core.config import settings


class RetrievalRequest(BaseModel):
    """
    Request payload schema for semantic search retrieval.
    """
    query: str = Field(
        ...,
        description="Query text string to retrieve relevant document segments",
        min_length=1,
        max_length=settings.MAX_QUERY_LENGTH,
        examples=["Explain supervised learning"]
    )
    top_k: int = Field(
        default=settings.DEFAULT_TOP_K,
        ge=1,
        le=settings.MAX_TOP_K,
        description="Maximum number of relevant chunks to retrieve",
        examples=[5]
    )


class RetrievalResult(BaseModel):
    """
    Metadata representation of a single matching text chunk search result.
    """
    chunk_id: str = Field(
        ...,
        description="Unique chunk identifier string",
        examples=["550e8400_chunk_000012"]
    )
    chunk_index: int = Field(
        ...,
        description="Sequential 1-based index of the chunk",
        examples=[12]
    )
    rank: int = Field(
        ...,
        description="Similarity ranking position (starting at 1)",
        examples=[1]
    )
    score: float = Field(
        ...,
        description="L2-normalized cosine similarity score rounded to 4 decimals",
        examples=[0.9438]
    )
    start_page: int = Field(
        ...,
        description="Page number where the chunk starts (1-based)",
        examples=[3]
    )
    end_page: int = Field(
        ...,
        description="Page number where the chunk ends (1-based)",
        examples=[3]
    )
    start_character: int = Field(
        ...,
        description="Starting character index within original document text",
        examples=[13241]
    )
    end_character: int = Field(
        ...,
        description="Ending character index within original document text",
        examples=[14002]
    )
    sentence_count: int = Field(
        ...,
        description="Sentence count within the chunk",
        examples=[8]
    )
    estimated_tokens: int = Field(
        ...,
        description="Estimated token count",
        examples=[195]
    )
    character_count: int = Field(
        ...,
        description="Total character count",
        examples=[761]
    )
    word_count: int = Field(
        ...,
        description="Total word count",
        examples=[128]
    )
    text: str = Field(
        ...,
        description="Text content of the retrieved chunk",
        examples=["Supervised learning is..."]
    )
    document_id: str = Field(
        ...,
        description="Unique UUID document identifier",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )
    last_chunk_index: Optional[int] = Field(
        default=None,
        description="Sequential 1-based index of the last chunk if merged",
        examples=[13]
    )


class RetrievalResponse(BaseModel):
    """
    Response schema detailing semantic search results.
    """
    success: bool = Field(
        default=True,
        description="Indicates successful retrieval execution",
        examples=[True]
    )
    document_id: str = Field(
        ...,
        description="Unique UUID document identifier",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )
    query: str = Field(
        ...,
        description="Query text string",
        examples=["Explain supervised learning"]
    )
    total_results: int = Field(
        ...,
        description="Total number of results returned above similarity threshold",
        examples=[5]
    )
    processing_time_ms: int = Field(
        ...,
        description="Duration of query execution in milliseconds",
        examples=[31]
    )
    retrieval_version: str = Field(
        default=settings.RETRIEVAL_VERSION,
        description="Retrieval engine schema version",
        examples=["1.0"]
    )
    results: List[RetrievalResult] = Field(
        ...,
        description="Ranked list of matching semantic results"
    )
