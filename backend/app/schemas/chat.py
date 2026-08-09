"""
AI Chat (RAG) Engine Pydantic Schemas.

Defines Pydantic request, source chunk item, and response models.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from app.core.config import settings


class ChatRequest(BaseModel):
    """
    Request payload schema for document-grounded AI Chat completion.
    """
    question: str = Field(
        ...,
        description="The user's query question grounded to the document context",
        min_length=1,
        max_length=settings.MAX_QUERY_LENGTH,
        examples=["Explain supervised learning"]
    )
    top_k: int = Field(
        default=settings.DEFAULT_TOP_K,
        ge=1,
        le=settings.MAX_TOP_K,
        description="Maximum number of context chunks to retrieve",
        examples=[5]
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional client-generated session UUID for conversation memory scoping. If omitted, a server-side UUID is generated and returned in the response.",
        examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890"]
    )


class SourceChunk(BaseModel):
    """
    Source citation representation detailing which chunk was referenced.
    """
    chunk_id: str = Field(
        ...,
        description="Unique chunk identifier string",
        examples=["550e8400_chunk_000012"]
    )
    chunk_index: int = Field(
        ...,
        description="Sequential 1-based index of the chunk in the document",
        examples=[12]
    )
    last_chunk_index: Optional[int] = Field(
        default=None,
        description="Sequential 1-based index of the last chunk if merged",
        examples=[13]
    )
    score: float = Field(
        ...,
        description="Normalized cosine similarity score rounded to 4 decimals",
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
    text: str = Field(
        ...,
        description="The raw text of the chunk",
        examples=["Grounded chunk text content here..."]
    )


class ChatResponse(BaseModel):
    """
    Response schema detailing AI Chat completion.
    """
    success: bool = Field(
        default=True,
        description="Indicates successful RAG generation",
        examples=[True]
    )
    document_id: str = Field(
        ...,
        description="Unique UUID document identifier",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )
    request_id: str = Field(
        ...,
        description="Unique UUID tracing the lifecycle of this query execution request",
        examples=["3c984920-a6e5-4f40-84a5-927429184ba2"]
    )
    question: str = Field(
        ...,
        description="The user's original query question",
        examples=["Explain supervised learning"]
    )
    answer: str = Field(
        ...,
        description="Grounded AI response generation",
        examples=["Supervised learning is an algorithm type that..."]
    )
    provider: str = Field(
        ...,
        description="Active LLM provider name",
        examples=["groq"]
    )
    model: str = Field(
        ...,
        description="Active LLM model name",
        examples=["llama-3.1-8b-instant"]
    )
    retrieval_version: str = Field(
        default=settings.RETRIEVAL_VERSION,
        description="Retrieval engine version",
        examples=["1.0"]
    )
    chat_version: str = Field(
        default=settings.CHAT_VERSION,
        description="Chat engine version",
        examples=["1.0"]
    )
    system_prompt_version: str = Field(
        default=settings.SYSTEM_PROMPT_VERSION,
        description="Core system prompt configuration version",
        examples=["1.0"]
    )
    processing_time_ms: int = Field(
        ...,
        description="Total duration of execution in milliseconds",
        examples=[660]
    )
    retrieval_time_ms: int = Field(
        ...,
        description="Duration of chunk retrieval in milliseconds",
        examples=[18]
    )
    generation_time_ms: int = Field(
        ...,
        description="Duration of LLM text generation in milliseconds",
        examples=[642]
    )
    sources: List[SourceChunk] = Field(
        ...,
        description="Cited chunk sources referenced in context"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session UUID for conversation memory. Returned so the client can persist and send on subsequent requests.",
        examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890"]
    )
    context_mode: Optional[str] = Field(
        default=None,
        description="Context routing mode used for this request: 'FULL_CONTEXT' or 'RAG'",
        examples=["FULL_CONTEXT"]
    )
