"""
Vector Embedding Pydantic Schemas.

Defines response models for embedding operation summaries.
"""

from pydantic import BaseModel, Field


class EmbeddingResponse(BaseModel):
    """
    Response model summarizing vector embedding generation execution.
    """
    success: bool = Field(
        default=True,
        description="Indicates successful embedding pipeline completion",
        examples=[True]
    )
    document_id: str = Field(
        ...,
        description="Unique UUID document identifier",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )
    embedding_model: str = Field(
        ...,
        description="Embedding model name used for vector generation",
        examples=["BAAI/bge-small-en-v1.5"]
    )
    embedding_dimension: int = Field(
        ...,
        description="Vector output dimension",
        examples=[384]
    )
    total_embeddings: int = Field(
        ...,
        description="Total number of chunks embedded",
        examples=[154]
    )
    processing_time_ms: int = Field(
        ...,
        description="Total embedding pipeline duration in milliseconds",
        examples=[742]
    )
    message: str = Field(
        default="Embeddings generated successfully.",
        description="Status summary message",
        examples=["Embeddings generated successfully."]
    )
