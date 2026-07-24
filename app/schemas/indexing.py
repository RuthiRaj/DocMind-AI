"""
Vector Indexing Pydantic Schemas.

Defines response models for indexing operation summaries.
"""

from pydantic import BaseModel, Field


class IndexingResponse(BaseModel):
    """
    Response model summarizing vector index creation execution.
    """
    success: bool = Field(
        default=True,
        description="Indicates successful indexing pipeline completion",
        examples=[True]
    )
    document_id: str = Field(
        ...,
        description="Unique UUID document identifier",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )
    index_type: str = Field(
        ...,
        description="Type of vector index created",
        examples=["IndexFlatIP"]
    )
    vector_dimension: int = Field(
        ...,
        description="Dimensionality size of indexed vectors",
        examples=[384]
    )
    indexed_vectors: int = Field(
        ...,
        description="Total number of vectors successfully inserted into the index",
        examples=[154]
    )
    processing_time_ms: int = Field(
        ...,
        description="Total indexing pipeline duration in milliseconds",
        examples=[121]
    )
    message: str = Field(
        default="Vector index created successfully.",
        description="Summary status message",
        examples=["Vector index created successfully."]
    )
