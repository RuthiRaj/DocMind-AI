"""
PDF Processing Pydantic Schemas.

Defines response models for PDF processing operation summaries.
"""

from typing import Optional
from pydantic import BaseModel, Field


class PDFProcessingResponse(BaseModel):
    """
    Response model for PDF processing summary operations.
    """
    success: bool = Field(
        default=True,
        description="Indicates successful PDF processing completion",
        examples=[True]
    )
    document_id: str = Field(
        ...,
        description="Unique UUID document identifier",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )
    filename: str = Field(
        ...,
        description="Filename or title of the processed PDF",
        examples=["Machine Learning.pdf"]
    )
    total_pages: int = Field(
        ...,
        description="Total page count of the processed PDF",
        examples=[87]
    )
    text_length: int = Field(
        ...,
        description="Total character length of extracted text",
        examples=[158420]
    )
    processing_time_ms: int = Field(
        ...,
        description="Processing duration in milliseconds",
        examples=[184]
    )
    processed_at: str = Field(
        ...,
        description="ISO 8601 UTC timestamp of processing completion",
        examples=["2026-07-22T10:00:00Z"]
    )
    message: str = Field(
        default="PDF processed successfully.",
        description="Summary status message",
        examples=["PDF processed successfully."]
    )
