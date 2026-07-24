"""
PDF Upload Pydantic Schemas.

Defines the response models for successful PDF file uploads and HTTP errors.
"""

from pydantic import BaseModel, Field


class UploadSuccessResponse(BaseModel):
    """
    Response model for successful PDF file upload operations.
    """
    success: bool = Field(
        default=True,
        description="Indicates successful upload completion",
        examples=[True]
    )
    document_id: str = Field(
        ...,
        description="Unique UUID identifier for the uploaded document",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )
    original_filename: str = Field(
        ...,
        description="Original name of the uploaded PDF file",
        examples=["Machine Learning.pdf"]
    )
    stored_filename: str = Field(
        default="original.pdf",
        description="Filename used inside the document directory",
        examples=["original.pdf"]
    )
    file_size_bytes: int = Field(
        ...,
        description="Size of the uploaded file in raw bytes",
        examples=[6215354]
    )
    file_size: str = Field(
        ...,
        description="Human-readable formatted file size",
        examples=["5.93 MB"]
    )
    uploaded_at: str = Field(
        ...,
        description="ISO 8601 UTC timestamp of when the file was uploaded",
        examples=["2026-07-21T21:30:00Z"]
    )
    message: str = Field(
        default="PDF uploaded successfully.",
        description="Status message summarizing the upload result",
        examples=["PDF uploaded successfully."]
    )


class HTTPErrorDetail(BaseModel):
    """
    Response schema for HTTP error responses (400, 413, 404, 500).
    """
    detail: str = Field(
        ...,
        description="Detailed description of the error condition",
        examples=["Invalid file extension. Only .pdf files are allowed."]
    )
