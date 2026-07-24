"""
Document Management & Pipeline Engine Pydantic Schemas.

Defines schemas for listing, details, status tracker, delete replies,
storage analytics, health status, and maintenance cleanup operations.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentListItem(BaseModel):
    """
    Brief summary properties of a single document in listing.
    """
    document_id: str = Field(..., description="Unique UUID document identifier")
    filename: str = Field(..., description="Original filename of the uploaded PDF")
    upload_time: Optional[str] = Field(None, description="ISO-8601 UTC upload timestamp")
    total_pages: Optional[int] = Field(None, description="Total pages extracted from the PDF")
    total_chunks: Optional[int] = Field(None, description="Total text chunks generated")
    embedding_count: Optional[int] = Field(None, description="Total vector embeddings generated")
    index_status: Optional[str] = Field(None, description="Current indexing pipeline state")
    chat_ready: bool = Field(False, description="True if indexing is completed and ready to chat")
    document_size: int = Field(0, description="Size of original uploaded PDF in bytes")
    current_pipeline_stage: str = Field("upload", description="Furthest completed step in the pipeline")


class DocumentListResponse(BaseModel):
    """
    Response model for paginated list queries.
    """
    success: bool = Field(True, description="Indicates if query was resolved successfully")
    documents: List[DocumentListItem] = Field(..., description="Array of matching document items")
    total_count: int = Field(..., description="Total documents matching filters in uploads folder")


class DocumentDetailResponse(BaseModel):
    """
    Response model containing detailed aggregated document properties.
    """
    success: bool = Field(True, description="Indicates if query was resolved successfully")
    document_id: str = Field(..., description="Unique UUID document identifier")
    metadata: Optional[dict] = Field(None, description="Content from metadata.json")
    status: Optional[dict] = Field(None, description="Content from status.json")
    chunk_statistics: Optional[dict] = Field(None, description="Content from chunk_statistics.json")
    embedding_metadata: Optional[dict] = Field(None, description="Content from embedding_metadata.json")
    index_metadata: Optional[dict] = Field(None, description="Content from index_metadata.json")


class PipelineStatusResponse(BaseModel):
    """
    Response model detailing pipeline stage state.
    """
    success: bool = Field(True, description="Indicates if query was resolved successfully")
    document_id: str = Field(..., description="Unique UUID document identifier")
    upload_status: Optional[str] = Field(None, description="State of file ingestion")
    processing_status: Optional[str] = Field(None, description="State of page extraction")
    chunking_status: Optional[str] = Field(None, description="State of smart chunking")
    embedding_status: Optional[str] = Field(None, description="State of vector generation")
    indexing_status: Optional[str] = Field(None, description="State of index compilation")
    chat_ready: bool = Field(False, description="Indicates if chat interface is ready")


class DeleteResponse(BaseModel):
    """
    Response model summarizing document deletion.
    """
    success: bool = Field(..., description="True if document deletion succeeded")
    message: str = Field(..., description="Detailed status message")


class StorageStatisticsResponse(BaseModel):
    """
    Response model representing system storage metrics.
    """
    total_documents: int = Field(0, description="Total documents uploaded")
    completed_documents: int = Field(0, description="Documents ready for chat")
    failed_documents: int = Field(0, description="Documents containing failure flags")
    processing_documents: int = Field(0, description="Documents currently being processed/embedded/indexed")
    total_pages: int = Field(0, description="Sum of all document pages")
    total_chunks: int = Field(0, description="Sum of all document chunks")
    total_embeddings: int = Field(0, description="Sum of all embedding vectors")
    total_indexes: int = Field(0, description="Sum of all generated indexes")
    storage_bytes: int = Field(0, description="Total storage consumed by uploads folder in bytes")
    generated_at: str = Field(..., description="Timestamp of stats evaluation")


class HealthCheckResponse(BaseModel):
    """
    Response model detailing comprehensive backend readiness diagnostic check.
    """
    status: str = Field(..., description="Overall health of the system ('healthy' or 'unhealthy')")
    uploads_directory: dict = Field(..., description="Upload directory existence and accessibility details")
    write_permission: dict = Field(..., description="Storage directory write permission verification")
    disk_usage: dict = Field(..., description="Local storage capacity metrics")
    embedding_model: dict = Field(..., description="SentenceTransformer dependency status")
    faiss_library: dict = Field(..., description="FAISS CPU dependency status")
    groq_service: dict = Field(..., description="Groq settings configuration status")
    backend_version: str = Field(..., description="FastAPI core backend system version")
    uptime_seconds: float = Field(..., description="Active server uptime duration in seconds")
    total_documents: int = Field(..., description="Active documents count in uploads path")


class CleanupResponse(BaseModel):
    """
    Response model summarizing system maintenance pruning.
    """
    success: bool = Field(..., description="True if maintenance cleanup succeeded")
    removed_temp_files: int = Field(..., description="Total temporary/orphan files deleted")
    removed_empty_directories: int = Field(..., description="Total empty folder directories removed")
    message: str = Field(..., description="Summary status message details")
