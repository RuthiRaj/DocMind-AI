"""
PDF Document Management API Route Handlers.

Provides document listing, detail queries, pipeline monitoring, deletion, and storage statistics.
"""

from typing import Optional
from fastapi import APIRouter, Query, status

from app.schemas.management import (
    DocumentListResponse,
    DocumentDetailResponse,
    PipelineStatusResponse,
    DeleteResponse,
    StorageStatisticsResponse,
)
from app.services.pdf.management_service import ManagementService

router = APIRouter()
management_service = ManagementService()


@router.get(
    "/documents/statistics",
    response_model=StorageStatisticsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get System Storage Statistics",
    description="Analyzes uploaded document structures and yields total documents count, status details, chunk totals, and storage byte metrics."
)
async def get_storage_statistics() -> StorageStatisticsResponse:
    """
    HTTP GET endpoint handler for aggregated storage statistics.
    """
    stats_data = management_service.calculate_storage_statistics()
    return StorageStatisticsResponse(**stats_data)


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Processed Documents",
    description="Retrieves a paginated list of uploaded document metadata summaries. Supports skipping, limit size bounds, sort keys, sorting direction, and stage state filters."
)
async def list_documents(
    skip: int = Query(default=0, ge=0, description="Pagination skip offset"),
    limit: int = Query(default=20, ge=1, le=100, description="Pagination limit size"),
    sort_by: str = Query(default="upload_time", description="Attribute property key to sort list by"),
    descending: bool = Query(default=True, description="True for descending order, False for ascending"),
    status_filter: Optional[str] = Query(default=None, description="Filter list items by furthest pipeline completion stage ('upload', 'processing', 'chunking', 'embedding', 'indexing')")
) -> DocumentListResponse:
    """
    HTTP GET endpoint handler to list processed documents.
    """
    items, count = management_service.list_documents(
        skip=skip,
        limit=limit,
        sort_by=sort_by,
        descending=descending,
        status_filter=status_filter
    )
    return DocumentListResponse(success=True, documents=items, total_count=count)


@router.get(
    "/documents/{document_id}",
    response_model=DocumentDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Document Metadata Details",
    description="Fetches aggregated metadata definitions from metadata.json, status.json, and statistical configurations. Excludes raw text arrays."
)
async def get_document(document_id: str) -> DocumentDetailResponse:
    """
    HTTP GET endpoint handler for specific document details.
    """
    return management_service.get_document(document_id)


@router.get(
    "/documents/{document_id}/status",
    response_model=PipelineStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Pipeline Stage Status Tracker",
    description="Dedicated query endpoint checking upload, page processing, text chunking, embedding generation, and indexing stages status."
)
async def get_pipeline_status(document_id: str) -> PipelineStatusResponse:
    """
    HTTP GET endpoint handler for pipeline status tracking.
    """
    return management_service.get_pipeline_status(document_id)


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete Uploaded Document",
    description="Recursively deletes the specified document folder directory, clearing all original files, text outputs, vectors, indexes, and statistics."
)
async def delete_document(document_id: str) -> DeleteResponse:
    """
    HTTP DELETE endpoint handler for document lifecycle removal.
    """
    res = management_service.delete_document(document_id)
    return DeleteResponse(**res)
