"""
PDF Vector Indexing API Route Handler.

Provides the POST /index/{document_id} endpoint for generating vector indices.
"""

from fastapi import APIRouter, Query, status
from app.schemas.indexing import IndexingResponse
from app.schemas.upload import HTTPErrorDetail
from app.services.pdf.indexing_service import IndexingService

router = APIRouter()
indexing_service = IndexingService()


@router.post(
    "/index/{document_id}",
    response_model=IndexingResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Vector Index",
    description="Loads dense embeddings for a document, builds a local FAISS IndexFlatIP index, validates it, and saves index files atomically.",
    responses={
        200: {
            "model": IndexingResponse,
            "description": "Vector index created successfully"
        },
        400: {
            "model": HTTPErrorDetail,
            "description": "Bad Request (Missing files, pipeline incomplete, or already indexed without force=true)"
        },
        404: {
            "model": HTTPErrorDetail,
            "description": "Not Found (Specified document_id does not exist)"
        },
        409: {
            "model": HTTPErrorDetail,
            "description": "Conflict (Indexing already running for this document)"
        },
        500: {
            "model": HTTPErrorDetail,
            "description": "Internal Server Error during indexing operation"
        }
    }
)
async def generate_index(
    document_id: str,
    force: bool = Query(
        default=False,
        description="Force index rebuilding even if indexing status is marked completed"
    )
) -> IndexingResponse:
    """
    HTTP POST endpoint handler for vector indexing.

    Args:
        document_id (str): Unique UUID document identifier.
        force (bool): Optional query flag to force index rebuilding.

    Returns:
        IndexingResponse: Metadata summary detailing indexing execution.
    """
    return await indexing_service.generate_document_index(document_id, force=force)
