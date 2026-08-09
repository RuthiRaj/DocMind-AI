"""
PDF Embedding API Route Handler.

Provides the POST /embed/{document_id} endpoint for generating vector embeddings.
"""

from fastapi import APIRouter, Query, status, HTTPException
from app.schemas.embedding import EmbeddingResponse
from app.schemas.upload import HTTPErrorDetail
from app.services.pdf.embedding_service import EmbeddingService
from app.services.pdf.pipeline_validator import is_valid_uuid

router = APIRouter()
embedding_service = EmbeddingService()


@router.post(
    "/embed/{document_id}",
    response_model=EmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Vector Embeddings",
    description="Loads text chunks for a document, generates dense float32 vector embeddings, and saves the vectors, metadata, and statistics.",
    responses={
        200: {
            "model": EmbeddingResponse,
            "description": "Embeddings generated successfully"
        },
        400: {
            "model": HTTPErrorDetail,
            "description": "Bad Request (Missing chunks, pipeline incomplete, or already embedded without force=true)"
        },
        404: {
            "model": HTTPErrorDetail,
            "description": "Not Found (Specified document_id does not exist)"
        },
        409: {
            "model": HTTPErrorDetail,
            "description": "Conflict (Embedding generation already running for this document)"
        },
        500: {
            "model": HTTPErrorDetail,
            "description": "Internal Server Error during vector generation"
        }
    }
)
async def generate_embeddings(
    document_id: str,
    force: bool = Query(
        default=False,
        description="Force vector regeneration even if embedding status is marked completed"
    )
) -> EmbeddingResponse:
    """
    HTTP POST endpoint handler for vector embedding generation.

    Args:
        document_id (str): Unique UUID document identifier.
        force (bool): Optional query flag to force embedding generation.

    Returns:
        EmbeddingResponse: Metadata summary detailing vector generation execution.
    """
    if not is_valid_uuid(document_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format. Document ID must be a valid UUID v4."
        )
    return await embedding_service.generate_document_embeddings(document_id, force=force)
