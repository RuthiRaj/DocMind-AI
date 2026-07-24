"""
PDF Vector Similarity Retrieval API Route Handler.

Provides the POST /retrieve/{document_id} endpoint for semantic chunk retrieval.
"""

from fastapi import APIRouter, status
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse
from app.schemas.upload import HTTPErrorDetail
from app.services.pdf.retrieval_service import RetrievalService

router = APIRouter()
retrieval_service = RetrievalService()


@router.post(
    "/retrieve/{document_id}",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic Retrieval Search",
    description="Embeds a query text, queries the local FAISS vector index, performs checks, filters and sorts by similarity score, and returns matching text chunks.",
    responses={
        200: {
            "model": RetrievalResponse,
            "description": "Semantic query retrieval successfully completed"
        },
        400: {
            "model": HTTPErrorDetail,
            "description": "Bad Request (Empty query, pipeline incomplete, or search parameters range validation failure)"
        },
        404: {
            "model": HTTPErrorDetail,
            "description": "Not Found (Specified document_id does not exist)"
        },
        500: {
            "model": HTTPErrorDetail,
            "description": "Internal Server Error (Index validation discrepancies or unexpected engine faults)"
        }
    }
)
async def query_document(
    document_id: str,
    request: RetrievalRequest
) -> RetrievalResponse:
    """
    HTTP POST endpoint handler for semantic chunk retrieval.

    Args:
        document_id (str): Unique UUID document identifier.
        request (RetrievalRequest): Query query parameters.

    Returns:
        RetrievalResponse: List of ranked similarity results.
    """
    return await retrieval_service.query_document(document_id, request)
