"""
Text Chunking Engine API Route Handler.

Provides the POST /chunk/{document_id} endpoint for chunking extracted document text.
"""

from fastapi import APIRouter, Query, status
from app.schemas.chunking import ChunkingResponse
from app.schemas.upload import HTTPErrorDetail
from app.services.pdf.chunking_service import ChunkingService

router = APIRouter()
chunking_service = ChunkingService()


@router.post(
    "/chunk/{document_id}",
    response_model=ChunkingResponse,
    status_code=status.HTTP_200_OK,
    summary="Chunk Extracted Document Text",
    description="Loads extracted_text.txt for a document, preprocesses text, and generates semantic overlapping chunks saved to chunks.json.",
    responses={
        200: {
            "model": ChunkingResponse,
            "description": "Chunking completed successfully"
        },
        400: {
            "model": HTTPErrorDetail,
            "description": "Bad Request (Missing extracted_text.txt, empty text, or already chunked without force=true)"
        },
        404: {
            "model": HTTPErrorDetail,
            "description": "Not Found (Specified document_id does not exist)"
        },
        500: {
            "model": HTTPErrorDetail,
            "description": "Internal Server Error during chunking operation"
        }
    }
)
async def chunk_document(
    document_id: str,
    force: bool = Query(
        default=False,
        description="Force re-chunking even if chunking status is marked completed"
    )
) -> ChunkingResponse:
    """
    HTTP POST endpoint handler for text chunking.

    Args:
        document_id (str): Unique UUID document identifier.
        force (bool): Optional query flag to force re-chunking.

    Returns:
        ChunkingResponse: Metadata summary detailing chunking execution.
    """
    return await chunking_service.chunk_document(document_id, force=force)
