"""
PDF Processing Engine API Route Handler.

Provides the POST /process/{document_id} endpoint for processing uploaded PDF documents.
"""

from fastapi import APIRouter, Query, status
from app.schemas.processing import PDFProcessingResponse
from app.schemas.upload import HTTPErrorDetail
from app.services.pdf.processing_service import PDFProcessingService

router = APIRouter()
processing_service = PDFProcessingService()


@router.post(
    "/process/{document_id}",
    response_model=PDFProcessingResponse,
    status_code=status.HTTP_200_OK,
    summary="Process Uploaded PDF Document",
    description="Opens an uploaded PDF document by document_id, validates structure, extracts metadata and full text to disk, and updates document status.",
    responses={
        200: {
            "model": PDFProcessingResponse,
            "description": "PDF processed successfully"
        },
        400: {
            "model": HTTPErrorDetail,
            "description": "Bad Request (e.g. Password protected or corrupted PDF file)"
        },
        404: {
            "model": HTTPErrorDetail,
            "description": "Not Found (Specified document_id does not exist)"
        },
        409: {
            "model": HTTPErrorDetail,
            "description": "Conflict (Processing already running for this document)"
        },
        500: {
            "model": HTTPErrorDetail,
            "description": "Internal Server Error during PDF processing"
        }
    }
)
async def process_pdf(
    document_id: str,
    force: bool = Query(
        default=False,
        description="Force PDF re-processing even if processing status is marked completed"
    )
) -> PDFProcessingResponse:
    """
    HTTP POST endpoint handler for PDF processing.

    Args:
        document_id (str): Unique UUID document identifier returned by upload API.
        force (bool): Optional query flag to force re-processing.

    Returns:
        PDFProcessingResponse: Summary metadata detailing processing results.
    """
    return await processing_service.process_pdf(document_id, force=force)
