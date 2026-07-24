"""
PDF Upload API Route Handler.

Provides the POST /upload endpoint for uploading PDF documents.
"""

from fastapi import APIRouter, File, UploadFile, status
from app.schemas.upload import UploadSuccessResponse, HTTPErrorDetail
from app.services.pdf.upload_service import PDFUploadService

router = APIRouter()
upload_service = PDFUploadService()


@router.post(
    "/upload",
    response_model=UploadSuccessResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload PDF Document",
    description="Accepts a single PDF file (max 10 MB), validates extension and MIME type, and stores it with a unique UUID.",
    responses={
        200: {
            "model": UploadSuccessResponse,
            "description": "PDF uploaded successfully"
        },
        400: {
            "model": HTTPErrorDetail,
            "description": "Bad Request (e.g. No file, invalid extension, invalid MIME type, or empty file)"
        },
        413: {
            "model": HTTPErrorDetail,
            "description": "Payload Too Large (file size exceeds 10 MB)"
        },
        500: {
            "model": HTTPErrorDetail,
            "description": "Internal Server Error during storage"
        }
    }
)
async def upload_pdf(
    file: UploadFile = File(..., description="Single PDF document file to upload")
) -> UploadSuccessResponse:
    """
    HTTP POST Endpoint handler for PDF file upload.

    Args:
        file (UploadFile): Uploaded file from multipart/form-data request.

    Returns:
        UploadSuccessResponse: Metadata object detailing successful storage.
    """
    return await upload_service.save_pdf(file)
