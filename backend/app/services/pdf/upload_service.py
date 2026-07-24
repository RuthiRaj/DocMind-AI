"""
PDF Upload Service Layer.

Encapsulates business logic for validating, streaming, and storing PDF files safely.
Organizes files into uploads/<document_id>/original.pdf and initializes status.json.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from fastapi import UploadFile, HTTPException, status

from app.core.config import settings
from app.schemas.upload import UploadSuccessResponse
from app.utils.formatters import format_file_size

# Initialize logger for upload operations
logger = logging.getLogger(__name__)

# Constants
ALLOWED_EXTENSION = ".pdf"
ALLOWED_MIME_TYPE = "application/pdf"
CHUNK_SIZE = 1024 * 1024  # 1 MB chunk for streaming read/write


class PDFUploadService:
    """
    Service responsible for validating PDF uploads, establishing document directories,
    and initializing lifecycle status tracking.
    """

    def __init__(self, target_dir: Path | None = None):
        """
        Initialize the PDFUploadService using application settings configuration.
        """
        if target_dir is None:
            self.target_dir = Path(__file__).resolve().parent.parent.parent.parent / settings.UPLOAD_DIRECTORY
        else:
            self.target_dir = target_dir

        self.target_dir.mkdir(parents=True, exist_ok=True)

    async def save_pdf(self, file: UploadFile) -> UploadSuccessResponse:
        """
        Validates an incoming UploadFile, creates a dedicated document folder uploads/<document_id>/,
        stores original.pdf, and writes initial status.json.

        Args:
            file (UploadFile): The uploaded file object from FastAPI route handler.

        Returns:
            UploadSuccessResponse: Metadata regarding the stored document including document_id.
        """
        # Rule 1: Validate file presence
        if not file or not file.filename:
            detail_msg = "No file uploaded."
            logger.warning("Upload validation failed: %s", detail_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=detail_msg
            )

        original_filename = file.filename.strip()
        logger.info("Upload started for file: '%s'", original_filename)

        # Rule 2: Validate file extension (.pdf)
        if not original_filename.lower().endswith(ALLOWED_EXTENSION):
            detail_msg = "Invalid file extension. Only .pdf files are allowed."
            logger.warning("Upload validation failed for file '%s': %s", original_filename, detail_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=detail_msg
            )

        # Rule 3: Validate MIME type (application/pdf)
        if file.content_type != ALLOWED_MIME_TYPE:
            detail_msg = "Invalid MIME type. Only application/pdf is supported."
            logger.warning("Upload validation failed for file '%s': %s", original_filename, detail_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=detail_msg
            )

        # Generate unique document_id UUID and dedicated document folder
        document_id = str(uuid.uuid4())
        doc_dir = self.target_dir / document_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        destination_path = doc_dir / "original.pdf"
        total_bytes = 0

        try:
            # Stream file chunks to destination_path
            with open(destination_path, "wb") as buffer:
                while True:
                    chunk = await file.read(CHUNK_SIZE)
                    if not chunk:
                        break

                    total_bytes += len(chunk)

                    # Validate maximum upload size using settings limit
                    if total_bytes > settings.MAX_UPLOAD_SIZE:
                        buffer.close()
                        if doc_dir.exists():
                            for f in doc_dir.iterdir():
                                f.unlink()
                            doc_dir.rmdir()

                        max_mb = settings.MAX_UPLOAD_SIZE // (1024 * 1024)
                        detail_msg = f"File size exceeds maximum limit of {max_mb} MB."
                        logger.warning("Upload failed for file '%s': %s", original_filename, detail_msg)
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=detail_msg
                        )

                    buffer.write(chunk)

        except HTTPException:
            raise
        except Exception as exc:
            if doc_dir.exists():
                for f in doc_dir.iterdir():
                    f.unlink()
                doc_dir.rmdir()
            logger.error("Upload failed due to unexpected error for file '%s': %s", original_filename, str(exc), exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An error occurred while saving the file: {str(exc)}"
            )

        # Reject empty files (0 bytes)
        if total_bytes == 0:
            if doc_dir.exists():
                for f in doc_dir.iterdir():
                    f.unlink()
                doc_dir.rmdir()
            detail_msg = "File is empty."
            logger.warning("Upload validation failed for file '%s': %s", original_filename, detail_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=detail_msg
            )

        human_size = format_file_size(total_bytes)
        uploaded_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Initialize status.json for document lifecycle management
        status_data = {
            "document_id": document_id,
            "upload_status": "completed",
            "processing_status": "pending",
            "chunking_status": "pending",
            "embedding_status": "pending",
            "indexing_status": "pending",
            "chat_ready": False,
            "created_at": uploaded_at,
            "updated_at": uploaded_at
        }
        status_path = doc_dir / "status.json"
        with open(status_path, "w", encoding="utf-8") as sf:
            json.dump(status_data, sf, indent=4)

        logger.info(
            "Upload successful for document_id '%s' ('%s', %d bytes / %s)",
            document_id,
            original_filename,
            total_bytes,
            human_size
        )

        return UploadSuccessResponse(
            success=True,
            document_id=document_id,
            original_filename=original_filename,
            stored_filename="original.pdf",
            file_size_bytes=total_bytes,
            file_size=human_size,
            uploaded_at=uploaded_at,
            message="PDF uploaded successfully."
        )
