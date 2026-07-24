"""
PDF Processing Engine Service Layer.

Encapsulates business logic for validating, reading, extracting metadata,
pages character offsets mapping (pages.json), and saving extracted text & analytics using PyMuPDF (fitz).
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
import fitz  # PyMuPDF
from fastapi import HTTPException, status

from app.core.config import settings
from app.schemas.processing import PDFProcessingResponse

# Initialize logger for processing operations
logger = logging.getLogger(__name__)


class PDFProcessingService:
    """
    Service responsible for opening, validating, and extracting metadata & text
    from uploaded PDF documents using PyMuPDF.
    """

    def __init__(self, target_dir: Path | None = None):
        """
        Initialize the PDFProcessingService with target upload storage directory.
        """
        if target_dir is None:
            self.target_dir = Path(__file__).resolve().parent.parent.parent.parent / settings.UPLOAD_DIRECTORY
        else:
            self.target_dir = target_dir

    async def process_pdf(self, document_id: str) -> PDFProcessingResponse:
        """
        Processes a previously uploaded PDF document by ID.

        Args:
            document_id (str): Unique document UUID identifier.

        Returns:
            PDFProcessingResponse: Summary response object detailing processing results.
        """
        safe_doc_id = Path(document_id).name
        doc_dir = self.target_dir / safe_doc_id
        pdf_path = doc_dir / "original.pdf"

        # Legacy fallback if stored directly as <document_id>.pdf
        if not pdf_path.exists():
            direct_pdf = self.target_dir / f"{safe_doc_id}.pdf"
            if direct_pdf.exists():
                doc_dir.mkdir(parents=True, exist_ok=True)
                destination_path = doc_dir / "original.pdf"
                direct_pdf.rename(destination_path)
                pdf_path = destination_path

        # Validate file existence
        if not doc_dir.exists() or not pdf_path.exists():
            detail_msg = f"File not found for document_id: {document_id}"
            logger.warning("Processing failed: %s", detail_msg)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=detail_msg
            )

        logger.info("Processing started for document_id: '%s'", safe_doc_id)
        start_time = time.perf_counter()

        doc = None
        try:
            # Open PDF document via PyMuPDF
            try:
                doc = fitz.open(pdf_path)
            except Exception as open_err:
                logger.warning("Corrupted PDF for document_id '%s': %s", safe_doc_id, str(open_err))
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid or corrupted PDF file."
                )

            # Check encryption / password protection
            if doc.is_encrypted:
                logger.warning("Password protected PDF for document_id '%s'", safe_doc_id)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="PDF is password protected and cannot be processed."
                )

            # Validate basic PDF structure
            if not doc.is_pdf or len(doc) == 0:
                logger.warning("Corrupted or empty PDF structure for document_id '%s'", safe_doc_id)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid or corrupted PDF file."
                )

            logger.info("PDF opened successfully for document_id: '%s'", safe_doc_id)

            # Extract PDF Document Metadata
            raw_metadata = doc.metadata or {}
            title = raw_metadata.get("title").strip() if raw_metadata.get("title") else None
            author = raw_metadata.get("author").strip() if raw_metadata.get("author") else None
            creator = raw_metadata.get("creator").strip() if raw_metadata.get("creator") else None
            producer = raw_metadata.get("producer").strip() if raw_metadata.get("producer") else None

            logger.info("Metadata extracted for document_id: '%s'", safe_doc_id)

            # Extract Text Page by Page & build pages.json character offset index
            logger.info("Text extraction started for document_id: '%s'", safe_doc_id)
            total_pages = len(doc)
            page_texts = []
            pages_meta = []
            empty_pages = 0
            current_char_offset = 0

            for page_index in range(total_pages):
                page = doc.load_page(page_index)
                page_text = page.get_text() or ""
                page_len = len(page_text)

                if not page_text.strip():
                    empty_pages += 1

                start_char = current_char_offset
                end_char = current_char_offset + page_len

                pages_meta.append({
                    "page": page_index + 1,
                    "start_character": start_char,
                    "end_character": end_char,
                    "character_count": page_len
                })

                page_texts.append(page_text)
                # Next page offset starts after page_text plus 2 characters for "\n\n" separator
                current_char_offset = end_char + 2

            full_text = "\n\n".join(page_texts)
            text_length = len(full_text)
            avg_chars = round(text_length / total_pages, 2) if total_pages > 0 else 0.0

            logger.info("Text extraction completed for document_id '%s' (%d characters extracted)", safe_doc_id, text_length)

            # Save clean extracted text to uploads/<document_id>/extracted_text.txt
            extracted_text_path = doc_dir / "extracted_text.txt"
            with open(extracted_text_path, "w", encoding="utf-8") as tf:
                tf.write(full_text)
            logger.info("Extracted text saved to '%s'", extracted_text_path.name)

            # Save page offset mapping to uploads/<document_id>/pages.json
            pages_json_path = doc_dir / "pages.json"
            with open(pages_json_path, "w", encoding="utf-8") as pf:
                json.dump(pages_meta, pf, indent=4)
            logger.info("Pages index saved to '%s'", pages_json_path.name)

            # Measure total processing duration in milliseconds
            end_time = time.perf_counter()
            processing_time_ms = int(round((end_time - start_time) * 1000))
            processed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            file_size_bytes = pdf_path.stat().st_size if pdf_path.exists() else 0

            display_filename = title if title else f"{safe_doc_id}.pdf"

            # Save metadata to uploads/<document_id>/metadata.json
            metadata_payload = {
                "document_id": safe_doc_id,
                "filename": display_filename,
                "total_pages": total_pages,
                "title": title,
                "author": author,
                "creator": creator,
                "producer": producer,
                "text_length": text_length,
                "average_characters_per_page": avg_chars,
                "empty_pages": empty_pages,
                "file_size_bytes": file_size_bytes,
                "processing_time_ms": processing_time_ms,
                "processed_at": processed_at
            }
            metadata_path = doc_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as mf:
                json.dump(metadata_payload, mf, indent=4)
            logger.info("Metadata saved to '%s'", metadata_path.name)

            # Update status.json with processing completion
            status_path = doc_dir / "status.json"
            if status_path.exists():
                try:
                    with open(status_path, "r", encoding="utf-8") as sf:
                        status_data = json.load(sf)
                    status_data["processing_status"] = "completed"
                    status_data["updated_at"] = processed_at
                    with open(status_path, "w", encoding="utf-8") as sf:
                        json.dump(status_data, sf, indent=4)
                except Exception as status_err:
                    logger.warning("Failed to update status.json for document_id '%s': %s", safe_doc_id, str(status_err))

            logger.info("Processing completed in %d ms for document_id: '%s'", processing_time_ms, safe_doc_id)

            return PDFProcessingResponse(
                success=True,
                document_id=safe_doc_id,
                filename=display_filename,
                total_pages=total_pages,
                text_length=text_length,
                processing_time_ms=processing_time_ms,
                processed_at=processed_at,
                message="PDF processed successfully."
            )

        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Unexpected processing failure for document_id '%s': %s", safe_doc_id, str(exc), exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred while processing the PDF: {str(exc)}"
            )
        finally:
            if doc:
                doc.close()
