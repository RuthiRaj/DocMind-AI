"""
PDF Processing Engine Service Layer.

Encapsulates business logic for validating, reading, extracting metadata,
pages character offsets mapping (pages.json), and saving extracted text & analytics using PyMuPDF (fitz).
Implements atomic file writing and multi-state pipeline transitions.
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import fitz  # PyMuPDF
from fastapi import HTTPException, status

from app.core.config import settings
from app.schemas.processing import PDFProcessingResponse
from app.services.pdf.pipeline_validator import PipelineLockManager, validate_process_artifacts

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

    def _clean_text(self, raw_text: str) -> str:
        """
        Cleans and normalizes text so extracted_text.txt and pages.json
        share identical character offsets.
        """
        import re
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        text = re.sub(r" ?\n ?", "\n", text)
        text = re.sub(r"\n\n+", "\n\n", text)
        return text.strip()

    def _write_atomic(self, target_path: Path, content: any, is_string: bool = False) -> None:
        """
        Atomically writes content to disk using temporary swap path.
        """
        temp_path = target_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as tf:
                if is_string:
                    tf.write(content)
                else:
                    json.dump(content, tf, indent=4)
                tf.flush()
                os.fsync(tf.fileno())

            # Atomic swap
            os.replace(temp_path, target_path)
        except Exception as exc:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            logger.exception("Atomic file write failed for path '%s': %s", target_path, str(exc))
            raise

    def _update_status(self, status_path: Path, new_status: str, extra_fields: dict | None = None) -> dict:
        """
        Helper method to update status.json securely and return the updated dictionary.
        """
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            with open(status_path, "r", encoding="utf-8") as sf:
                status_data = json.load(sf)
        except Exception as err:
            logger.warning("Failed to read status.json. Constructing default state tracker: %s", str(err))
            status_data = {}

        status_data["processing_status"] = new_status
        status_data["updated_at"] = now_str

        if extra_fields:
            status_data.update(extra_fields)

        self._write_atomic(status_path, status_data)
        logger.info("Pipeline state updated to '%s' in status.json", new_status)
        return status_data

    async def process_pdf(self, document_id: str, force: bool = False) -> PDFProcessingResponse:
        """
        Processes a previously uploaded PDF document by ID.

        Args:
            document_id (str): Unique document UUID identifier.
            force (bool): Force re-processing even if already completed.

        Returns:
            PDFProcessingResponse: Summary response object detailing processing results.
        """
        safe_doc_id = Path(document_id).name
        doc_dir = self.target_dir / safe_doc_id
        pdf_path = doc_dir / "original.pdf"
        status_path = doc_dir / "status.json"
        metadata_path = doc_dir / "metadata.json"

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

        # Prevent concurrent duplicate execution for the same document ID
        if not PipelineLockManager.acquire_stage(safe_doc_id, "process"):
            detail_msg = "PDF processing is currently running for this document. Duplicate execution rejected."
            logger.warning(detail_msg)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=detail_msg
            )

        try:
            import asyncio
            await asyncio.sleep(0.05)

            # Idempotency and artifact validation check
            is_completed = False
            if status_path.exists():
                try:
                    with open(status_path, "r", encoding="utf-8") as sf:
                        status_data = json.load(sf)
                    if status_data.get("processing_status") == "completed":
                        is_completed = True
                except Exception:
                    pass

            artifacts_valid, validation_msg = validate_process_artifacts(doc_dir)

            if not force and is_completed and artifacts_valid:
                logger.info("PDF processing already completed for document_id '%s'. Reusing valid existing artifacts.", safe_doc_id)
                try:
                    with open(metadata_path, "r", encoding="utf-8") as mf:
                        meta = json.load(mf)
                    return PDFProcessingResponse(
                        success=True,
                        document_id=safe_doc_id,
                        filename=meta.get("filename", f"{safe_doc_id}.pdf"),
                        total_pages=meta.get("total_pages", 0),
                        text_length=meta.get("text_length", 0),
                        processing_time_ms=meta.get("processing_time_ms", 0),
                        processed_at=meta.get("processed_at", ""),
                        message="PDF processed successfully (cached result)."
                    )
                except Exception as e:
                    logger.warning("Failed to read metadata.json for document_id '%s': %s. Re-running process stage...", safe_doc_id, str(e))

            request_id = str(uuid.uuid4())
            logger.info("[STAGE: PROCESS] request_id=%s document_id='%s' status=started", request_id, safe_doc_id)
            self._update_status(status_path, "running")
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

                total_pages = len(doc)
                if total_pages > settings.MAX_PDF_PAGES:
                    logger.warning("PDF page count (%d) exceeds limit of %d for document_id '%s'", total_pages, settings.MAX_PDF_PAGES, safe_doc_id)
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"PDF page count ({total_pages}) exceeds the maximum limit of {settings.MAX_PDF_PAGES} pages."
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
                    raw_page_text = page.get_text() or ""
                    page_text = self._clean_text(raw_page_text)
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
                if text_length > settings.MAX_EXTRACTED_TEXT_SIZE:
                    logger.warning("Extracted text size (%d) exceeds limit of %d for document_id '%s'", text_length, settings.MAX_EXTRACTED_TEXT_SIZE, safe_doc_id)
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Extracted text length ({text_length}) exceeds the maximum allowed size of {settings.MAX_EXTRACTED_TEXT_SIZE} characters."
                    )
                avg_chars = round(text_length / total_pages, 2) if total_pages > 0 else 0.0

                logger.info("Text extraction completed for document_id '%s' (%d characters extracted)", safe_doc_id, text_length)

                # Save clean extracted text to uploads/<document_id>/extracted_text.txt atomically
                extracted_text_path = doc_dir / "extracted_text.txt"
                self._write_atomic(extracted_text_path, full_text, is_string=True)
                logger.info("Extracted text saved atomically to '%s'", extracted_text_path.name)

                # Save page offset mapping to uploads/<document_id>/pages.json atomically
                pages_json_path = doc_dir / "pages.json"
                self._write_atomic(pages_json_path, pages_meta)
                logger.info("Pages index saved atomically to '%s'", pages_json_path.name)

                # Measure total processing duration in milliseconds
                end_time = time.perf_counter()
                processing_time_ms = int(round((end_time - start_time) * 1000))
                processed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                file_size_bytes = pdf_path.stat().st_size if pdf_path.exists() else 0

                orig_fn = None
                if status_path.exists():
                    try:
                        with open(status_path, "r", encoding="utf-8") as sf:
                            orig_fn = json.load(sf).get("original_filename")
                    except Exception:
                        pass
                display_filename = orig_fn or title or f"{safe_doc_id}.pdf"

                # Save metadata to uploads/<document_id>/metadata.json atomically
                metadata_payload = {
                    "document_id": safe_doc_id,
                    "pipeline_version": 2,
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
                self._write_atomic(metadata_path, metadata_payload)
                logger.info("Metadata saved atomically to '%s'", metadata_path.name)

                # Update status.json with processing completion
                self._update_status(status_path, "completed")
                logger.info(
                    "[STAGE: PROCESS] request_id=%s document_id='%s' status=completed elapsed_ms=%d pages=%d chars=%d",
                    request_id,
                    safe_doc_id,
                    processing_time_ms,
                    total_pages,
                    text_length
                )

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

        except Exception as err:
            self._update_status(status_path, "failed")
            if isinstance(err, HTTPException):
                raise
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to complete PDF processing: {str(err)}"
            )
        finally:
            PipelineLockManager.release_stage(safe_doc_id, "process")
