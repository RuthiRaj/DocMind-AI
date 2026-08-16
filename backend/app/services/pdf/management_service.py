"""
Document Management Service Layer.

Provides business logic for querying document listing, fetching aggregated details,
monitoring pipeline stages, deleting document folders, and compiling storage analytics.
"""

import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from fastapi import HTTPException, status

from app.core.config import settings
from app.schemas.management import DocumentListItem, DocumentDetailResponse, PipelineStatusResponse, StorageStatisticsResponse

logger = logging.getLogger(__name__)


class ManagementService:
    """
    Service layer coordinating document collection listing, metadata details, and safe deletes.
    """

    def __init__(self, target_dir: Path | None = None):
        """
        Initialize the service.
        """
        if target_dir is None:
            self.target_dir = Path(__file__).resolve().parent.parent.parent.parent / settings.UPLOAD_DIRECTORY
        else:
            self.target_dir = target_dir

        self.target_dir.mkdir(parents=True, exist_ok=True)

    def _write_atomic(self, target_path: Path, content: dict) -> None:
        """
        Atomically writes system statistics dictionary to disk.
        """
        temp_path = target_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as tf:
                json.dump(content, tf, indent=4)
                tf.flush()
                os.fsync(tf.fileno())

            # Atomic rename replace
            os.replace(temp_path, target_path)
        except Exception as exc:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            logger.exception("Atomic file write failed for management service statistics: %s", str(exc))
            raise

    def list_documents(
        self,
        skip: int = 0,
        limit: int = 20,
        sort_by: str = "upload_time",
        descending: bool = True,
        status_filter: Optional[str] = None
    ) -> Tuple[List[DocumentListItem], int]:
        """
        Retrieves summary items listing for document directories inside the uploads folder.
        """
        logger.info("Listing documents: skip=%d, limit=%d, sort_by=%s", skip, limit, sort_by)
        document_items: List[DocumentListItem] = []

        # List folders recursively in the uploads path (skipping statistics files or non-dirs)
        for child in self.target_dir.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue

            doc_id = child.name
            status_path = child / "status.json"
            metadata_path = child / "metadata.json"
            pdf_path = child / "original.pdf"
            emb_meta_path = child / "embedding_metadata.json"

            # Skip directories that do not contain original file
            if not pdf_path.exists():
                continue

            # Default values if status.json is missing or corrupt
            upload_time = None
            total_pages = None
            total_chunks = None
            embedding_count = None
            index_status = "pending"
            chat_ready = False
            document_size = pdf_path.stat().st_size if pdf_path.exists() else 0
            current_stage = "upload"

            # Parse status.json
            if status_path.exists():
                try:
                    with open(status_path, "r", encoding="utf-8") as sf:
                        status_data = json.load(sf)
                    
                    index_status = status_data.get("indexing_status", "pending")
                    chat_ready = (index_status == "completed")

                    # Deduce furthest completed stage
                    if status_data.get("indexing_status") == "completed":
                        current_stage = "indexing"
                    elif status_data.get("embedding_status") == "completed":
                        current_stage = "embedding"
                    elif status_data.get("chunking_status") == "completed":
                        current_stage = "chunking"
                    elif status_data.get("processing_status") == "completed":
                        current_stage = "processing"
                    else:
                        current_stage = "upload"
                except Exception as err:
                    logger.warning("Failed to parse status.json for document_id '%s': %s", doc_id, str(err))

            # Filter out status if specified
            if status_filter and current_stage != status_filter:
                continue

            # Parse metadata.json
            meta_filename = None
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r", encoding="utf-8") as mf:
                        meta_data = json.load(mf)
                    upload_time = meta_data.get("upload_time") or meta_data.get("processed_at")
                    total_pages = meta_data.get("total_pages")
                    meta_filename = meta_data.get("filename")
                except Exception as err:
                    logger.warning("Failed to parse metadata.json for document_id '%s': %s", doc_id, str(err))

            # Retrieve total chunks from metadata or chunks.json
            chunks_path = child / "chunks.json"
            if chunks_path.exists():
                try:
                    with open(chunks_path, "r", encoding="utf-8") as cf:
                        chunks_data = json.load(cf)
                    total_chunks = len(chunks_data)
                except Exception:
                    pass

            # Retrieve embedding counts from embedding metadata
            if emb_meta_path.exists():
                try:
                    with open(emb_meta_path, "r", encoding="utf-8") as emf:
                        emb_data = json.load(emf)
                    embedding_count = emb_data.get("total_chunks") or emb_data.get("embedding_count")
                except Exception:
                    pass

            # Default upload_time fallback if missing
            if not upload_time:
                # Use PDF creation date from filesystem
                stat_info = pdf_path.stat()
                creation_time = datetime.fromtimestamp(stat_info.st_mtime, timezone.utc)
                upload_time = creation_time.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Resolve user-facing display filename (prefer original_filename, then metadata filename, fallback to doc_id)
            orig_name = None
            if status_path.exists():
                try:
                    with open(status_path, "r", encoding="utf-8") as sf:
                        st_json = json.load(sf)
                    orig_name = st_json.get("original_filename")
                except Exception:
                    pass

            resolved_filename = orig_name or meta_filename or f"Document-{doc_id[:8]}.pdf"
            if resolved_filename in ["(anonymous)", "original.pdf"]:
                resolved_filename = orig_name or f"Document-{doc_id[:8]}.pdf"

            document_items.append(
                DocumentListItem(
                    document_id=doc_id,
                    filename=resolved_filename,
                    upload_time=upload_time,
                    total_pages=total_pages,
                    total_chunks=total_chunks,
                    embedding_count=embedding_count,
                    index_status=index_status,
                    chat_ready=chat_ready,
                    document_size=document_size,
                    current_pipeline_stage=current_stage
                )
            )

        total_count = len(document_items)

        # Dynamic Sorting
        def get_sort_key(item: DocumentListItem):
            val = getattr(item, sort_by, None)
            return val if val is not None else ""

        document_items.sort(key=get_sort_key, reverse=descending)

        paginated_items = document_items[skip : skip + limit]
        return paginated_items, total_count

    def get_document(self, document_id: str) -> DocumentDetailResponse:
        """
        Gathers aggregated metadata from pipeline directories, excluding raw text arrays.
        """
        safe_doc_id = Path(document_id).name
        doc_dir = self.target_dir / safe_doc_id

        if not doc_dir.exists() or not doc_dir.is_dir():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document directory not found for document_id: {document_id}"
            )

        metadata = None
        status_data = None
        chunk_stats = None
        emb_meta = None
        idx_meta = None

        # Load metadata.json
        meta_path = doc_dir / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as err:
                logger.warning("Failed to load metadata.json for doc %s: %s", safe_doc_id, str(err))

        # Load status.json
        status_path = doc_dir / "status.json"
        if status_path.exists():
            try:
                with open(status_path, "r", encoding="utf-8") as f:
                    status_data = json.load(f)
            except Exception as err:
                logger.warning("Failed to load status.json for doc %s: %s", safe_doc_id, str(err))

        # Load chunk_statistics.json
        chunk_stats_path = doc_dir / "chunk_statistics.json"
        if chunk_stats_path.exists():
            try:
                with open(chunk_stats_path, "r", encoding="utf-8") as f:
                    chunk_stats = json.load(f)
            except Exception as err:
                logger.warning("Failed to load chunk_statistics.json for doc %s: %s", safe_doc_id, str(err))

        # Load embedding_metadata.json
        emb_meta_path = doc_dir / "embedding_metadata.json"
        if emb_meta_path.exists():
            try:
                with open(emb_meta_path, "r", encoding="utf-8") as f:
                    emb_meta = json.load(f)
            except Exception as err:
                logger.warning("Failed to load embedding_metadata.json for doc %s: %s", safe_doc_id, str(err))

        # Load index_metadata.json
        idx_meta_path = doc_dir / "index_metadata.json"
        if idx_meta_path.exists():
            try:
                with open(idx_meta_path, "r", encoding="utf-8") as f:
                    idx_meta = json.load(f)
            except Exception as err:
                logger.warning("Failed to load index_metadata.json for doc %s: %s", safe_doc_id, str(err))

        return DocumentDetailResponse(
            success=True,
            document_id=safe_doc_id,
            metadata=metadata,
            status=status_data,
            chunk_statistics=chunk_stats,
            embedding_metadata=emb_meta,
            index_metadata=idx_meta
        )

    def get_pipeline_status(self, document_id: str) -> PipelineStatusResponse:
        """
        Retrieves dedicated pipeline stage trackers.
        """
        safe_doc_id = Path(document_id).name
        status_path = self.target_dir / safe_doc_id / "status.json"

        if not status_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pipeline status not found for document_id: {document_id}"
            )

        try:
            with open(status_path, "r", encoding="utf-8") as f:
                status_data = json.load(f)
        except Exception as err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to read status: {str(err)}"
            )

        index_status = status_data.get("indexing_status", "pending")
        chat_ready = (index_status == "completed")

        return PipelineStatusResponse(
            success=True,
            document_id=safe_doc_id,
            upload_status=status_data.get("upload_status"),
            processing_status=status_data.get("processing_status"),
            chunking_status=status_data.get("chunking_status"),
            embedding_status=status_data.get("embedding_status"),
            indexing_status=index_status,
            chat_ready=chat_ready
        )

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Recursively deletes the uploads/<document_id> folder.
        """
        safe_doc_id = Path(document_id).name
        doc_dir = self.target_dir / safe_doc_id

        # Deletion is idempotent
        if not doc_dir.exists():
            logger.info("Delete document called for non-existing directory '%s'. Returning success.", safe_doc_id)
            return {"success": True, "message": "Document deleted successfully."}

        logger.info("Deleting document folder recursively: '%s'", doc_dir)
        try:
            shutil.rmtree(doc_dir)
            logger.info("Directory recursively pruned successfully.")
        except Exception as err:
            logger.exception("Failed to recursively delete document path '%s': %s", doc_dir, str(err))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to remove document files: {str(err)}"
            )

        # Recalculate system statistics atomically after delete
        try:
            self.calculate_storage_statistics()
        except Exception as stats_err:
            logger.warning("Failed to refresh statistics following delete: %s", str(stats_err))

        return {"success": True, "message": "Document deleted successfully."}

    def calculate_storage_statistics(self) -> Dict[str, Any]:
        """
        Walks uploads directory, calculates metrics, and persists statistics atomically.
        """
        stats_path = self.target_dir / "system_statistics.json"

        total_docs = 0
        completed_docs = 0
        failed_docs = 0
        processing_docs = 0
        total_pages = 0
        total_chunks = 0
        total_embeddings = 0
        storage_bytes = 0

        # Calculate bytes recursively
        def get_dir_size(path: Path) -> int:
            total = 0
            try:
                for entry in os.scandir(str(path)):
                    if entry.is_file():
                        total += entry.stat().st_size
                    elif entry.is_dir():
                        total += get_dir_size(Path(entry.path))
            except Exception:
                pass
            return total

        storage_bytes = get_dir_size(self.target_dir)

        # Gather metadata of folders
        for child in self.target_dir.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue

            pdf_path = child / "original.pdf"
            if not pdf_path.exists():
                continue

            total_docs += 1
            status_path = child / "status.json"
            metadata_path = child / "metadata.json"
            chunks_path = child / "chunks.json"
            emb_meta_path = child / "embedding_metadata.json"

            # Parse status
            if status_path.exists():
                try:
                    with open(status_path, "r", encoding="utf-8") as sf:
                        status_data = json.load(sf)
                    
                    indexing_status = status_data.get("indexing_status", "pending")
                    
                    if (
                        status_data.get("upload_status") == "failed" or
                        status_data.get("processing_status") == "failed" or
                        status_data.get("chunking_status") == "failed" or
                        status_data.get("embedding_status") == "failed" or
                        indexing_status == "failed"
                    ):
                        failed_docs += 1
                    elif indexing_status == "completed":
                        completed_docs += 1
                    elif (
                        status_data.get("processing_status") == "processing" or
                        status_data.get("chunking_status") == "processing" or
                        status_data.get("embedding_status") == "processing" or
                        indexing_status == "processing"
                    ):
                        processing_docs += 1
                    else:
                        processing_docs += 1
                except Exception:
                    processing_docs += 1
            else:
                processing_docs += 1

            # Sum pages
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r", encoding="utf-8") as mf:
                        meta = json.load(mf)
                    total_pages += meta.get("total_pages", 0)
                except Exception:
                    pass

            # Sum chunks
            if chunks_path.exists():
                try:
                    with open(chunks_path, "r", encoding="utf-8") as cf:
                        chunks_data = json.load(cf)
                    total_chunks += len(chunks_data)
                except Exception:
                    pass

            # Sum embeddings
            if emb_meta_path.exists():
                try:
                    with open(emb_meta_path, "r", encoding="utf-8") as emf:
                        emb_data = json.load(emf)
                    total_embeddings += emb_data.get("total_chunks") or emb_data.get("embedding_count", 0)
                except Exception:
                    pass

        stats_payload = {
            "total_documents": total_docs,
            "completed_documents": completed_docs,
            "failed_documents": failed_docs,
            "processing_documents": processing_docs,
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "total_embeddings": total_embeddings,
            "total_indexes": completed_docs, # Every completed document compiles 1 index.faiss
            "storage_bytes": storage_bytes,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        }

        self._write_atomic(stats_path, stats_payload)
        logger.info("Global system_statistics.json updated atomically.")
        return stats_payload
