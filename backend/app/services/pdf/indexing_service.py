"""
Vector Indexing Service Layer.

Orchestrates loading embeddings, calling the IndexProvider, validation checks,
and atomic persistence of FAISS indices, metadata, and pipeline statuses.
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from fastapi import HTTPException, status

from app.core.config import settings
from app.schemas.indexing import IndexingResponse
from app.services.indexing.provider import IndexProvider
from app.services.indexing.faiss_provider import FaissIndexProvider
from app.services.pdf.pipeline_validator import PipelineLockManager, validate_embedding_artifacts, validate_indexing_artifacts

logger = logging.getLogger(__name__)


class IndexingService:
    """
    Service responsible for building and validating vector indices.
    """

    def __init__(self, provider: IndexProvider | None = None, target_dir: Path | None = None):
        """
        Initialize the service. Falls back to local FAISS provider and target directories if omitted.
        """
        if provider is None:
            self.provider = FaissIndexProvider()
        else:
            self.provider = provider

        if target_dir is None:
            self.target_dir = Path(__file__).resolve().parent.parent.parent.parent / settings.UPLOAD_DIRECTORY
        else:
            self.target_dir = target_dir

    def _write_atomic(self, target_path: Path, content: any, is_faiss: bool = False) -> None:
        """
        Atomically writes file content to disk. Reuses temp swap pattern to prevent corruption.
        """
        temp_path = target_path.with_suffix(".tmp")
        try:
            if is_faiss:
                self.provider.save(content, temp_path)
            else:
                with open(temp_path, "w", encoding="utf-8") as tf:
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

        status_data["indexing_status"] = new_status
        status_data["updated_at"] = now_str

        if extra_fields:
            status_data.update(extra_fields)

        self._write_atomic(status_path, status_data)
        logger.info("Pipeline state updated to '%s' in status.json", new_status)
        return status_data

    async def generate_document_index(self, document_id: str, force: bool = False) -> IndexingResponse:
        """
        Generates and persists a local FAISS index for a document's embedding vectors.

        Args:
            document_id (str): Unique UUID document identifier.
            force (bool): Force index regeneration even if already completed.

        Returns:
            IndexingResponse: Summary response object.
        """
        safe_doc_id = Path(document_id).name
        doc_dir = self.target_dir / safe_doc_id
        embeddings_path = doc_dir / "embeddings.npy"
        chunks_path = doc_dir / "chunks.json"
        status_path = doc_dir / "status.json"
        embedding_metadata_path = doc_dir / "embedding_metadata.json"
        index_faiss_path = doc_dir / "index.faiss"
        metadata_path = doc_dir / "index_metadata.json"
        statistics_path = doc_dir / "index_statistics.json"

        # 1. Locate Document Folder
        if not doc_dir.exists():
            detail_msg = f"Document directory not found for document_id: {document_id}"
            logger.warning("Indexing failed: %s", detail_msg)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=detail_msg
            )

        # Prevent concurrent duplicate execution for the same document ID
        if not PipelineLockManager.acquire_stage(safe_doc_id, "index"):
            detail_msg = "Vector indexing is currently running for this document. Duplicate execution rejected."
            logger.warning(detail_msg)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=detail_msg
            )

        try:
            import asyncio
            await asyncio.sleep(0.05)

            # 2. Dependency Validation (Verify previous stage artifacts)
            embeds_valid, embeds_msg = validate_embedding_artifacts(doc_dir)
            if not embeds_valid:
                detail_msg = f"Embedding stage must be completed and valid before indexing. Reason: {embeds_msg}"
                logger.warning("Indexing dependency validation failed for document_id '%s': %s", safe_doc_id, detail_msg)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=detail_msg
                )

            # 3. Idempotency Check (Check if already indexed and valid)
            is_completed = False
            if status_path.exists():
                try:
                    with open(status_path, "r", encoding="utf-8") as sf:
                        status_data = json.load(sf)
                    if status_data.get("indexing_status") == "completed":
                        is_completed = True
                except Exception:
                    pass

            index_valid, index_msg = validate_indexing_artifacts(doc_dir)

            if not force and is_completed and index_valid:
                logger.info("Vector index already created for document_id '%s'. Reusing valid existing artifacts.", safe_doc_id)
                try:
                    with open(metadata_path, "r", encoding="utf-8") as mf:
                        meta = json.load(mf)
                    return IndexingResponse(
                        success=True,
                        document_id=safe_doc_id,
                        index_type=meta.get("index_type", self.provider.index_type()),
                        vector_dimension=meta.get("vector_dimension", settings.EMBEDDING_DIMENSION),
                        indexed_vectors=meta.get("indexed_vectors", 0),
                        processing_time_ms=meta.get("processing_time_ms", 0),
                        message="Vector index created successfully (cached result)."
                    )
                except Exception as e:
                    logger.warning("Failed to read index_metadata.json for document_id '%s': %s. Re-indexing...", safe_doc_id, str(e))

            request_id = str(uuid.uuid4())
            logger.info("[STAGE: INDEX] request_id=%s document_id='%s' status=started", request_id, safe_doc_id)
            self._update_status(status_path, "running")
            start_time = time.perf_counter()

            # 4. Load input files
            try:
                embeddings = np.load(embeddings_path)
            except Exception as load_err:
                logger.exception("Failed to load embeddings.npy for document_id '%s'", safe_doc_id)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Corrupted embedding file: {str(load_err)}"
                )

            try:
                with open(chunks_path, "r", encoding="utf-8") as cf:
                    chunks_data = json.load(cf)
            except Exception as load_err:
                logger.exception("Failed to load chunks.json for document_id '%s'", safe_doc_id)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Corrupted chunks file: {str(load_err)}"
                )

            # 5. Validate Inputs
            logger.info("Validating input embedding matrix and text chunk counts...")
            if embeddings.size == 0 or embeddings.ndim != 2:
                raise ValueError("Embedding matrix is empty or invalid shape.")

            if embeddings.shape[0] != len(chunks_data):
                raise ValueError(f"Embedding count ({embeddings.shape[0]}) does not match text chunk count ({len(chunks_data)}).")

            if embeddings.shape[1] != settings.EMBEDDING_DIMENSION:
                raise ValueError(f"Vector dimension ({embeddings.shape[1]}) does not match configuration ({settings.EMBEDDING_DIMENSION}).")

            if not np.issubdtype(embeddings.dtype, np.float32):
                raise ValueError(f"Vector array dtype is {embeddings.dtype} instead of float32.")

            if np.isnan(embeddings).any() or np.isinf(embeddings).any():
                raise ValueError("Embedding vectors contain NaN or Infinite values.")

            logger.info("Input validation completed successfully: %d vectors validated.", len(embeddings))

            # 6. Build FAISS Index
            logger.info("Building vector index...")
            try:
                index = self.provider.create_index(embeddings)
            except Exception as index_err:
                logger.exception("Provider failed to build index for document_id '%s'", safe_doc_id)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to build vector index: {str(index_err)}"
                )

            # 7. Save Index Atomically
            logger.info("Persisting FAISS index file atomically...")
            try:
                self._write_atomic(index_faiss_path, index, is_faiss=True)
            except Exception as write_err:
                logger.exception("Failed to save index file for document_id '%s'", safe_doc_id)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Index persistence failure: {str(write_err)}"
                )

            # 8. Integrity Validation: Reload and run search query
            logger.info("Performing index integrity validation reload check...")
            try:
                reloaded_index = self.provider.load(index_faiss_path)
                is_valid = self.provider.validate(reloaded_index, len(embeddings), settings.EMBEDDING_DIMENSION)
                if not is_valid:
                    raise ValueError("Index validation reload integrity check failed.")
            except Exception as reload_err:
                # Clean up corrupted index on validation failure
                if index_faiss_path.exists():
                    index_faiss_path.unlink()
                logger.exception("Validation reload failed for document_id '%s'", safe_doc_id)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Index integrity validation failed: {str(reload_err)}"
                )

            end_time = time.perf_counter()
            processing_time_ms = int(round((end_time - start_time) * 1000))
            created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            # Get file size of index.faiss
            index_size_bytes = index_faiss_path.stat().st_size if index_faiss_path.exists() else 0

            # 9. Save index_metadata.json
            meta_payload = {
                "document_id": safe_doc_id,
                "index_type": self.provider.index_type(),
                "distance_metric": settings.VECTOR_DISTANCE,
                "vector_dimension": self.provider.dimension(index),
                "indexed_vectors": self.provider.vector_count(index),
                "index_version": settings.INDEX_VERSION,
                "created_at": created_at,
                "processing_time_ms": processing_time_ms
            }
            self._write_atomic(metadata_path, meta_payload)
            logger.info("index_metadata.json saved atomically.")

            # 10. Save index_statistics.json
            stats_payload = {
                "document_id": safe_doc_id,
                "indexed_vectors": self.provider.vector_count(index),
                "vector_dimension": self.provider.dimension(index),
                "index_size_bytes": index_size_bytes,
                "processing_time_ms": processing_time_ms
            }
            self._write_atomic(statistics_path, stats_payload)
            logger.info("index_statistics.json saved atomically.")

            # 11. Update status.json with indexing completed
            self._update_status(
                status_path=status_path,
                new_status="completed",
                extra_fields={
                    "index_type": self.provider.index_type(),
                    "index_version": settings.INDEX_VERSION,
                    "chat_ready": True
                }
            )

            logger.info(
                "[STAGE: INDEX] request_id=%s document_id='%s' status=completed elapsed_ms=%d indexed_vectors=%d",
                request_id,
                safe_doc_id,
                processing_time_ms,
                self.provider.vector_count(index)
            )

            return IndexingResponse(
                success=True,
                document_id=safe_doc_id,
                index_type=self.provider.index_type(),
                vector_dimension=self.provider.dimension(index),
                indexed_vectors=self.provider.vector_count(index),
                processing_time_ms=processing_time_ms,
                message="Vector index created successfully."
            )

        except Exception as exc:
            # Transition state to "failed" on any exception
            self._update_status(status_path, "failed")
            logger.exception("Unexpected exception during indexing pipeline execution for document_id '%s'", safe_doc_id)
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred while generating the vector index: {str(exc)}"
            )
        finally:
            PipelineLockManager.release_stage(safe_doc_id, "index")
