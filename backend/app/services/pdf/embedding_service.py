"""
Embedding Generation Service Layer.

Orchestrates loading chunks, calling the vector provider, performing
rigorous validation, and persisting NumPy arrays, metadata, and status updates.
Implements atomic file writing and multi-state pipeline transitions.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from fastapi import HTTPException, status

from app.core.config import settings
from app.schemas.embedding import EmbeddingResponse
from app.services.embeddings.provider import EmbeddingProvider
from app.services.embeddings.sentence_transformer_provider import SentenceTransformerProvider
from app.services.pdf.pipeline_validator import PipelineLockManager, validate_chunk_artifacts, validate_embedding_artifacts

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service responsible for coordinating vector embedding generation pipelines
    for document chunk collections.
    """

    def __init__(self, provider: EmbeddingProvider | None = None, target_dir: Path | None = None):
        """
        Initialize the service. Falls back to default provider and target directories if omitted.
        """
        if provider is None:
            self.provider = SentenceTransformerProvider()
        else:
            self.provider = provider

        if target_dir is None:
            self.target_dir = Path(__file__).resolve().parent.parent.parent.parent / settings.UPLOAD_DIRECTORY
        else:
            self.target_dir = target_dir

    def _write_atomic(self, target_path: Path, content: bytes | str | np.ndarray, is_numpy: bool = False) -> None:
        """
        Atomically writes content to a target file by writing to a temporary file first,
        flushing to disk, and then replacing the target path atomically.
        """
        temp_path = target_path.with_suffix(".tmp")
        try:
            if is_numpy:
                if not isinstance(content, np.ndarray):
                    raise ValueError("Content must be a NumPy ndarray when is_numpy is True.")
                with open(temp_path, "wb") as tf:
                    np.save(tf, content)
                    tf.flush()
                    os.fsync(tf.fileno())
            else:
                with open(temp_path, "w", encoding="utf-8") as tf:
                    if isinstance(content, str):
                        tf.write(content)
                    else:
                        json.dump(content, tf, indent=4)
                    tf.flush()
                    os.fsync(tf.fileno())

            # Atomic rename replacement
            os.replace(temp_path, target_path)

        except Exception as exc:
            # Clean up temp file on failure
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

        status_data["embedding_status"] = new_status
        status_data["updated_at"] = now_str

        if extra_fields:
            status_data.update(extra_fields)

        self._write_atomic(status_path, status_data)
        logger.info("Pipeline state updated to '%s' in status.json", new_status)
        return status_data

    async def generate_document_embeddings(self, document_id: str, force: bool = False) -> EmbeddingResponse:
        """
        Generates and saves dense vector embeddings for a document's chunks.

        Args:
            document_id (str): Unique UUID document identifier.
            force (bool): Force regeneration of embeddings even if already completed.

        Returns:
            EmbeddingResponse: Summary payload metadata.
        """
        safe_doc_id = Path(document_id).name
        doc_dir = self.target_dir / safe_doc_id
        chunks_path = doc_dir / "chunks.json"
        status_path = doc_dir / "status.json"
        embeddings_path = doc_dir / "embeddings.npy"
        metadata_path = doc_dir / "embedding_metadata.json"
        statistics_path = doc_dir / "embedding_statistics.json"

        # 1. Locate Document Folder
        if not doc_dir.exists():
            detail_msg = f"Document directory not found for document_id: {document_id}"
            logger.warning("Embedding failed: %s", detail_msg)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=detail_msg
            )

        # Prevent concurrent duplicate execution for the same document ID
        if not PipelineLockManager.acquire_stage(safe_doc_id, "embed"):
            detail_msg = "Embedding generation is currently running for this document. Duplicate execution rejected."
            logger.warning(detail_msg)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=detail_msg
            )

        try:
            import asyncio
            await asyncio.sleep(0.05)

            # 2. Dependency Validation (Verify previous stage artifacts)
            chunks_valid, chunks_msg = validate_chunk_artifacts(doc_dir)
            if not chunks_valid:
                detail_msg = f"Chunking stage must be completed and valid before embedding. Reason: {chunks_msg}"
                logger.warning("Embedding dependency validation failed for document_id '%s': %s", safe_doc_id, detail_msg)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=detail_msg
                )

            # 3. Idempotency Check (Check if already embedded and valid)
            is_completed = False
            if status_path.exists():
                try:
                    with open(status_path, "r", encoding="utf-8") as sf:
                        status_data = json.load(sf)
                    if status_data.get("embedding_status") == "completed":
                        is_completed = True
                except Exception:
                    pass

            emb_valid, emb_msg = validate_embedding_artifacts(doc_dir)

            if not force and is_completed and emb_valid:
                logger.info("Embeddings already generated for document_id '%s'. Reusing valid existing artifacts.", safe_doc_id)
                try:
                    with open(metadata_path, "r", encoding="utf-8") as mf:
                        meta = json.load(mf)
                    return EmbeddingResponse(
                        success=True,
                        document_id=safe_doc_id,
                        embedding_model=meta.get("embedding_model", self.provider.model_name()),
                        embedding_dimension=meta.get("embedding_dimension", self.provider.dimension()),
                        total_embeddings=meta.get("embedding_count", 0),
                        processing_time_ms=meta.get("processing_time_ms", 0),
                        message="Embeddings generated successfully (cached result)."
                    )
                except Exception as e:
                    logger.warning("Failed to read embedding_metadata.json for document_id '%s': %s. Re-embedding...", safe_doc_id, str(e))

            logger.info("Embedding started for document_id: '%s'", safe_doc_id)
            self._update_status(status_path, "running")
            start_time = time.perf_counter()

            # Load chunks.json & Extract Text
            try:
                with open(chunks_path, "r", encoding="utf-8") as cf:
                    chunks_data = json.load(cf)
            except Exception as err:
                logger.exception("Failed to read chunks.json for document_id '%s'", safe_doc_id)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to load chunks: {str(err)}"
                )

            if not chunks_data:
                detail_msg = "chunks.json is empty. Cannot generate vector embeddings."
                logger.warning("Embedding failed for document_id '%s': %s", safe_doc_id, detail_msg)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=detail_msg
                )

            texts = []
            chunk_ids = set()

            for chunk in chunks_data:
                c_text = chunk.get("text")
                c_id = chunk.get("chunk_id")

                if not c_text or not c_text.strip():
                    detail_msg = f"Empty text chunk encountered at chunk_id: '{c_id}'."
                    logger.warning("Embedding validation failed for document_id '%s': %s", safe_doc_id, detail_msg)
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=detail_msg
                    )

                if c_id in chunk_ids:
                    detail_msg = f"Duplicate chunk_id '{c_id}' detected."
                    logger.warning("Embedding validation failed for document_id '%s': %s", safe_doc_id, detail_msg)
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=detail_msg
                    )

                chunk_ids.add(c_id)
                texts.append(c_text)

            logger.info("Chunks loaded successfully for document_id '%s' (%d chunks ready)", safe_doc_id, len(texts))

            # Transition state to "processing"
            self._update_status(status_path, "processing")

            try:
                # Triggers lazy loading singleton
                self.provider.model_name()
            except Exception as load_err:
                logger.exception("Model loading failed for model '%s': %s", settings.EMBEDDING_MODEL, str(load_err))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Embedding model loading failure: {str(load_err)}"
                )

            try:
                embeddings = self.provider.generate_embeddings(texts)
                logger.info("Embeddings generated for document_id '%s'", safe_doc_id)
            except Exception as gen_err:
                logger.exception("Embedding generation failed for document_id '%s': %s", safe_doc_id, str(gen_err))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Embedding generation failure: {str(gen_err)}"
                )

            # 4. Strengthen Vector Validation
            logger.info("Strengthening vector validation checks...")
            if embeddings.size == 0 or embeddings.ndim != 2:
                raise ValueError("Embedding matrix is empty or does not have exactly 2 dimensions.")

            if embeddings.shape[0] != len(texts):
                raise ValueError(f"Embedding count ({embeddings.shape[0]}) does not match text chunk count ({len(texts)}).")

            if embeddings.shape[1] != self.provider.dimension():
                raise ValueError(f"Vector dimensions ({embeddings.shape[1]}) do not match configuration ({self.provider.dimension()}).")

            if not np.issubdtype(embeddings.dtype, np.float32):
                raise ValueError(f"Vector dtype is {embeddings.dtype} instead of float32.")

            if np.isnan(embeddings).any() or np.isinf(embeddings).any():
                raise ValueError("Generated vector array contains NaN or Infinite values.")

            logger.info("Validation passed successfully: %d vectors checked.", len(embeddings))

            # 5. Persistence using atomic file writes
            try:
                # Save embeddings.npy
                self._write_atomic(embeddings_path, embeddings, is_numpy=True)
                logger.info("embeddings.npy saved atomically.")
            except Exception as save_err:
                logger.exception("Failed to write embeddings.npy atomically for document_id '%s'", safe_doc_id)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Atomic file write failed for embeddings: {str(save_err)}"
                )

            end_time = time.perf_counter()
            processing_time_ms = int(round((end_time - start_time) * 1000))
            created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            # Calculate Vector norms
            norms = np.linalg.norm(embeddings, axis=1)
            min_norm = float(np.min(norms))
            max_norm = float(np.max(norms))
            avg_norm = float(np.mean(norms))

            # Save embedding_metadata.json atomically
            meta_payload = {
                "document_id": safe_doc_id,
                "total_chunks": len(texts),
                "embedding_count": len(texts),
                "embedding_model": self.provider.model_name(),
                "embedding_dimension": self.provider.dimension(),
                "embedding_version": settings.EMBEDDING_VERSION,
                "normalized": settings.EMBEDDING_NORMALIZE,
                "created_at": created_at,
                "processing_time_ms": processing_time_ms
            }
            self._write_atomic(metadata_path, meta_payload)
            logger.info("embedding_metadata.json saved atomically.")

            # Save embedding_statistics.json atomically
            stats_payload = {
                "document_id": safe_doc_id,
                "embedding_count": len(texts),
                "embedding_dimension": self.provider.dimension(),
                "minimum_vector_norm": round(min_norm, 6),
                "maximum_vector_norm": round(max_norm, 6),
                "average_vector_norm": round(avg_norm, 6),
                "total_processing_time_ms": processing_time_ms
            }
            self._write_atomic(statistics_path, stats_payload)
            logger.info("embedding_statistics.json saved atomically.")

            # Transition state to "completed" in status.json
            self._update_status(
                status_path=status_path,
                new_status="completed",
                extra_fields={
                    "embedding_model": self.provider.model_name(),
                    "embedding_dimension": self.provider.dimension(),
                    "embedding_version": settings.EMBEDDING_VERSION,
                    "indexing_status": "pending" # Reset downstream stage
                }
            )
            logger.info("Embedding completed for document_id '%s' in %d ms.", safe_doc_id, processing_time_ms)

            return EmbeddingResponse(
                success=True,
                document_id=safe_doc_id,
                embedding_model=self.provider.model_name(),
                embedding_dimension=self.provider.dimension(),
                total_embeddings=len(texts),
                processing_time_ms=processing_time_ms,
                message="Embeddings generated successfully."
            )

        except Exception as exc:
            # Transition state to "failed" on any exception during processing
            self._update_status(status_path, "failed")
            logger.exception("Unexpected exception during embedding pipeline execution for document_id '%s'", safe_doc_id)
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred while generating embeddings: {str(exc)}"
            )
        finally:
            PipelineLockManager.release_stage(safe_doc_id, "embed")
