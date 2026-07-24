"""
Semantic Retrieval Engine Service Layer.

Provides business logic for querying local vector indices, validating pipeline stages,
mapping vector matches to chunk metadata, filtering, ranking, and writing retrieval statistics.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from fastapi import HTTPException, status

from app.core.config import settings
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse, RetrievalResult
from app.services.embeddings.sentence_transformer_provider import SentenceTransformerProvider
from app.services.retrieval.provider import RetrievalProvider
from app.services.retrieval.faiss_provider import FaissRetrievalProvider

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Service responsible for orchestrating semantic similarity retrieval queries
    against persisted vector indices.
    """

    def __init__(self, provider: RetrievalProvider | None = None, target_dir: Path | None = None):
        """
        Initialize the RetrievalService.
        """
        if provider is None:
            self.provider = FaissRetrievalProvider()
        else:
            self.provider = provider

        if target_dir is None:
            self.target_dir = Path(__file__).resolve().parent.parent.parent.parent / settings.UPLOAD_DIRECTORY
        else:
            self.target_dir = target_dir

        self.embedding_provider = SentenceTransformerProvider()

    def _write_atomic(self, target_path: Path, content: dict) -> None:
        """
        Atomically writes retrieval statistics dictionary to disk.
        """
        temp_path = target_path.with_suffix(".tmp")
        try:
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
            logger.exception("Atomic file write failed for retrieval statistics: %s", str(exc))
            raise

    # 3. Private Validation Helpers
    def _validate_pipeline(self, status_path: Path) -> dict:
        """
        Validates pipeline completion status.
        """
        if not status_path.exists():
            detail_msg = "Pipeline status file (status.json) is missing."
            logger.error("Pipeline validation failed: %s", detail_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=detail_msg
            )

        try:
            with open(status_path, "r", encoding="utf-8") as sf:
                status_data = json.load(sf)
        except Exception as err:
            logger.exception("Failed to read status.json")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to read status tracker: {str(err)}"
            )

        required_stages = [
            "upload_status",
            "processing_status",
            "chunking_status",
            "embedding_status",
            "indexing_status"
        ]
        for stage in required_stages:
            if status_data.get(stage) != "completed":
                detail_msg = f"Document pipeline stage '{stage}' is incomplete."
                logger.error("Pipeline validation failed: %s", detail_msg)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=detail_msg
                )

        return status_data

    def _validate_request(self, query: str, top_k: int) -> None:
        """
        Validates query parameters and boundary constraints.
        """
        if not query or not query.strip():
            detail_msg = "Search query string cannot be empty."
            logger.error("Request validation failed: %s", detail_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=detail_msg
            )

        if len(query) > settings.MAX_QUERY_LENGTH:
            detail_msg = f"Search query string exceeds maximum length of {settings.MAX_QUERY_LENGTH} characters."
            logger.error("Request validation failed: %s", detail_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=detail_msg
            )

        if top_k <= 0 or top_k > settings.MAX_TOP_K:
            detail_msg = f"Parameter top_k must be between 1 and {settings.MAX_TOP_K} inclusive."
            logger.error("Request validation failed: %s", detail_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=detail_msg
            )

    def _validate_index(self, index_path: Path, meta_path: Path, expected_chunks: int) -> dict:
        """
        Validates consistency between FAISS index, chunks count, and embedding metadata.
        """
        if not index_path.exists():
            detail_msg = "FAISS index file (index.faiss) is missing."
            logger.error("Index validation failed: %s", detail_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=detail_msg
            )

        if not meta_path.exists():
            detail_msg = "Embedding metadata file (embedding_metadata.json) is missing."
            logger.error("Index validation failed: %s", detail_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=detail_msg
            )

        try:
            with open(meta_path, "r", encoding="utf-8") as mf:
                meta_data = json.load(mf)
        except Exception as err:
            logger.exception("Failed to read embedding metadata")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to read embedding metadata: {str(err)}"
            )

        # Retrieve direct properties from index provider
        try:
            is_valid_index = self.provider.validate(index_path)
            if not is_valid_index:
                raise ValueError("Provider index validation failed.")
        except Exception as err:
            detail_msg = "Index and metadata are inconsistent."
            logger.error("Index validation failed: %s (Detail: %s)", detail_msg, str(err))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=detail_msg
            )

        # Cross check counts
        emb_count = meta_data.get("embedding_count")
        emb_dim = meta_data.get("embedding_dimension")

        if emb_dim != settings.EMBEDDING_DIMENSION:
            logger.error("Dimension mismatch: metadata = %s, configured = %s", emb_dim, settings.EMBEDDING_DIMENSION)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Index and metadata are inconsistent."
            )

        if emb_count != expected_chunks:
            logger.error("Count mismatch: metadata embedding_count = %s, chunks_count = %s", emb_count, expected_chunks)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Index and metadata are inconsistent."
            )

        return meta_data

    def _validate_chunks(self, chunks: list) -> None:
        """
        Validates populated chunk array structure.
        """
        if not chunks:
            detail_msg = "Parsed chunks list is empty."
            logger.error("Chunks validation failed: %s", detail_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=detail_msg
            )

    def retrieve(self, document_id: str, query: str, top_k: int = settings.DEFAULT_TOP_K) -> List[RetrievalResult]:
        """
        Performs vector similarity search on document vectors returning enriched chunk metadata
        with raw similarity score precision. Suitable for direct integration with Chat/RAG services.

        Args:
            document_id (str): Unique UUID document identifier.
            query (str): Input user search query string.
            top_k (int): Total results to retrieve.

        Returns:
            List[RetrievalResult]: Sorted list of retrieval result objects.
        """
        safe_doc_id = Path(document_id).name
        doc_dir = self.target_dir / safe_doc_id
        status_path = doc_dir / "status.json"
        chunks_path = doc_dir / "chunks.json"
        index_path = doc_dir / "index.faiss"
        embedding_metadata_path = doc_dir / "embedding_metadata.json"

        # 1. Locate Document Folder
        if not doc_dir.exists():
            detail_msg = f"Document directory not found for document_id: {document_id}"
            logger.error("Retrieval failed: %s", detail_msg)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=detail_msg
            )

        # 2. Pipeline State Check
        self._validate_pipeline(status_path)

        # 3. Request Boundary Validation
        self._validate_request(query, top_k)

        # 4. Load Chunk Collection
        if not chunks_path.exists():
            detail_msg = f"Missing chunks.json for document_id: {document_id}"
            logger.error("Retrieval failed: %s", detail_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=detail_msg
            )

        try:
            with open(chunks_path, "r", encoding="utf-8") as cf:
                chunks_data = json.load(cf)
        except Exception as err:
            logger.exception("Failed to read chunks.json for document_id '%s'", safe_doc_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load document text chunks: {str(err)}"
            )

        self._validate_chunks(chunks_data)
        total_chunks = len(chunks_data)

        # 5. Load and Validate index
        self._validate_index(index_path, embedding_metadata_path, total_chunks)

        logger.info("Retrieval started for document_id '%s' (Query: '%s', top_k=%d)", safe_doc_id, query, top_k)

        # 6. Generate Query Embedding using existing singleton
        try:
            # SentenceTransformer model generates float32 array
            query_embedding = self.embedding_provider.generate_embeddings([query])[0]
            logger.info("Query embedding generated successfully.")
        except Exception as err:
            logger.exception("Embedding provider failed to encode query")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query embedding generation failed: {str(err)}"
            )

        # 7. Perform Semantic Search on persisting FAISS index
        try:
            # Query vector is L2 normalized within FaissRetrievalProvider search execution
            search_matches = self.provider.search(query_embedding, top_k, index_path)
        except Exception as err:
            logger.exception("Similarity search execution failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Index similarity search failed: {str(err)}"
            )

        results: List[RetrievalResult] = []
        seen_chunk_ids = set()
        rank_counter = 1

        # 8 & 9. Map matches to Chunks, filter duplicates and low similarity thresholds
        for score, vector_idx in search_matches:
            if vector_idx < 0 or vector_idx >= total_chunks:
                logger.warning("Vector index match '%d' is out of bounds for chunk collection size %d", vector_idx, total_chunks)
                continue

            chunk = chunks_data[vector_idx]
            chunk_id = chunk.get("chunk_id")

            # Check for duplicate vector index matches
            if chunk_id in seen_chunk_ids:
                continue

            # Filter scores below similarity threshold
            if score < settings.MIN_SIMILARITY_SCORE:
                continue

            seen_chunk_ids.add(chunk_id)

            # Preserve raw float score in the created RetrievalResult object
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    chunk_index=chunk.get("chunk_index") or vector_idx + 1,
                    rank=rank_counter,
                    score=score,  # Raw float score preserved internally
                    start_page=chunk.get("start_page", 1),
                    end_page=chunk.get("end_page", 1),
                    start_character=chunk.get("start_character", 0),
                    end_character=chunk.get("end_character", 0),
                    sentence_count=chunk.get("sentence_count", 0),
                    estimated_tokens=chunk.get("estimated_tokens", 0),
                    character_count=chunk.get("character_count", 0),
                    word_count=chunk.get("word_count", 0),
                    text=chunk.get("text", "")
                )
            )
            rank_counter += 1

        if not results:
            logger.warning("No chunks matched similarity score threshold: %.2f", settings.MIN_SIMILARITY_SCORE)

        # 10. Sort results by similarity score descending and re-rank sequentially
        results.sort(key=lambda x: x.score, reverse=True)
        for i, res in enumerate(results, start=1):
            res.rank = i

        logger.info("Similarity search executed. Found %d matches above threshold.", len(results))
        return results

    async def query_document(self, document_id: str, request: RetrievalRequest) -> RetrievalResponse:
        """
        Coordinates full retrieval query pipeline including stats recording, rounded score formatting,
        and returning a RetrievalResponse payload.
        """
        safe_doc_id = Path(document_id).name
        stats_path = self.target_dir / safe_doc_id / "retrieval_statistics.json"

        start_time = time.perf_counter()

        # Run core retrieve
        raw_results = self.retrieve(document_id, request.query, request.top_k)

        end_time = time.perf_counter()
        processing_time_ms = int(round((end_time - start_time) * 1000))

        # Extract stats scores
        scores = [res.score for res in raw_results]
        highest_score = float(max(scores)) if scores else 0.0
        lowest_score = float(min(scores)) if scores else 0.0
        avg_score = float(sum(scores) / len(scores)) if scores else 0.0

        # Save retrieval_statistics.json atomically
        stats_payload = {
            "retrieval_version": settings.RETRIEVAL_VERSION,
            "query_length": len(request.query),
            "requested_top_k": request.top_k,
            "returned_results": len(raw_results),
            "minimum_similarity_threshold": settings.MIN_SIMILARITY_SCORE,
            "processing_time_ms": processing_time_ms
        }
        try:
            self._write_atomic(stats_path, stats_payload)
            logger.info("retrieval_statistics.json saved atomically.")
        except Exception as err:
            logger.warning("Failed to save retrieval_statistics.json: %s", str(err))

        # Round scores to 4 decimal places *only* in response payload
        formatted_results = []
        for res in raw_results:
            formatted_res = res.model_copy()
            formatted_res.score = round(res.score, 4)
            formatted_results.append(formatted_res)

        logger.info(
            "Query matching completed. Returned %d chunks in %d ms.",
            len(formatted_results),
            processing_time_ms
        )

        return RetrievalResponse(
            success=True,
            document_id=safe_doc_id,
            query=request.query,
            total_results=len(formatted_results),
            processing_time_ms=processing_time_ms,
            retrieval_version=settings.RETRIEVAL_VERSION,
            results=formatted_results
        )
