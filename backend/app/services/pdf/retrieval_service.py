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
            detail_msg = "FAISS index file (index.faiss) is missing. Document index is missing."
            logger.error("Index validation failed: %s", detail_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=detail_msg
            )

        embeddings_path = index_path.parent / "embeddings.npy"
        if not embeddings_path.exists():
            detail_msg = "Embeddings binary file (embeddings.npy) is missing. Document embeddings are missing."
            logger.error("Embeddings validation failed: %s", detail_msg)
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

    def retrieve(
        self,
        document_id: str,
        query: str,
        top_k: int = settings.DEFAULT_TOP_K,
        request_id: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Performs vector similarity search on document vectors returning enriched chunk metadata
        with raw similarity score precision. Suitable for direct integration with Chat/RAG services.

        Args:
            document_id (str): Unique UUID document identifier.
            query (str): Input user search query string.
            top_k (int): Total results to retrieve.
            request_id (str): Optional request ID for correlation across services.

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

        total_start = time.perf_counter()

        # 5. Load and Validate index
        self._validate_index(index_path, embedding_metadata_path, total_chunks)

        logger.info("Retrieval started for document_id '%s' (Query: '%s', top_k=%d)", safe_doc_id, query, top_k)

        # 6. Generate Query Embedding using existing singleton
        embed_start = time.perf_counter()
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
        embed_time_ms = int(round((time.perf_counter() - embed_start) * 1000))

        # 7. Perform Semantic Search on persisting FAISS index
        faiss_start = time.perf_counter()
        try:
            # Query vector is L2 normalized within FaissRetrievalProvider search execution
            search_matches = self.provider.search(query_embedding, top_k, index_path)
        except Exception as err:
            logger.exception("Similarity search execution failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Index similarity search failed: {str(err)}"
            )
        faiss_time_ms = int(round((time.perf_counter() - faiss_start) * 1000))

        filter_start = time.perf_counter()
        initial_candidates = []
        seen_chunk_ids = set()
        seen_texts = set()

        # 8 & 9. Map matches to Chunks, filter duplicates
        for score, vector_idx in search_matches:
            if vector_idx < 0 or vector_idx >= total_chunks:
                logger.warning("Vector index match '%d' is out of bounds for chunk collection size %d", vector_idx, total_chunks)
                continue

            chunk = chunks_data[vector_idx]
            chunk_id = chunk.get("chunk_id")
            text_content = chunk.get("text", "")
            
            # Normalize text content for deduplication
            norm_text = " ".join(text_content.lower().split())

            # Check for duplicate vector index matches
            if chunk_id in seen_chunk_ids:
                continue

            # Check for duplicate text content to avoid returning identical text twice
            if norm_text in seen_texts:
                continue

            seen_chunk_ids.add(chunk_id)
            seen_texts.add(norm_text)
            
            initial_candidates.append((score, chunk))

        # Filter scores below similarity threshold (absolute check)
        valid_candidates = [cand for cand in initial_candidates if cand[0] >= settings.MIN_SIMILARITY_SCORE]

        # Apply Adaptive Top-K Retrieval
        filtered_candidates = []
        if valid_candidates:
            # Sort by score descending (already sorted from search, but to be sure)
            valid_candidates.sort(key=lambda x: x[0], reverse=True)
            highest_score = valid_candidates[0][0]
            adaptive_threshold = highest_score - settings.ADAPTIVE_SCORE_DROP_LIMIT
            
            filtered_candidates = [
                cand for cand in valid_candidates 
                if cand[0] >= adaptive_threshold and cand[0] >= settings.MIN_SIMILARITY_SCORE
            ]
            
            # Always return at least one chunk if any match exists
            if not filtered_candidates:
                filtered_candidates = [valid_candidates[0]]

        filtered_results = []
        rank_counter = 1
        for score, chunk in filtered_candidates:
            # Defensive check: ensure chunk actually belongs to the requested document
            chunk_doc_id = chunk.get("document_id")
            if chunk_doc_id and chunk_doc_id != safe_doc_id:
                logger.error("Security boundary violation: chunk document_id '%s' does not match safe_doc_id '%s'", chunk_doc_id, safe_doc_id)
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: Retrieval candidate belongs to a different document."
                )

            filtered_results.append(
                RetrievalResult(
                    chunk_id=chunk.get("chunk_id"),
                    chunk_index=chunk.get("chunk_index") or 1,
                    rank=rank_counter,
                    score=score,
                    start_page=chunk.get("start_page", 1),
                    end_page=chunk.get("end_page", 1),
                    start_character=chunk.get("start_character", 0),
                    end_character=chunk.get("end_character", 0),
                    sentence_count=chunk.get("sentence_count", 0),
                    estimated_tokens=chunk.get("estimated_tokens", 0),
                    character_count=chunk.get("character_count", 0),
                    word_count=chunk.get("word_count", 0),
                    text=chunk.get("text", ""),
                    document_id=safe_doc_id,
                    last_chunk_index=None
                )
            )
            rank_counter += 1
            
        filter_time_ms = int(round((time.perf_counter() - filter_start) * 1000))

        # Merge Neighboring Chunks
        merge_start = time.perf_counter()
        # Sort by chunk_index ascending to find sequential neighbors
        sorted_results = sorted(filtered_results, key=lambda x: x.chunk_index)
        merged_results = []
        
        for res in sorted_results:
            if not merged_results:
                merged_results.append(res)
            else:
                last_res = merged_results[-1]
                # Check merge conditions: sequential indices AND same page OR adjacent page
                current_last_idx = (
                    last_res.last_chunk_index
                    if last_res.last_chunk_index is not None
                    else last_res.chunk_index
                )
                index_diff = abs(res.chunk_index - current_last_idx)
                page_overlap = (
                    abs(res.start_page - last_res.end_page) <= 1 or
                    abs(res.end_page - last_res.start_page) <= 1
                )
                
                if index_diff == 1 and page_overlap:
                    # Merge current res into last_res
                    last_res.text = last_res.text + "\n\n" + res.text
                    last_res.last_chunk_index = res.chunk_index
                    last_res.start_page = min(last_res.start_page, res.start_page)
                    last_res.end_page = max(last_res.end_page, res.end_page)
                    last_res.score = max(last_res.score, res.score)
                    
                    # Update other metadata fields
                    last_res.sentence_count += res.sentence_count
                    last_res.estimated_tokens += res.estimated_tokens
                    last_res.character_count = len(last_res.text)
                    last_res.word_count = len(last_res.text.split())
                    last_res.start_character = min(last_res.start_character, res.start_character)
                    last_res.end_character = max(last_res.end_character, res.end_character)
                else:
                    merged_results.append(res)

        # Re-sort back by similarity score descending for ranking, and reassign rank counters
        merged_results.sort(key=lambda x: x.score, reverse=True)
        for i, res in enumerate(merged_results, start=1):
            res.rank = i
            
        merge_time_ms = int(round((time.perf_counter() - merge_start) * 1000))
        total_time_ms = int(round((time.perf_counter() - total_start) * 1000))

        # Performance timing logs
        logger.info(
            "Embedding Time : %d ms\n"
            "FAISS Search : %d ms\n"
            "Filtering : %d ms\n"
            "Merge : %d ms\n"
            "Total Retrieval : %d ms",
            embed_time_ms,
            faiss_time_ms,
            filter_time_ms,
            merge_time_ms,
            total_time_ms
        )

        # Write to debug_info.json (Requirement 5)
        debug_path = doc_dir / "debug_info.json"
        highest_similarity = float(max(res.score for res in merged_results)) if merged_results else 0.0
        minimum_similarity = float(min(res.score for res in merged_results)) if merged_results else 0.0
        
        debug_entry = {
            "request_id": request_id,
            "question": query,
            "embedding_time_ms": embed_time_ms,
            "faiss_time_ms": faiss_time_ms,
            "filter_time_ms": filter_time_ms,
            "merge_time_ms": merge_time_ms,
            "total_time_ms": total_time_ms,
            "top_k_requested": top_k,
            "top_k_returned": len(merged_results),
            "highest_similarity": highest_similarity,
            "minimum_similarity": minimum_similarity,
            "retrieved_chunks": [res.model_dump() for res in merged_results]
        }
        
        debug_list = []
        if debug_path.exists():
            try:
                with open(debug_path, "r", encoding="utf-8") as df:
                    debug_list = json.load(df)
                    if not isinstance(debug_list, list):
                        debug_list = []
            except Exception:
                debug_list = []
                
        debug_list.append(debug_entry)
        
        # Enforce log retention limits (Milestone 13 Phase 3)
        from app.core.telemetry import prune_debug_list
        debug_list = prune_debug_list(debug_list)
        
        try:
            self._write_atomic(debug_path, debug_list)
        except Exception as err:
            logger.warning("Failed to save debug_info.json: %s", str(err))

        if not merged_results:
            logger.warning("No relevant chunks found matching query '%s' above similarity threshold of %.2f.", query, settings.MIN_SIMILARITY_SCORE)

        logger.info("Similarity search executed. Found %d matches above threshold.", len(merged_results))
        return merged_results

    async def query_document(
        self,
        document_id: str,
        request: RetrievalRequest,
        request_id: Optional[str] = None
    ) -> RetrievalResponse:
        """
        Coordinates full retrieval query pipeline including stats recording, rounded score formatting,
        and returning a RetrievalResponse payload.
        """
        safe_doc_id = Path(document_id).name
        stats_path = self.target_dir / safe_doc_id / "retrieval_statistics.json"

        start_time = time.perf_counter()

        # Run core retrieve
        raw_results = self.retrieve(document_id, request.query, request.top_k, request_id=request_id)

        if not raw_results:
            detail_msg = f"No relevant chunks found matching query '{request.query}' above similarity threshold of {settings.MIN_SIMILARITY_SCORE}."
            logger.warning(detail_msg)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=detail_msg
            )

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

    def get_debug_info(self, document_id: str) -> List[dict]:
        """
        Loads the debug_info.json file for a given document_id.
        """
        safe_doc_id = Path(document_id).name
        doc_dir = self.target_dir / safe_doc_id
        
        if not doc_dir.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document directory not found for document_id: {document_id}"
            )
            
        debug_path = doc_dir / "debug_info.json"
        if not debug_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Debug info log file does not exist for document_id: {document_id}"
            )
            
        try:
            with open(debug_path, "r", encoding="utf-8") as df:
                return json.load(df)
        except Exception as err:
            logger.exception("Failed to read debug_info.json")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load debug info: {str(err)}"
            )
