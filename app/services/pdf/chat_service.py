"""
AI Chat (RAG) Engine Service Layer.

Orchestrates retrieving document segments, validating pipeline completions,
modular prompt compilation, querying abstract LLMs, and saving atomic analytics.
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import HTTPException, status

from app.core.config import settings
from app.schemas.chat import ChatRequest, ChatResponse, SourceChunk
from app.services.chat.provider import LLMProvider
from app.services.chat.groq_provider import GroqProvider
from app.services.chat.prompt_builder import PromptBuilder
from app.services.pdf.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


class ChatService:
    """
    Core RAG service coordinating stateless completions queries.
    """

    def __init__(self, provider: LLMProvider | None = None, target_dir: Path | None = None):
        """
        Initialize the ChatService.
        """
        if provider is None:
            self.provider = GroqProvider()
        else:
            self.provider = provider

        if target_dir is None:
            self.target_dir = Path(__file__).resolve().parent.parent.parent.parent / settings.UPLOAD_DIRECTORY
        else:
            self.target_dir = target_dir

        self.retrieval_service = RetrievalService(target_dir=self.target_dir)

    def _write_atomic(self, target_path: Path, content: dict) -> None:
        """
        Atomically writes chat statistics dictionary to disk.
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
            logger.exception("Atomic file write failed for chat statistics: %s", str(exc))
            raise

    def _validate_pipeline(self, status_path: Path) -> None:
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

    def _validate_request(self, question: str, top_k: int) -> None:
        """
        Validates question boundaries.
        """
        if not question or not question.strip():
            detail_msg = "Chat question cannot be empty."
            logger.error("Request validation failed: %s", detail_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=detail_msg
            )

        if len(question) > settings.MAX_QUERY_LENGTH:
            detail_msg = f"Chat question exceeds maximum length of {settings.MAX_QUERY_LENGTH} characters."
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

    async def answer_question(self, document_id: str, request: ChatRequest) -> ChatResponse:
        """
        Retrieves context segments and coordinates generation from the LLM provider.
        """
        # 1. Generate Request ID
        request_id = str(uuid.uuid4())
        safe_doc_id = Path(document_id).name
        doc_dir = self.target_dir / safe_doc_id
        status_path = doc_dir / "status.json"
        stats_path = doc_dir / "chat_statistics.json"

        logger.info(
            "[Request: %s] Grounded RAG query query received for document_id '%s'",
            request_id,
            safe_doc_id
        )
        overall_start = time.perf_counter()

        # 2 & 3. Validate Document Folder & Pipeline Completion
        if not doc_dir.exists():
            detail_msg = f"Document directory not found for document_id: {document_id}"
            logger.error("[Request: %s] %s", request_id, detail_msg)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=detail_msg
            )

        self._validate_pipeline(status_path)

        # 4. Validate Request
        self._validate_request(request.question, request.top_k)

        # 5. Context Retrieval (Using existing RetrievalService - NO DIRECT FAISS ACCESS)
        retrieval_start = time.perf_counter()
        try:
            # Query embedding and matching vectors executed entirely within retrieval service
            retrieved_chunks = self.retrieval_service.retrieve(
                document_id=safe_doc_id,
                query=request.question,
                top_k=request.top_k
            )
        except Exception as ret_err:
            logger.exception("[Request: %s] RetrievalService query execution failed", request_id)
            if isinstance(ret_err, HTTPException):
                raise
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Semantic chunk retrieval failed: {str(ret_err)}"
            )

        retrieval_time_ms = int(round((time.perf_counter() - retrieval_start) * 1000))
        logger.info(
            "[Request: %s] Retrieval completed in %d ms. Chunks matching score limit: %d",
            request_id,
            retrieval_time_ms,
            len(retrieved_chunks)
        )

        # 6. Empty Retrieval Optimization: Return fallback directly without LLM call
        fallback_msg = "I couldn't find enough information in this document to answer your question."
        if not retrieved_chunks:
            logger.warning("[Request: %s] Retrieval returned 0 matches. Skipping LLM call.", request_id)
            overall_time_ms = int(round((time.perf_counter() - overall_start) * 1000))
            
            # Save stats atomically
            stats_payload = {
                "request_id": request_id,
                "provider": self.provider.provider_name(),
                "model": self.provider.model_name(),
                "processing_time_ms": overall_time_ms,
                "retrieval_time_ms": retrieval_time_ms,
                "generation_time_ms": 0,
                "question_length": len(request.question),
                "context_chunks": 0,
                "context_characters": 0,
                "estimated_prompt_tokens": 0,
                "estimated_completion_tokens": 0,
                "estimated_total_tokens": 0,
                "returned_sources": 0,
                "chat_version": settings.CHAT_VERSION,
                "system_prompt_version": settings.SYSTEM_PROMPT_VERSION,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            self._write_atomic(stats_path, stats_payload)

            return ChatResponse(
                success=True,
                document_id=safe_doc_id,
                request_id=request_id,
                question=request.question,
                answer=fallback_msg,
                provider=self.provider.provider_name(),
                model=self.provider.model_name(),
                processing_time_ms=overall_time_ms,
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=0,
                sources=[]
            )

        # 7. Context Deduplication & Boundary Truncation Limits
        cleaned_chunks = []
        seen_ids = set()
        context_char_count = 0

        for chunk in retrieved_chunks:
            if chunk.chunk_id in seen_ids:
                continue
            if not chunk.text or not chunk.text.strip():
                continue
            
            # Boundary limit check on top chunks count
            if len(cleaned_chunks) >= settings.MAX_CONTEXT_CHUNKS:
                break
                
            # Boundary limit check on characters length
            chunk_len = len(chunk.text)
            if context_char_count + chunk_len > settings.MAX_CONTEXT_CHARACTERS:
                # If adding this chunk exceeds limit, truncate context assembly
                break

            seen_ids.add(chunk.chunk_id)
            cleaned_chunks.append(chunk)
            context_char_count += chunk_len

        # Double check that we still have chunks after cleanup
        if not cleaned_chunks:
            logger.warning("[Request: %s] No valid chunks remaining after deduplication. Skipping LLM.", request_id)
            overall_time_ms = int(round((time.perf_counter() - overall_start) * 1000))
            return ChatResponse(
                success=True,
                document_id=safe_doc_id,
                request_id=request_id,
                question=request.question,
                answer=fallback_msg,
                provider=self.provider.provider_name(),
                model=self.provider.model_name(),
                processing_time_ms=overall_time_ms,
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=0,
                sources=[]
            )

        # 8 & 9. Compile Grounded Prompt Context
        system_prompt = PromptBuilder.get_system_prompt()
        compiled_context = PromptBuilder.compile_context(cleaned_chunks)
        logger.info("[Request: %s] Modular prompt compiled.", request_id)

        # 10. Call LLM Provider completions
        generation_start = time.perf_counter()
        try:
            raw_answer = self.provider.generate(
                system_prompt=system_prompt,
                context=compiled_context,
                question=request.question
            )
        except Exception as gen_err:
            logger.exception("[Request: %s] LLM generation failed", request_id)
            if isinstance(gen_err, HTTPException):
                raise
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Grounded AI response generation failure: {str(gen_err)}"
            )

        generation_time_ms = int(round((time.perf_counter() - generation_start) * 1000))
        logger.info("[Request: %s] LLM text generation completed in %d ms.", request_id, generation_time_ms)

        # 11. Validate generated response text
        if not raw_answer or not raw_answer.strip():
            logger.error("[Request: %s] Generated completion response is empty or whitespace.", request_id)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The AI provider returned an empty completion response."
            )
        
        answer = raw_answer.strip()

        # 12. Map cited Source Chunks (Rounded to 4 decimal places in response payload)
        sources: List[SourceChunk] = []
        for chunk in cleaned_chunks:
            sources.append(
                SourceChunk(
                    chunk_id=chunk.chunk_id,
                    chunk_index=chunk.chunk_index,
                    score=round(chunk.score, 4),
                    start_page=chunk.start_page,
                    end_page=chunk.end_page
                )
            )

        overall_time_ms = int(round((time.perf_counter() - overall_start) * 1000))

        # 13. Estimate token usages (≈ 4 characters = 1 token ratio)
        estimated_prompt_tokens = 0
        estimated_completion_tokens = 0
        
        if settings.ENABLE_TOKEN_ESTIMATION:
            # System prompt + Context + User query
            total_prompt_chars = len(system_prompt) + len(compiled_context) + len(request.question)
            estimated_prompt_tokens = int(round(total_prompt_chars / 4))
            estimated_completion_tokens = int(round(len(answer) / 4))

        estimated_total_tokens = estimated_prompt_tokens + estimated_completion_tokens

        # Persist stats atomically
        stats_payload = {
            "request_id": request_id,
            "provider": self.provider.provider_name(),
            "model": self.provider.model_name(),
            "processing_time_ms": overall_time_ms,
            "retrieval_time_ms": retrieval_time_ms,
            "generation_time_ms": generation_time_ms,
            "question_length": len(request.question),
            "context_chunks": len(cleaned_chunks),
            "context_characters": context_char_count,
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "estimated_completion_tokens": estimated_completion_tokens,
            "estimated_total_tokens": estimated_total_tokens,
            "returned_sources": len(sources),
            "chat_version": settings.CHAT_VERSION,
            "system_prompt_version": settings.SYSTEM_PROMPT_VERSION,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        try:
            self._write_atomic(stats_path, stats_payload)
            logger.info("[Request: %s] chat_statistics.json persisted atomically.", request_id)
        except Exception as err:
            logger.warning("[Request: %s] Failed to write chat_statistics.json: %s", request_id, str(err))

        # 14. Return Response payload
        return ChatResponse(
            success=True,
            document_id=safe_doc_id,
            request_id=request_id,
            question=request.question,
            answer=answer,
            provider=self.provider.provider_name(),
            model=self.provider.model_name(),
            processing_time_ms=overall_time_ms,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            sources=sources
        )
