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

    def _context_token_budget(
        self,
        system_prompt: str,
        question: str,
        history: list,
        reserve_query_rewrite: bool = False
    ) -> int:
        """Calculate the context token budget under the shared Groq TPM ceiling."""
        fixed_prompt_chars = len(system_prompt) + len(question) + len("DOCUMENT CONTEXT:\n\nUSER QUESTION: ")
        history_chars = sum(len(message.get("content", "")) for message in history)
        reserved_tokens = (
            int((fixed_prompt_chars + history_chars + settings.TOKEN_ESTIMATION_RATIO - 1) / settings.TOKEN_ESTIMATION_RATIO)
            + settings.LLM_MAX_TOKENS
        )
        if reserve_query_rewrite:
            reserved_tokens += settings.GROQ_QUERY_REWRITE_RESERVE_TOKENS
        available_tokens = max(0, settings.GROQ_TPM_LIMIT - reserved_tokens)
        return int(available_tokens * 0.7)

    def _estimate_groq_prompt_tokens(
        self,
        system_prompt: str,
        context: str,
        question: str,
        history: list | None,
    ) -> int:
        """Estimate prompt tokens for system prompt, history, compiled context, and question."""
        user_content = f"DOCUMENT CONTEXT:\n{context}\n\nUSER QUESTION: {question}"
        total_chars = len(system_prompt) + len(user_content)
        if history:
            total_chars += sum(len(message.get("content", "")) for message in history)
        return (total_chars + settings.TOKEN_ESTIMATION_RATIO - 1) // settings.TOKEN_ESTIMATION_RATIO

    def _preflight_trim_for_groq(
        self,
        system_prompt: str,
        question: str,
        history: list | None,
        chunks: list | None = None,
        compiled_context: str | None = None,
    ) -> tuple[list | None, list | None, str, bool]:
        """
        Trim conversation history and context to fit GROQ_PREFLIGHT_PROMPT_TOKEN_BUDGET
        before calling Groq. History is reduced first; RAG chunks drop lowest scores next.

        Returns:
            tuple: (trimmed_history, trimmed_chunks, context, context_truncated)
        """
        # Dynamically compute remaining available prompt token budget from rolling token window
        from app.core.rate_limit import groq_token_window
        current_used = groq_token_window.current_usage(window=60)
        remaining_window = max(0, settings.GROQ_TPM_LIMIT - current_used)
        completion_reserve = min(settings.LLM_MAX_TOKENS, settings.GROQ_COMPLETION_RESERVE_TOKENS)
        available_headroom = max(settings.GROQ_PREFLIGHT_HEADROOM_FLOOR, remaining_window - completion_reserve)
        budget = min(settings.GROQ_PREFLIGHT_PROMPT_TOKEN_BUDGET, available_headroom)

        payload_too_large_detail = (
            "Document context too large for the AI model — "
            "try a more specific question or a shorter document."
        )
        trimmed_history = list(history) if history else []
        original_history_len = len(trimmed_history)

        if chunks is not None:
            trimmed_chunks = sorted(chunks, key=lambda chunk: chunk.score, reverse=True)
            context = PromptBuilder.compile_context(trimmed_chunks) if trimmed_chunks else ""
            original_chunk_count = len(trimmed_chunks)
        else:
            trimmed_chunks = None
            context = compiled_context or ""
            original_chunk_count = None
        original_context_len = len(context)

        original_tokens = self._estimate_groq_prompt_tokens(
            system_prompt, context, question, trimmed_history or None
        )

        while trimmed_history and self._estimate_groq_prompt_tokens(
            system_prompt, context, question, trimmed_history
        ) > budget:
            if len(trimmed_history) >= 2:
                trimmed_history = trimmed_history[2:]
            else:
                trimmed_history = trimmed_history[1:]

        if trimmed_chunks is not None:
            while trimmed_chunks and self._estimate_groq_prompt_tokens(
                system_prompt, context, question, trimmed_history or None
            ) > budget:
                trimmed_chunks.pop()
                context = PromptBuilder.compile_context(trimmed_chunks) if trimmed_chunks else ""
        elif context:
            while context and self._estimate_groq_prompt_tokens(
                system_prompt, context, question, trimmed_history or None
            ) > budget:
                context = context[: max(0, int(len(context) * settings.CONTEXT_TRIM_DECAY_RATE))]

        final_tokens = self._estimate_groq_prompt_tokens(
            system_prompt, context, question, trimmed_history or None
        )
        if final_tokens > budget:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=payload_too_large_detail,
            )

        context_truncated = (
            len(trimmed_history) < original_history_len
            or (
                original_chunk_count is not None
                and len(trimmed_chunks) < original_chunk_count
            )
            or (
                original_chunk_count is None
                and len(context) < original_context_len
            )
        )

        if final_tokens < original_tokens:
            logger.info(
                "Preflight trimmed Groq payload from %d to %d estimated tokens (budget=%d)",
                original_tokens,
                final_tokens,
                budget,
            )

        return (
            trimmed_history if trimmed_history else None,
            trimmed_chunks,
            context,
            context_truncated,
        )

    async def answer_question(self, document_id: str, request: ChatRequest) -> ChatResponse:
        """
        Orchestrates the authoritative production RAG question-answering pipeline:
        1. Validate pipeline artifacts & request boundaries
        2. Load conversation history
        3. Semantic retrieval via RetrievalService (BM25 + FAISS + RRF + Reranker + Neighbor Merging)
        4. Strict context bounding (max chunks, max characters, preflight token trimming)
        5. Grounded LLM generation via Groq
        6. Citation construction exclusively from surviving context chunks (no regex extraction)
        """
        request_id = str(uuid.uuid4())
        session_id = request.session_id if request.session_id else str(uuid.uuid4())
        safe_doc_id = Path(document_id).name
        doc_dir = self.target_dir / safe_doc_id
        status_path = doc_dir / "status.json"
        stats_path = doc_dir / "chat_statistics.json"

        logger.info(
            "[REQUEST] request_id=%s document_id='%s' session_id=%s question='%s' top_k=%d",
            request_id,
            safe_doc_id,
            session_id[:8],
            request.question[:80],
            request.top_k
        )
        logger.info("[ROUTING] mode=RAG full_context_bypass=false")
        overall_start = time.perf_counter()

        # 1. Validate Document Folder & Pipeline Completion
        if not doc_dir.exists():
            detail_msg = f"Document directory not found for document_id: {document_id}"
            logger.error("[REQUEST: %s] %s", request_id, detail_msg)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=detail_msg
            )

        self._validate_pipeline(status_path)

        # 2. Validate Request Boundaries
        self._validate_request(request.question, request.top_k)

        # 3. Retrieve conversation history for this session
        from app.services.chat.conversation_store import conversation_store
        history = conversation_store.get_history(
            document_id=safe_doc_id,
            session_id=session_id,
            max_turns=settings.CONVERSATION_MAX_TURNS,
            max_tokens=settings.CONVERSATION_MAX_TOKENS
        )
        if history:
            logger.info(
                "[REQUEST: %s] Conversation history loaded: %d messages from session %s",
                request_id, len(history), session_id[:8]
            )

        fallback_msg = "I couldn't find enough information in this document to answer your question."
        context_mode = "RAG"

        # 4. Context Retrieval via RetrievalService
        retrieval_start = time.perf_counter()
        try:
            retrieved_chunks = self.retrieval_service.retrieve(
                document_id=safe_doc_id,
                query=request.question,
                top_k=request.top_k,
                request_id=request_id
            )
        except Exception as ret_err:
            logger.exception("[REQUEST: %s] RetrievalService query execution failed", request_id)
            if isinstance(ret_err, HTTPException):
                raise
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Semantic chunk retrieval failed: {str(ret_err)}"
            )

        retrieval_time_ms = int(round((time.perf_counter() - retrieval_start) * 1000))
        logger.info(
            "[RETRIEVAL] request_id=%s time_ms=%d retrieved_candidates=%d",
            request_id,
            retrieval_time_ms,
            len(retrieved_chunks)
        )

        # 5. Empty Retrieval Early-Exit Optimization
        if not retrieved_chunks:
            logger.warning("[RETRIEVAL] request_id=%s 0 matches above threshold. Early exit with grounded fallback.", request_id)
            overall_time_ms = int(round((time.perf_counter() - overall_start) * 1000))

            stats_payload = {
                "request_id": request_id,
                "session_id": session_id,
                "context_mode": context_mode,
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

            self._enrich_debug_info(
                doc_dir=doc_dir,
                question=request.question,
                cleaned_chunks=[],
                context_char_count=0,
                estimated_prompt_tokens=0,
                estimated_completion_tokens=0,
                estimated_total_tokens=0,
                generation_time_ms=0,
                system_prompt="",
                fallback_used=True,
                request_id=request_id
            )

            logger.info("[RESPONSE] request_id=%s answer_generated=fallback sources=0 total_time_ms=%d", request_id, overall_time_ms)
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
                sources=[],
                session_id=session_id,
                context_mode=context_mode,
                context_truncated=False,
            )

        # 6. Context Deduplication & Boundary Clamping
        cleaned_chunks = []
        seen_ids = set()
        context_char_count = 0

        for chunk in retrieved_chunks:
            if chunk.chunk_id in seen_ids:
                continue
            if not chunk.text or not chunk.text.strip():
                continue

            if len(cleaned_chunks) >= settings.MAX_CONTEXT_CHUNKS:
                break

            chunk_len = len(chunk.text)
            if context_char_count + chunk_len > settings.MAX_CONTEXT_CHARACTERS:
                break

            seen_ids.add(chunk.chunk_id)
            cleaned_chunks.append(chunk)
            context_char_count += chunk_len

        if not cleaned_chunks:
            logger.warning("[CONTEXT] request_id=%s No valid chunks remaining after deduplication. Skipping LLM.", request_id)
            overall_time_ms = int(round((time.perf_counter() - overall_start) * 1000))

            self._enrich_debug_info(
                doc_dir=doc_dir,
                question=request.question,
                cleaned_chunks=[],
                context_char_count=0,
                estimated_prompt_tokens=0,
                estimated_completion_tokens=0,
                estimated_total_tokens=0,
                generation_time_ms=0,
                system_prompt="",
                fallback_used=True,
                request_id=request_id
            )

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
                sources=[],
                session_id=session_id,
                context_mode=context_mode,
                context_truncated=False,
            )

        # 7. Compile prompt context & Authoritative Preflight Trimming
        system_prompt = PromptBuilder.get_system_prompt()
        context_truncated = False
        try:
            history, cleaned_chunks, compiled_context, context_truncated = self._preflight_trim_for_groq(
                system_prompt=system_prompt,
                question=request.question,
                history=history if history else None,
                chunks=cleaned_chunks,
            )
        except HTTPException:
            raise

        if not cleaned_chunks:
            logger.warning("[CONTEXT] request_id=%s No context chunks remain after preflight trimming. Skipping LLM.", request_id)
            return ChatResponse(
                success=True,
                document_id=safe_doc_id,
                request_id=request_id,
                question=request.question,
                answer=fallback_msg,
                provider=self.provider.provider_name(),
                model=self.provider.model_name(),
                processing_time_ms=int(round((time.perf_counter() - overall_start) * 1000)),
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=0,
                sources=[],
                session_id=session_id,
                context_mode=context_mode,
                context_truncated=context_truncated,
            )

        context_char_count = sum(len(chunk.text) for chunk in cleaned_chunks)
        context_page_numbers = sorted({chunk.start_page for chunk in cleaned_chunks} | {chunk.end_page for chunk in cleaned_chunks})
        logger.info(
            "[CONTEXT] request_id=%s final_chunks=%d final_chars=%d pages=%s",
            request_id,
            len(cleaned_chunks),
            context_char_count,
            context_page_numbers
        )

        # 8. Call LLM Provider with conversation history
        generation_start = time.perf_counter()
        try:
            raw_answer, groq_truncated = self.provider.generate(
                system_prompt=system_prompt,
                context=compiled_context,
                question=request.question,
                history=history if history else None,
                context_mode=context_mode
            )
            if groq_truncated:
                context_truncated = True
        except Exception as gen_err:
            logger.exception("[LLM] request_id=%s LLM generation failed: %s", request_id, str(gen_err))
            if isinstance(gen_err, HTTPException):
                raise
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Grounded AI response generation failure: {str(gen_err)}"
            )

        generation_time_ms = int(round((time.perf_counter() - generation_start) * 1000))
        logger.info("[LLM] request_id=%s text generation completed in %d ms.", request_id, generation_time_ms)

        # 9. Validate generated response text
        if not raw_answer or not raw_answer.strip():
            logger.error("[LLM] request_id=%s Generated completion response is empty or whitespace.", request_id)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The AI provider returned an empty completion response."
            )

        answer = raw_answer.strip()

        # 10. Store conversation turn
        conversation_store.add_turn(safe_doc_id, session_id, request.question, answer)

        # 11. Authoritative Citation Construction: Exclusively from surviving context chunks
        fallback_phrases = [
            "i couldn't find enough information in this document",
            "i cannot find enough information in this document",
            "not enough information in this document"
        ]
        is_fallback_answer = any(phrase in answer.lower() for phrase in fallback_phrases)

        sources: List[SourceChunk] = []
        if not is_fallback_answer:
            for chunk in cleaned_chunks:
                sources.append(
                    SourceChunk(
                        chunk_id=chunk.chunk_id,
                        chunk_index=chunk.chunk_index,
                        last_chunk_index=getattr(chunk, "last_chunk_index", None) if hasattr(chunk, "last_chunk_index") else (chunk.get("last_chunk_index", None) if isinstance(chunk, dict) else None),
                        score=round(chunk.score, 4),
                        start_page=chunk.start_page,
                        end_page=chunk.end_page,
                        text=chunk.text
                    )
                )

        source_pages = sorted({s.start_page for s in sources} | {s.end_page for s in sources})
        logger.info(
            "[SOURCES] request_id=%s source_count=%d source_pages=%s chunk_ids=%s",
            request_id,
            len(sources),
            source_pages,
            [s.chunk_id for s in sources]
        )

        overall_time_ms = int(round((time.perf_counter() - overall_start) * 1000))

        # 12. Token estimation
        estimated_prompt_tokens = 0
        estimated_completion_tokens = 0
        if settings.ENABLE_TOKEN_ESTIMATION:
            total_prompt_chars = len(system_prompt) + len(compiled_context) + len(request.question)
            history_chars = sum(len(m.get("content", "")) for m in (history or []))
            total_prompt_chars += history_chars
            estimated_prompt_tokens = int(round(total_prompt_chars / 4))
            estimated_completion_tokens = int(round(len(answer) / 4))

        estimated_total_tokens = estimated_prompt_tokens + estimated_completion_tokens

        # Persist stats atomically
        stats_payload = {
            "request_id": request_id,
            "session_id": session_id,
            "context_mode": context_mode,
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
            "conversation_history_messages": len(history) if history else 0,
            "chat_version": settings.CHAT_VERSION,
            "system_prompt_version": settings.SYSTEM_PROMPT_VERSION,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        try:
            self._write_atomic(stats_path, stats_payload)
            logger.info("[Request: %s] chat_statistics.json persisted atomically.", request_id)
        except Exception as err:
            logger.warning("[Request: %s] Failed to write chat_statistics.json: %s", request_id, str(err))

        # Enrich debug_info.json with complete generation info
        self._enrich_debug_info(
            doc_dir=doc_dir,
            question=request.question,
            cleaned_chunks=cleaned_chunks,
            context_char_count=context_char_count,
            estimated_prompt_tokens=estimated_prompt_tokens,
            estimated_completion_tokens=estimated_completion_tokens,
            estimated_total_tokens=estimated_total_tokens,
            generation_time_ms=generation_time_ms,
            system_prompt=system_prompt,
            fallback_used=(answer == fallback_msg),
            request_id=request_id
        )

        logger.info("[RESPONSE] request_id=%s answer_generated=true processing_time_ms=%d", request_id, overall_time_ms)
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
            sources=sources,
            session_id=session_id,
            context_mode=context_mode,
            context_truncated=context_truncated,
        )


    def _enrich_debug_info(
        self,
        doc_dir: Path,
        question: str,
        cleaned_chunks: list,
        context_char_count: int,
        estimated_prompt_tokens: int,
        estimated_completion_tokens: int,
        estimated_total_tokens: int,
        generation_time_ms: int,
        system_prompt: str,
        fallback_used: bool,
        request_id: Optional[str] = None
    ) -> None:
        """
        Enriches the debug entry in debug_info.json matching the request_id with generation and context metadata.
        """
        debug_path = doc_dir / "debug_info.json"
        if debug_path.exists():
            try:
                with open(debug_path, "r", encoding="utf-8") as df:
                    debug_list = json.load(df)
                if debug_list and isinstance(debug_list, list):
                    # Try to locate the entry matching request_id
                    target_entry = None
                    if request_id:
                        for entry in reversed(debug_list):
                            if entry.get("request_id") == request_id:
                                target_entry = entry
                                break
                    
                    # Fallback to search by question ONLY if that entry has no request_id (legacy compatibility)
                    if not target_entry:
                        for entry in reversed(debug_list):
                            if entry.get("question") == question and not entry.get("request_id"):
                                target_entry = entry
                                break
                                
                    if target_entry:
                        target_entry["model"] = self.provider.model_name()
                        target_entry["retrieval"] = {
                            "top_k_requested": target_entry.get("top_k_requested", settings.DEFAULT_TOP_K),
                            "top_k_returned": target_entry.get("top_k_returned", 0),
                            "highest_similarity": target_entry.get("highest_similarity", 0.0),
                            "minimum_similarity": target_entry.get("minimum_similarity", 0.0)
                        }
                        target_entry["context"] = {
                            "segment_count": len(cleaned_chunks),
                            "character_count": context_char_count,
                            "estimated_tokens": estimated_prompt_tokens
                        }
                        target_entry["generation"] = {
                            "provider": self.provider.provider_name(),
                            "model": self.provider.model_name(),
                            "latency_ms": generation_time_ms
                        }
                        target_entry["final_prompt_sent_to_groq"] = system_prompt
                        target_entry["token_estimates"] = {
                            "input_tokens": estimated_prompt_tokens,
                            "output_tokens": estimated_completion_tokens,
                            "total_tokens": estimated_total_tokens
                        }
                        target_entry["fallback_used"] = fallback_used
                        target_entry["response_generation_time_ms"] = generation_time_ms
                        
                        # Enforce log retention limits (Milestone 13 Phase 3)
                        from app.core.telemetry import prune_debug_list
                        debug_list = prune_debug_list(debug_list)
                        
                        self._write_atomic(debug_path, debug_list)
                    else:
                        logger.warning("[Request: %s] Debug log entry not found in debug_info.json. Skipping enrichment.", request_id)
            except Exception as err:
                logger.warning("Failed to enrich debug_info.json: %s", str(err))
