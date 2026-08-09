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

    def _full_context_token_budget(self, system_prompt: str, question: str, history: list) -> int:
        """Calculate the document token budget after reserving prompt and output tokens."""
        fixed_prompt_chars = len(system_prompt) + len(question) + len("DOCUMENT CONTEXT:\n\nUSER QUESTION: ")
        history_chars = sum(len(message.get("content", "")) for message in history)
        reserved_tokens = (
            int((fixed_prompt_chars + history_chars + settings.TOKEN_ESTIMATION_RATIO - 1) / settings.TOKEN_ESTIMATION_RATIO)
            + settings.LLM_MAX_TOKENS
        )
        available_tokens = max(0, settings.MODEL_CONTEXT_WINDOW - reserved_tokens)
        return int(available_tokens * 0.7)

    async def answer_question(self, document_id: str, request: ChatRequest) -> ChatResponse:
        """
        Retrieves context segments and coordinates generation from the LLM provider.
        Routes to full-context mode for small documents or RAG mode for large ones.
        Includes session-scoped conversation memory for multi-turn follow-ups.
        """
        import re as _re

        # 1. Generate Request ID and resolve session ID
        request_id = str(uuid.uuid4())
        session_id = request.session_id if request.session_id else str(uuid.uuid4())
        safe_doc_id = Path(document_id).name
        doc_dir = self.target_dir / safe_doc_id
        status_path = doc_dir / "status.json"
        stats_path = doc_dir / "chat_statistics.json"

        logger.info(
            "[Request: %s] Chat query received for document_id '%s' (session: %s)",
            request_id,
            safe_doc_id,
            session_id[:8]
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

        # 5. Retrieve conversation history for this session
        from app.services.chat.conversation_store import conversation_store
        history = conversation_store.get_history(
            document_id=safe_doc_id,
            session_id=session_id,
            max_turns=settings.CONVERSATION_MAX_TURNS,
            max_tokens=settings.CONVERSATION_MAX_TOKENS
        )
        if history:
            logger.info(
                "[Request: %s] Conversation history loaded: %d messages from session %s",
                request_id, len(history), session_id[:8]
            )

        # 6. Document-size routing decision
        full_text_path = doc_dir / "extracted_text.txt"
        pages_path = doc_dir / "pages.json"

        context_mode = "RAG"  # Default
        full_text = None
        pages_data = []

        if full_text_path.exists():
            try:
                full_text = full_text_path.read_text(encoding="utf-8")
                if pages_path.exists():
                    with open(pages_path, "r", encoding="utf-8") as pf:
                        pages_data = json.load(pf)

                if len(full_text) > settings.FULL_CONTEXT_MAX_CHARS:
                    logger.info(
                        "[Request: %s] RAG mode selected — document is %d chars, exceeds char cap %d",
                        request_id, len(full_text), settings.FULL_CONTEXT_MAX_CHARS
                    )
                else:
                    full_context_system_prompt = PromptBuilder.get_full_context_system_prompt()
                    full_context = PromptBuilder.compile_full_context(full_text, pages_data)
                    estimated_context_tokens = int(
                        (len(full_context) + settings.TOKEN_ESTIMATION_RATIO - 1)
                        / settings.TOKEN_ESTIMATION_RATIO
                    )
                    safe_context_token_budget = self._full_context_token_budget(
                        system_prompt=full_context_system_prompt,
                        question=request.question,
                        history=history
                    )

                    if estimated_context_tokens <= safe_context_token_budget:
                        context_mode = "FULL_CONTEXT"
                        logger.info(
                            "[Request: %s] FULL_CONTEXT mode selected — document is %d chars (%d estimated tokens; budget: %d)",
                            request_id, len(full_text), estimated_context_tokens, safe_context_token_budget
                        )
                    else:
                        logger.info(
                            "[Request: %s] RAG mode selected — document is %d estimated tokens, exceeds budget %d",
                            request_id, estimated_context_tokens, safe_context_token_budget
                        )
            except Exception as err:
                logger.warning(
                    "[Request: %s] Failed to read extracted_text.txt, falling back to RAG mode: %s",
                    request_id, str(err)
                )

        fallback_msg = "I couldn't find enough information in this document to answer your question."

        # ── FULL-CONTEXT PATH ──
        if context_mode == "FULL_CONTEXT":
            retrieval_time_ms = 0

            # Compile full document context with [Page N] markers
            compiled_context = PromptBuilder.compile_full_context(full_text, pages_data)
            system_prompt = PromptBuilder.get_full_context_system_prompt()
            context_char_count = len(compiled_context)
            logger.info("[Request: %s] Full-context prompt compiled (%d chars).", request_id, context_char_count)

            # Call LLM with conversation history
            generation_start = time.perf_counter()
            try:
                raw_answer = self.provider.generate(
                    system_prompt=system_prompt,
                    context=compiled_context,
                    question=request.question,
                    history=history if history else None
                )
            except Exception as gen_err:
                logger.exception("[Request: %s] LLM generation failed (FULL_CONTEXT)", request_id)
                if isinstance(gen_err, HTTPException):
                    raise
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Grounded AI response generation failure: {str(gen_err)}"
                )

            generation_time_ms = int(round((time.perf_counter() - generation_start) * 1000))
            logger.info("[Request: %s] LLM generation completed in %d ms (FULL_CONTEXT).", request_id, generation_time_ms)

            if not raw_answer or not raw_answer.strip():
                logger.error("[Request: %s] Generated completion response is empty.", request_id)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="The AI provider returned an empty completion response."
                )

            answer = raw_answer.strip()

            # Parse page citations from LLM response (e.g. "Page 5", "Pages 3-4", "page 12")
            sources: List[SourceChunk] = []
            cited_pages = set()
            # Match patterns: "Page 5", "page 12", "Pages 3 and 5", "Pages 3-4", "Pages 3, 5"
            page_matches = _re.findall(r'[Pp]ages?\s+(\d+(?:\s*(?:,|and|-|to)\s*\d+)*)', answer)
            for match in page_matches:
                # Extract all numbers from the match
                numbers = _re.findall(r'\d+', match)
                for n in numbers:
                    cited_pages.add(int(n))

            # Build source citations from cited pages
            for page_num in sorted(cited_pages):
                # Find the page's text segment from pages_data
                page_text = ""
                for page_info in pages_data:
                    if page_info.get("page") == page_num:
                        start = page_info.get("start_character", 0)
                        end = page_info.get("end_character", len(full_text))
                        page_text = full_text[start:end].strip()
                        # Truncate for response payload (keep first 500 chars)
                        if len(page_text) > 500:
                            page_text = page_text[:500] + "..."
                        break

                sources.append(
                    SourceChunk(
                        chunk_id=f"{safe_doc_id}_page_{page_num:03d}",
                        chunk_index=page_num,
                        score=1.0,  # Full-context mode — full document was visible
                        start_page=page_num,
                        end_page=page_num,
                        text=page_text
                    )
                )

            # Store conversation turn
            conversation_store.add_turn(safe_doc_id, session_id, request.question, answer)

            overall_time_ms = int(round((time.perf_counter() - overall_start) * 1000))

            # Token estimation
            estimated_prompt_tokens = 0
            estimated_completion_tokens = 0
            if settings.ENABLE_TOKEN_ESTIMATION:
                total_prompt_chars = len(system_prompt) + len(compiled_context) + len(request.question)
                # Add conversation history token estimate
                history_chars = sum(len(m.get("content", "")) for m in (history or []))
                total_prompt_chars += history_chars
                estimated_prompt_tokens = int(round(total_prompt_chars / 4))
                estimated_completion_tokens = int(round(len(answer) / 4))
            estimated_total_tokens = estimated_prompt_tokens + estimated_completion_tokens

            # Persist stats
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
                "context_chunks": 0,
                "context_characters": context_char_count,
                "estimated_prompt_tokens": estimated_prompt_tokens,
                "estimated_completion_tokens": estimated_completion_tokens,
                "estimated_total_tokens": estimated_total_tokens,
                "returned_sources": len(sources),
                "cited_pages": sorted(cited_pages),
                "conversation_history_messages": len(history) if history else 0,
                "chat_version": settings.CHAT_VERSION,
                "system_prompt_version": settings.SYSTEM_PROMPT_VERSION,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            try:
                self._write_atomic(stats_path, stats_payload)
                logger.info("[Request: %s] chat_statistics.json persisted.", request_id)
            except Exception as err:
                logger.warning("[Request: %s] Failed to write chat_statistics.json: %s", request_id, str(err))

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
                context_mode=context_mode
            )

        # ── RAG PATH (existing flow, unchanged logic) ──
        # 5r. Context Retrieval via existing RetrievalService
        retrieval_start = time.perf_counter()
        try:
            retrieved_chunks = self.retrieval_service.retrieve(
                document_id=safe_doc_id,
                query=request.question,
                top_k=request.top_k,
                request_id=request_id
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

        # 6r. Empty Retrieval Optimization
        if not retrieved_chunks:
            logger.warning("[Request: %s] Retrieval returned 0 matches. Skipping LLM call.", request_id)
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
                context_mode=context_mode
            )

        # 7r. Context Deduplication & Boundary Truncation Limits
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
            logger.warning("[Request: %s] No valid chunks remaining after deduplication. Skipping LLM.", request_id)
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
                context_mode=context_mode
            )

        # 8r & 9r. Compile Grounded Prompt Context
        system_prompt = PromptBuilder.get_system_prompt()
        compiled_context = PromptBuilder.compile_context(cleaned_chunks)
        logger.info("[Request: %s] Modular prompt compiled (RAG).", request_id)

        # 10r. Call LLM Provider with conversation history
        generation_start = time.perf_counter()
        try:
            raw_answer = self.provider.generate(
                system_prompt=system_prompt,
                context=compiled_context,
                question=request.question,
                history=history if history else None
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
        logger.info("[Request: %s] LLM text generation completed in %d ms (RAG).", request_id, generation_time_ms)

        # 11r. Validate generated response text
        if not raw_answer or not raw_answer.strip():
            logger.error("[Request: %s] Generated completion response is empty or whitespace.", request_id)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The AI provider returned an empty completion response."
            )
        
        answer = raw_answer.strip()

        # Store conversation turn
        conversation_store.add_turn(safe_doc_id, session_id, request.question, answer)

        # 12r. Map cited Source Chunks
        sources: List[SourceChunk] = []
        for chunk in cleaned_chunks:
            sources.append(
                SourceChunk(
                    chunk_id=chunk.chunk_id,
                    chunk_index=chunk.chunk_index,
                    last_chunk_index=getattr(chunk, "last_chunk_index", None) if hasattr(chunk, "last_chunk_index") else chunk.get("last_chunk_index", None),
                    score=round(chunk.score, 4),
                    start_page=chunk.start_page,
                    end_page=chunk.end_page,
                    text=chunk.text
                )
            )

        overall_time_ms = int(round((time.perf_counter() - overall_start) * 1000))

        # 13r. Estimate token usages
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

        # 14r. Return Response payload
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
            context_mode=context_mode
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
