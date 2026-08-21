"""
Groq API Large Language Model Provider.

Implements the LLMProvider interface utilizing the Groq SDK client.
Features lazy-loading singletons, timeout boundaries, deterministic parameters,
and advanced error classification handling.
"""

import json
import logging
import math
import re
import threading
import time
from fastapi import HTTPException, status
from groq import Groq

# Try importing specific exception types from groq SDK, fallback to general if unavailable
try:
    from groq import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        RateLimitError,
    )
except ImportError:
    # Safe fallbacks if specific exceptions cannot be imported directly
    class APIConnectionError(Exception): pass
    class APIStatusError(Exception): pass
    class APITimeoutError(Exception): pass
    class AuthenticationError(Exception): pass
    class RateLimitError(Exception): pass

from app.core.config import settings
from app.core.rate_limit import groq_token_window
from app.services.chat.provider import LLMProvider

logger = logging.getLogger(__name__)

PAYLOAD_TOO_LARGE_DETAIL = (
    "Document context too large for the AI model — try a more specific question or a shorter document."
)


def _estimate_messages_tokens(messages: list) -> int:
    total_chars = sum(len(message.get("content", "")) for message in messages)
    return (total_chars + settings.TOKEN_ESTIMATION_RATIO - 1) // settings.TOKEN_ESTIMATION_RATIO


def _trim_oldest_history_turn(messages: list) -> list:
    """Remove the oldest user/assistant pair from messages (indices 1-2)."""
    if len(messages) <= 2:
        return messages
    history = messages[1:-1]
    if len(history) >= 2:
        history = history[2:]
    elif history:
        history = history[1:]
    return [messages[0], *history, messages[-1]]


def _truncate_context_in_user_message(content: str, max_context_chars: int) -> str:
    prefix = "DOCUMENT CONTEXT:\n"
    marker = "\n\nUSER QUESTION: "
    marker_idx = content.find(marker)
    if marker_idx == -1:
        return content[:max_context_chars]

    context_part = content[len(prefix):marker_idx]
    suffix = content[marker_idx:]
    if len(context_part) <= max_context_chars:
        return content

    return prefix + context_part[:max_context_chars] + suffix


def _preflight_trim_messages(messages: list) -> tuple[list, bool]:
    """
    Trim history and context until estimated prompt tokens fit the preflight budget.
    Raises HTTP 413 if the payload cannot be reduced enough without calling Groq.

    Returns:
        tuple[list, bool]: Trimmed messages and whether any history/context was removed.
    """
    budget = settings.GROQ_PREFLIGHT_PROMPT_TOKEN_BUDGET
    working = list(messages)
    truncated = False
    original_tokens = _estimate_messages_tokens(messages)

    while _estimate_messages_tokens(working) > budget and len(working) > 2:
        working = _trim_oldest_history_turn(working)
        truncated = True

    last_content = working[-1].get("content", "")
    while _estimate_messages_tokens(working) > budget and len(last_content) > len("DOCUMENT CONTEXT:\n\nUSER QUESTION: "):
        marker = "\n\nUSER QUESTION: "
        marker_idx = last_content.find(marker)
        if marker_idx == -1:
            break
        context_len = marker_idx - len("DOCUMENT CONTEXT:\n")
        if context_len <= 0:
            break
        new_context_len = max(0, int(context_len * 0.9))
        last_content = _truncate_context_in_user_message(last_content, new_context_len)
        working[-1] = {"role": "user", "content": last_content}
        truncated = True

    if _estimate_messages_tokens(working) > budget:
        logger.warning(
            "Groq preflight budget exceeded after trimming: estimated=%d budget=%d",
            _estimate_messages_tokens(working),
            budget,
        )
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=PAYLOAD_TOO_LARGE_DETAIL,
        )

    trimmed_tokens = _estimate_messages_tokens(working)
    if trimmed_tokens < original_tokens:
        logger.info(
            "Groq preflight trimmed payload from %d to %d estimated tokens (budget=%d)",
            original_tokens,
            trimmed_tokens,
            budget,
        )

    return working, truncated


class GroqProvider(LLMProvider):
    """
    Groq completion provider using settings configs.
    """

    _client_instance = None
    _lock = threading.Lock()

    def __init__(self):
        """
        Initialize configurations.
        """
        self._provider = "groq"
        self._model = settings.LLM_MODEL
        self._temperature = settings.LLM_TEMPERATURE
        self._max_tokens = settings.LLM_MAX_TOKENS
        self._timeout = settings.LLM_TIMEOUT

    def _get_client(self) -> Groq:
        """
        Thread-safe singleton loader for Groq API client connection.
        """
        if GroqProvider._client_instance is None:
            with GroqProvider._lock:
                if GroqProvider._client_instance is None:
                    # Validate API Key existence
                    api_key = settings.GROQ_API_KEY
                    if not api_key:
                        import os
                        api_key = os.environ.get("GROQ_API_KEY")
                    
                    if not api_key or not api_key.strip() or api_key.strip() == "your_groq_api_key_here":
                        detail_msg = (
                            "GROQ_API_KEY is not configured. Please copy backend/.env.example to "
                            "backend/.env and set your active Groq API key from https://console.groq.com/keys"
                        )
                        logger.error("Configuration failure: %s", detail_msg)
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=detail_msg
                        )

                    logger.info("Initializing Groq API client singleton with max_retries=0...")
                    try:
                        # Instantiate Groq client with max_retries=0 for immediate app-level arbitration
                        GroqProvider._client_instance = Groq(api_key=api_key, max_retries=0)
                    except Exception as err:
                        logger.exception("Failed to initialize Groq client: %s", str(err))
                        raise RuntimeError(f"Client initialization failed: {str(err)}")
                else:
                    logger.info("Groq client retrieved from double-checked locked instance.")
        else:
            logger.info("Groq client retrieved from cached instance.")

        return GroqProvider._client_instance

    @classmethod
    def calculate_completion_budget(
        cls,
        final_chunks_count: int = 0,
        context_chars: int = 0,
        question: str = ""
    ) -> int:
        """
        Calculates dynamic completion token budget based on query & context complexity:
        - Tier 1 (Narrow / 1-2 chunks / <= 2000 chars): 768 tokens (LLM_MIN_COMPLETION_TOKENS)
        - Tier 2 (Moderate / 3-5 chunks / 2001-5000 chars): 1536 tokens (LLM_DEFAULT_COMPLETION_TOKENS)
        - Tier 3 (Broad Multi-Chunk Synthesis / >= 6 chunks / > 5000 chars): 3072 tokens (LLM_MAX_COMPLETION_TOKENS)
        """
        if not getattr(settings, "ENABLE_DYNAMIC_TOKEN_BUDGETING", True):
            return getattr(settings, "LLM_MAX_TOKENS", 1536)

        min_tokens = getattr(settings, "LLM_MIN_COMPLETION_TOKENS", 768)
        default_tokens = getattr(settings, "LLM_DEFAULT_COMPLETION_TOKENS", 1536)
        max_tokens = getattr(settings, "LLM_MAX_COMPLETION_TOKENS", 3072)

        # Tier 3: Broad multi-chunk architectural or synthesis questions (6+ chunks or >= 5000 chars)
        if final_chunks_count >= 6 or context_chars >= 5000:
            return max_tokens
        # Tier 1: Narrow context / focused question (1-2 chunks or <= 2000 chars when few chunks)
        elif (final_chunks_count > 0 and final_chunks_count <= 2) or (final_chunks_count == 0 and 0 < context_chars <= 2000):
            return min_tokens
        # Tier 2: Moderate component-level questions (3-5 chunks / 2001-4999 chars)
        else:
            return default_tokens

    def generate(
        self,
        system_prompt: str,
        context: str,
        question: str,
        history: list | None = None,
        context_mode: str = "RAG",
        request_id: str | None = None,
        retrieved_chunks_count: int = 0,
        final_chunks_count: int = 0,
        **kwargs
    ) -> tuple[str, bool]:
        """
        Invokes completions endpoint.

        Args:
            system_prompt (str): Core grounding prompt.
            context (str): Joint chunks text.
            question (str): User question.
            history (list | None): Conversation turns.
            context_mode (str): Mode label ("RAG" or "FULL_CONTEXT").
            request_id (str | None): Correlated request ID.
            retrieved_chunks_count (int): Initial retrieved candidates.
            final_chunks_count (int): Final surviving context chunks.

        Returns:
            tuple[str, bool]: Grounded answer and whether preflight trimming occurred.
        """
        try:
            client = self._get_client()
        except HTTPException:
            raise
        except Exception as client_err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Configuration error: Groq client failed to initialize: {str(client_err)}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
        ]
        # Insert conversation history between system prompt and current question
        if history:
            messages.extend(history)
        messages.append(
            {"role": "user", "content": f"DOCUMENT CONTEXT:\n{context}\n\nUSER QUESTION: {question}"}
        )

        messages, preflight_truncated = _preflight_trim_messages(messages)

        # Dynamically compute completion token headroom based on context and retrieval complexity
        completion_reserve = self.calculate_completion_budget(
            final_chunks_count=final_chunks_count,
            context_chars=len(context),
            question=question
        )

        request_payload = {
            "messages": messages,
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": completion_reserve,
        }
        total_prompt_chars = sum(len(message.get("content", "")) for message in messages)
        estimated_prompt_tokens = (
            total_prompt_chars + settings.TOKEN_ESTIMATION_RATIO - 1
        ) // settings.TOKEN_ESTIMATION_RATIO
        serialized_payload_bytes = len(json.dumps(request_payload, ensure_ascii=False).encode("utf-8"))

        logger.info(
            "Groq request diagnostics: context_mode=%s context_chars=%d estimated_prompt_tokens=%d "
            "completion_budget=%d max_output_tokens=%d serialized_payload_bytes=%d",
            context_mode,
            len(context),
            estimated_prompt_tokens,
            completion_reserve,
            completion_reserve,
            serialized_payload_bytes,
        )

        session_tag = kwargs.get("session_id") or (request_id[:8] if request_id else "N/A")

        # Dynamic completion reservation based on tier budget for this specific request
        request_tokens = estimated_prompt_tokens + completion_reserve
        reserve_res = groq_token_window.reserve(
            tokens=request_tokens,
            limit=settings.GROQ_TPM_LIMIT,
            window=60,
            session_id=session_tag,
        )
        if isinstance(reserve_res, tuple) and len(reserve_res) == 3:
            allowed, retry_after, res_id = reserve_res
        else:
            allowed, retry_after = reserve_res[0], reserve_res[1]
            res_id = ""

        # Application-Level In-Memory Queue: Hold connection and retry until headroom frees up or max wait ceiling is reached
        max_queue_wait = getattr(settings, "GROQ_MAX_QUEUE_WAIT_SECONDS", 25.0)
        queue_start_time = time.perf_counter()
        queued = False

        while not allowed:
            elapsed = time.perf_counter() - queue_start_time
            if elapsed >= max_queue_wait:
                break
                
            queued = True
            # Sleep in responsive increments (max 1.5s per check to immediately catch in-flight settlements)
            wait_slice = min(1.5, max(0.5, float(retry_after)), max(0.1, max_queue_wait - elapsed))
            logger.info(
                "[APP_QUEUE_WAIT] session=%s request_id=%s waiting %.1fs for token window headroom (elapsed=%.1fs/%.1fs, retry_after=%ds)",
                session_tag,
                request_id[:8] if request_id else "N/A",
                wait_slice,
                elapsed,
                max_queue_wait,
                retry_after,
            )
            time.sleep(wait_slice)

            # Re-attempt reservation
            reserve_res = groq_token_window.reserve(
                tokens=request_tokens,
                limit=settings.GROQ_TPM_LIMIT,
                window=60,
                session_id=session_tag,
            )
            if isinstance(reserve_res, tuple) and len(reserve_res) == 3:
                allowed, retry_after, res_id = reserve_res
            else:
                allowed, retry_after = reserve_res[0], reserve_res[1]
                res_id = ""

            # If still rejected and history is present, attempt shedding history to fit within window
            if not allowed and history and len(messages) > 2:
                minimal_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"DOCUMENT CONTEXT:\n{context}\n\nUSER QUESTION: {question}"}
                ]
                min_chars = sum(len(m.get("content", "")) for m in minimal_messages)
                min_tokens = (min_chars + settings.TOKEN_ESTIMATION_RATIO - 1) // settings.TOKEN_ESTIMATION_RATIO + completion_reserve
                min_reserve = groq_token_window.reserve(
                    tokens=min_tokens,
                    limit=settings.GROQ_TPM_LIMIT,
                    window=60,
                    session_id=session_tag,
                )
                if min_reserve[0]:
                    logger.info(
                        "[RATE_LIMIT] Shedding %d history messages to fit queued request into token window (%d -> %d tokens)",
                        len(messages) - 2,
                        request_tokens,
                        min_tokens
                    )
                    messages = minimal_messages
                    preflight_truncated = True
                    allowed, retry_after, res_id = min_reserve
                    request_tokens = min_tokens
                    break

        if queued and allowed:
            total_waited = time.perf_counter() - queue_start_time
            logger.info("[APP_QUEUE_GRANTED] session=%s reservation granted after waiting %.2fs in queue.", session_tag, total_waited)

        if not allowed:
            total_waited = time.perf_counter() - queue_start_time
            logger.warning(
                "[RATE_LIMIT] Groq token window queue timeout (waited %.1fs): requested=%d retry_after=%ds",
                total_waited,
                request_tokens,
                retry_after,
            )

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"The AI service rate limit was exceeded. The request was queued for {int(total_waited)}s but could not be processed. Please wait {retry_after} seconds before trying again.",
                headers={"Retry-After": str(retry_after)},
            )

        logger.info("Submitting query request to Groq (%s) with timeout=%ds...", self._model, self._timeout)
        
        try:
            # Synchronous completions request capturing raw HTTP response headers
            raw_response = client.chat.completions.with_raw_response.create(
                messages=messages,
                model=self._model,
                temperature=self._temperature,
                max_tokens=completion_reserve,
                timeout=self._timeout
            )
            chat_completion = raw_response.parse()
            headers = dict(raw_response.headers)

            if not chat_completion.choices or len(chat_completion.choices) == 0:
                raise ValueError("Groq returned completion response containing zero choices.")

            msg_obj = chat_completion.choices[0].message
            content = (msg_obj.content or "").strip()
            finish_reason = getattr(chat_completion.choices[0], "finish_reason", "unknown")
            reasoning_text = getattr(msg_obj, "reasoning", "") or ""

            # Check whether content contains substantive alphanumeric text (not just whitespace or bare markdown control delimiters like '**' or '* ')
            has_substantive_content = bool(re.search(r"[A-Za-z0-9]", content))

            if has_substantive_content:
                # Substantive content exists — preserve complete or partial answer
                answer = content
            else:
                # Content is empty or contains only non-substantive markdown syntax
                if finish_reason == "length" and final_chunks_count > 0:
                    logger.warning(
                        "Groq generation truncated before substantive content (finish_reason=length, reasoning_len=%d, content_repr=%r, chunks=%d). Returning honest synthesis guidance.",
                        len(reasoning_text),
                        content,
                        final_chunks_count
                    )
                    answer = (
                        "This question involves synthesizing a substantial amount of information across the document. "
                        "Please try asking a more focused question about a specific component or section."
                    )
                else:
                    logger.warning(
                        "Groq response content is non-substantive or empty (finish_reason=%s, content_repr=%r, reasoning_tokens_present=%s). Returning grounded fallback.",
                        finish_reason,
                        content,
                        bool(reasoning_text.strip())
                    )
                    answer = "I couldn't find enough information in this document to answer your question."

            # Settle token window with actual usage if provided by Groq
            usage_obj = getattr(chat_completion, "usage", None)
            prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0)) if usage_obj else estimated_prompt_tokens
            completion_tokens = int(getattr(usage_obj, "completion_tokens", 0)) if usage_obj else (len(answer) // 4)
            actual_total_tokens = getattr(usage_obj, "total_tokens", None) if usage_obj else None

            if isinstance(actual_total_tokens, (int, float)) and actual_total_tokens > 0:
                total_tokens = int(actual_total_tokens)
            else:
                total_tokens = prompt_tokens + completion_tokens

            groq_token_window.settle(res_id, actual_tokens=total_tokens, session_id=session_tag)

            # Extract rate limit quota headers from Groq response
            rl_headers = {
                "remaining_tokens": headers.get("x-ratelimit-remaining-tokens"),
                "limit_tokens": headers.get("x-ratelimit-limit-tokens"),
                "reset_tokens": headers.get("x-ratelimit-reset-tokens"),
                "remaining_requests": headers.get("x-ratelimit-remaining-requests"),
                "limit_requests": headers.get("x-ratelimit-limit-requests"),
                "reset_requests": headers.get("x-ratelimit-reset-requests"),
            }

            # Parse provider telemetry numbers for window synchronization
            try:
                p_limit = int(rl_headers["limit_tokens"]) if rl_headers.get("limit_tokens") and rl_headers["limit_tokens"].isdigit() else None
                p_rem = int(rl_headers["remaining_tokens"]) if rl_headers.get("remaining_tokens") and rl_headers["remaining_tokens"].isdigit() else None
                p_reset = None
                raw_reset = rl_headers.get("reset_tokens")
                if raw_reset:
                    raw_reset = raw_reset.strip().lower()
                    if raw_reset.endswith("ms"):
                        p_reset = float(raw_reset[:-2]) / 1000.0
                    elif raw_reset.endswith("s"):
                        p_reset = float(raw_reset[:-1])
                groq_token_window.update_provider_telemetry(limit=p_limit, remaining=p_rem, reset_seconds=p_reset)
            except Exception:
                pass

            from app.core.telemetry import groq_telemetry
            groq_telemetry.record_call(
                call_type="chat_completion",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                ratelimit_headers=rl_headers,
                query=question,
                extra={"context_mode": context_mode, "model": self._model}
            )

            # Structured Diagnostic Metrics Log Entry
            logger.info(
                "[LLM_METRICS] request_id=%s llm_calls_in_request=1 retrieved_chunks=%d final_context_chunks=%d "
                "context_chars=%d estimated_prompt_tokens=%d completion_tokens=%d total_tokens=%d model=%s "
                "upstream_status=200 remaining_tokens=%s reset_tokens=%s",
                request_id or "N/A",
                retrieved_chunks_count,
                final_chunks_count or len(context.split("--- Chunk ")),
                len(context),
                prompt_tokens,
                completion_tokens,
                total_tokens,
                self._model,
                rl_headers.get("remaining_tokens") or "N/A",
                rl_headers.get("reset_tokens") or "N/A",
            )

            logger.info(
                "[GROQ_TELEMETRY] call=chat_completion prompt_tokens=%d completion_tokens=%d total_tokens=%d "
                "remaining_tokens=%s limit_tokens=%s reset_tokens=%s remaining_requests=%s",
                prompt_tokens,
                completion_tokens,
                total_tokens,
                rl_headers.get("remaining_tokens"),
                rl_headers.get("limit_tokens"),
                rl_headers.get("reset_tokens"),
                rl_headers.get("remaining_requests"),
            )

            return answer, preflight_truncated

        except APITimeoutError as err:
            groq_token_window.settle(res_id, actual_tokens=0)
            logger.error("Groq API request timed out: %s", str(err))
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="The AI provider connection timed out. Please try your request again later."
            )
        except AuthenticationError as err:
            groq_token_window.settle(res_id, actual_tokens=0)
            logger.error("Groq authentication failure: %s", str(err))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Invalid API credentials configuration. Authentication failed."
            )
        except RateLimitError as err:
            groq_token_window.settle(res_id, actual_tokens=0)
            logger.warning("[RATE_LIMIT] Groq provider-side rate limit exceeded: %s", str(err))

            # Parse retry-after from headers or error message
            err_headers = getattr(err, "response", None)
            retry_after_val = 2
            if err_headers and hasattr(err_headers, "headers"):
                ra = err_headers.headers.get("retry-after")
                if ra and ra.isdigit():
                    retry_after_val = int(ra)

            m = re.search(r"try again in ([\d\.]+)s", str(err), re.IGNORECASE)
            if m:
                try:
                    retry_after_val = max(1, int(math.ceil(float(m.group(1)))))
                except Exception:
                    pass

            # If upstream retry_after is small (<= 2.0s), attempt a quick retry after shedding history
            if retry_after_val <= 2:
                sleep_duration = max(1.0, float(retry_after_val))
                logger.info(
                    "[RATE_LIMIT] 429 received from Groq. Quick retry with history dropped (waiting %.1fs)...",
                    sleep_duration
                )
                time.sleep(sleep_duration)

            try:
                minimal_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"DOCUMENT CONTEXT:\n{context}\n\nUSER QUESTION: {question}"}
                ]
                retry_response = client.chat.completions.with_raw_response.create(
                    messages=minimal_messages,
                    model=self._model,
                    temperature=self._temperature,
                    max_tokens=completion_reserve,
                    timeout=self._timeout
                )
                retry_comp = retry_response.parse()
                if retry_comp.choices and len(retry_comp.choices) > 0:
                    retry_ans = retry_comp.choices[0].message.content
                    if retry_ans:
                        retry_headers = dict(retry_response.headers)
                        u_obj = getattr(retry_comp, "usage", None)
                        p_toks = int(getattr(u_obj, "prompt_tokens", 0)) if u_obj else 0
                        c_toks = int(getattr(u_obj, "completion_tokens", 0)) if u_obj else (len(retry_ans) // 4)
                        t_toks = getattr(u_obj, "total_tokens", None) or (p_toks + c_toks)
                        groq_token_window.settle(res_id, actual_tokens=int(t_toks))

                        logger.info("[RATE_LIMIT] Auto-retry succeeded cleanly.")
                        return retry_ans, True
            except Exception as retry_err:
                logger.warning("[RATE_LIMIT] Auto-retry failed: %s", str(retry_err))

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"The AI service rate limit was exceeded. Please wait {retry_after_val} seconds before trying again.",
                headers={"Retry-After": str(retry_after_val)}
            )
        except APIConnectionError as err:
            groq_token_window.settle(res_id, actual_tokens=0)
            logger.error("Failed to connect to Groq server API: %s", str(err))
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Connection to the external AI provider failed."
            )
        except APIStatusError as err:
            groq_token_window.settle(res_id, actual_tokens=0)
            response = getattr(err, "response", None)
            response_body = getattr(err, "body", None)
            if response_body is None and response is not None:
                response_body = getattr(response, "text", None)
            logger.error(
                "Groq API returned status failure %d. message=%s response=%r body=%r",
                err.status_code,
                str(err),
                response,
                response_body,
            )
            if err.status_code == 413:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=PAYLOAD_TOO_LARGE_DETAIL,
                )
            
            # Extract detailed error message from provider response body if present
            err_detail = ""
            if isinstance(response_body, dict) and "error" in response_body:
                err_detail = response_body["error"].get("message", "")

            if err.status_code == 404:
                detail_msg = f"AI model '{self._model}' was not found: {err_detail}" if err_detail else f"Configured AI model '{self._model}' was not found by the AI provider (404)."
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=detail_msg,
                )
            elif err.status_code in (401, 403):
                detail_msg = f"AI provider authentication error ({err.status_code}): {err_detail}" if err_detail else f"AI provider authentication failed ({err.status_code})."
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=detail_msg,
                )

            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"External AI service error ({err.status_code}): {err_detail or str(err)}"
            )
        except Exception as exc:
            groq_token_window.settle(res_id, actual_tokens=0)
            logger.exception("Unexpected error during Groq completion call: %s", str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected completion failure occurred: {str(exc)}"
            )


    def model_name(self) -> str:
        return self._model

    def provider_name(self) -> str:
        return self._provider

    def health_check(self) -> bool:
        """
        Validates connection status.
        """
        try:
            client = self._get_client()
            # Issue a minimal cost query check
            client.chat.completions.create(
                messages=[{"role": "user", "content": "Ping"}],
                model=self._model,
                max_tokens=5,
                timeout=5
            )
            return True
        except Exception as err:
            logger.warning("Groq provider health check failed: %s", str(err))
            return False
