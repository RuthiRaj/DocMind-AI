"""
Groq API Large Language Model Provider.

Implements the LLMProvider interface utilizing the Groq SDK client.
Features lazy-loading singletons, timeout boundaries, deterministic parameters,
and advanced error classification handling.
"""

import json
import logging
import threading
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
                    
                    if not api_key or not api_key.strip():
                        detail_msg = "GROQ_API_KEY environment variable is not configured on the backend server."
                        logger.error("Configuration failure: %s", detail_msg)
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=detail_msg
                        )

                    logger.info("Initializing Groq API client singleton...")
                    try:
                        # Instantiate Groq client
                        GroqProvider._client_instance = Groq(api_key=api_key)
                    except Exception as err:
                        logger.exception("Failed to initialize Groq client: %s", str(err))
                        raise RuntimeError(f"Client initialization failed: {str(err)}")
                else:
                    logger.info("Groq client retrieved from double-checked locked instance.")
        else:
            logger.info("Groq client retrieved from cached instance.")

        return GroqProvider._client_instance

    def generate(
        self,
        system_prompt: str,
        context: str,
        question: str,
        history: list | None = None,
        context_mode: str = "RAG"
    ) -> tuple[str, bool]:
        """
        Invokes completions endpoint.

        Args:
            system_prompt (str): Core grounding prompt.
            context (str): Joint chunks text.
            question (str): User question.

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

        request_payload = {
            "messages": messages,
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        total_prompt_chars = sum(len(message.get("content", "")) for message in messages)
        estimated_prompt_tokens = (
            total_prompt_chars + settings.TOKEN_ESTIMATION_RATIO - 1
        ) // settings.TOKEN_ESTIMATION_RATIO
        serialized_payload_bytes = len(json.dumps(request_payload, ensure_ascii=False).encode("utf-8"))

        logger.info(
            "Groq request diagnostics: context_mode=%s context_chars=%d estimated_prompt_tokens=%d "
            "max_output_tokens=%d serialized_payload_bytes=%d",
            context_mode,
            len(context),
            estimated_prompt_tokens,
            self._max_tokens,
            serialized_payload_bytes,
        )

        request_tokens = estimated_prompt_tokens + self._max_tokens
        allowed, retry_after = groq_token_window.reserve(
            tokens=request_tokens,
            limit=settings.GROQ_TPM_LIMIT,
            window=60
        )
        if not allowed:
            logger.warning(
                "Groq token window rejected request: requested=%d retry_after=%ds",
                request_tokens,
                retry_after,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"The AI service token limit is temporarily exhausted. Please try again in {retry_after} seconds."
            )

        logger.info("Submitting query request to Groq (%s) with timeout=%ds...", self._model, self._timeout)
        
        try:
            # Synchronous non-streaming completions request
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                timeout=self._timeout
            )

            if not chat_completion.choices or len(chat_completion.choices) == 0:
                raise ValueError("Groq returned completion response containing zero choices.")

            answer = chat_completion.choices[0].message.content
            if not answer:
                raise ValueError("Groq returned empty text content in completion message.")

            return answer, preflight_truncated

        except APITimeoutError as err:
            logger.error("Groq API request timed out: %s", str(err))
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="The AI provider connection timed out. Please try your request again later."
            )
        except AuthenticationError as err:
            logger.error("Groq authentication failure: %s", str(err))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Invalid API credentials configuration. Authentication failed."
            )
        except RateLimitError as err:
            logger.warning("Groq rate limit exceeded: %s", str(err))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="The AI service rate limit was exceeded. Please try again shortly."
            )
        except APIConnectionError as err:
            logger.error("Failed to connect to Groq server API: %s", str(err))
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Connection to the external AI provider failed."
            )
        except APIStatusError as err:
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
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"External AI service returned an error status: {err.status_code}."
            )
        except Exception as exc:
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
