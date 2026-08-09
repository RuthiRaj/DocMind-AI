"""
Groq API Large Language Model Provider.

Implements the LLMProvider interface utilizing the Groq SDK client.
Features lazy-loading singletons, timeout boundaries, deterministic parameters,
and advanced error classification handling.
"""

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
from app.services.chat.provider import LLMProvider

logger = logging.getLogger(__name__)


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

    def generate(self, system_prompt: str, context: str, question: str, history: list | None = None) -> str:
        """
        Invokes completions endpoint.

        Args:
            system_prompt (str): Core grounding prompt.
            context (str): Joint chunks text.
            question (str): User question.

        Returns:
            str: Grounded answer.
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

            return answer

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
            logger.error("Groq API returned status failure %d: %s", err.status_code, str(err))
            if err.status_code == 413:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Document context too large for the AI model — try a more specific question or a shorter document."
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
