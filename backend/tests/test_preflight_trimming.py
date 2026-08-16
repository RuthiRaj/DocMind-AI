"""
Tests for Groq preflight trimming: token budget enforcement, question preservation,
and clean 413 responses when the payload cannot be reduced further.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.core.config import settings
from app.services.chat.groq_provider import (
    GroqProvider,
    PAYLOAD_TOO_LARGE_DETAIL,
    _estimate_messages_tokens,
    _preflight_trim_messages,
)
from app.services.chat.prompt_builder import PromptBuilder
from app.services.pdf.chat_service import ChatService

USER_QUESTION_MARKER = "\n\nUSER QUESTION: "


def _make_history(turns: int) -> list[dict]:
    """Build alternating user/assistant messages for multi-turn sessions."""
    history: list[dict] = []
    for turn in range(turns):
        history.append({"role": "user", "content": f"User turn {turn}: " + ("x" * 250)})
        history.append({"role": "assistant", "content": f"Assistant turn {turn}: " + ("y" * 350)})
    return history


def _make_chunks(count: int, chars_each: int, base_score: float = 0.5) -> list:
    return [
        SimpleNamespace(
            chunk_id=f"chunk_{index}",
            chunk_index=index + 1,
            score=base_score + (index * 0.02),
            start_page=1,
            end_page=1,
            text=("Z" * chars_each) + f" chunk-{index}",
        )
        for index in range(count)
    ]


def _build_groq_messages(system_prompt: str, history: list | None, context: str, question: str) -> list:
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append(
        {"role": "user", "content": f"DOCUMENT CONTEXT:\n{context}\n\nUSER QUESTION: {question}"}
    )
    return messages


def _extract_question_from_user_message(content: str) -> str:
    marker_index = content.find(USER_QUESTION_MARKER)
    assert marker_index != -1, "Expected USER QUESTION marker in Groq user message"
    return content[marker_index + len(USER_QUESTION_MARKER) :]


@pytest.fixture
def chat_service() -> ChatService:
    mock_provider = SimpleNamespace(
        provider_name=lambda: "mock",
        model_name=lambda: "mock",
    )
    return ChatService(provider=mock_provider)


class TestChatServicePreflightTrim:
    def test_multi_turn_history_and_near_full_context_document_stays_within_budget(
        self, chat_service: ChatService, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings, "GROQ_PREFLIGHT_PROMPT_TOKEN_BUDGET", 2500)

        system_prompt = PromptBuilder.get_system_prompt()
        question = "Summarize the methodology and list the three main conclusions."
        history = _make_history(8)
        chunks = _make_chunks(
            count=8,
            chars_each=max(500, settings.FULL_CONTEXT_MAX_CHARS // 8),
        )

        trimmed_history, trimmed_chunks, compiled_context, context_truncated = (
            chat_service._preflight_trim_for_groq(
                system_prompt=system_prompt,
                question=question,
                history=history,
                chunks=chunks,
            )
        )

        final_tokens = chat_service._estimate_groq_prompt_tokens(
            system_prompt,
            compiled_context,
            question,
            trimmed_history,
        )

        assert final_tokens <= settings.GROQ_PREFLIGHT_PROMPT_TOKEN_BUDGET
        assert context_truncated is True
        assert len(trimmed_history or []) < len(history)
        assert trimmed_chunks is not None
        assert len(trimmed_chunks) < len(chunks)

        groq_messages = _build_groq_messages(
            system_prompt, trimmed_history, compiled_context, question
        )
        trimmed_messages, groq_truncated = _preflight_trim_messages(groq_messages)
        assert _estimate_messages_tokens(trimmed_messages) <= settings.GROQ_PREFLIGHT_PROMPT_TOKEN_BUDGET
        assert groq_truncated in (True, False)
        assert _extract_question_from_user_message(trimmed_messages[-1]["content"]) == question

    def test_lowest_similarity_chunks_are_dropped_first(
        self, chat_service: ChatService, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings, "GROQ_PREFLIGHT_PROMPT_TOKEN_BUDGET", 800)

        system_prompt = PromptBuilder.get_system_prompt()
        question = "What is the primary result?"
        chunks = _make_chunks(count=6, chars_each=1200, base_score=0.1)

        _, trimmed_chunks, _, context_truncated = chat_service._preflight_trim_for_groq(
            system_prompt=system_prompt,
            question=question,
            history=None,
            chunks=chunks,
        )

        assert context_truncated is True
        assert trimmed_chunks is not None
        assert len(trimmed_chunks) < len(chunks)
        remaining_scores = [chunk.score for chunk in trimmed_chunks]
        dropped_scores = sorted({chunk.score for chunk in chunks} - set(remaining_scores))
        if dropped_scores:
            assert min(remaining_scores) > min(dropped_scores)

    def test_user_question_is_never_trimmed(
        self, chat_service: ChatService, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings, "GROQ_PREFLIGHT_PROMPT_TOKEN_BUDGET", 1200)

        system_prompt = PromptBuilder.get_system_prompt()
        question = "Exact question text that must remain intact " + ("?" * 120)
        history = _make_history(6)
        chunks = _make_chunks(count=5, chars_each=1500)

        trimmed_history, _, compiled_context, _ = chat_service._preflight_trim_for_groq(
            system_prompt=system_prompt,
            question=question,
            history=history,
            chunks=chunks,
        )

        messages = _build_groq_messages(system_prompt, trimmed_history, compiled_context, question)
        trimmed_messages, _ = _preflight_trim_messages(messages)

        assert _extract_question_from_user_message(trimmed_messages[-1]["content"]) == question

    def test_returns_context_truncated_flag_when_payload_was_reduced(
        self, chat_service: ChatService, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings, "GROQ_PREFLIGHT_PROMPT_TOKEN_BUDGET", 1500)

        _, _, _, context_truncated = chat_service._preflight_trim_for_groq(
            system_prompt=PromptBuilder.get_system_prompt(),
            question="Short question?",
            history=_make_history(4),
            chunks=_make_chunks(count=4, chars_each=1000),
        )

        assert context_truncated is True

    def test_raises_clean_413_when_only_system_prompt_and_question_exceed_budget(
        self, chat_service: ChatService, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings, "GROQ_PREFLIGHT_PROMPT_TOKEN_BUDGET", 80)

        with pytest.raises(HTTPException) as exc_info:
            chat_service._preflight_trim_for_groq(
                system_prompt=PromptBuilder.get_system_prompt(),
                question="Q" * 600,
                history=None,
                compiled_context="",
            )

        assert exc_info.value.status_code == 413
        assert exc_info.value.detail == (
            "Document context too large for the AI model — "
            "try a more specific question or a shorter document."
        )


class TestGroqProviderPreflightTrim:
    def test_preflight_messages_never_exceed_budget_with_long_history_and_context(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings, "GROQ_PREFLIGHT_PROMPT_TOKEN_BUDGET", 3000)

        system_prompt = PromptBuilder.get_system_prompt()
        question = "What changed between section 2 and section 4?"
        history = _make_history(9)
        context = "D" * settings.FULL_CONTEXT_MAX_CHARS

        messages = _build_groq_messages(system_prompt, history, context, question)
        trimmed_messages, truncated = _preflight_trim_messages(messages)

        assert _estimate_messages_tokens(trimmed_messages) <= settings.GROQ_PREFLIGHT_PROMPT_TOKEN_BUDGET
        assert truncated is True
        assert _extract_question_from_user_message(trimmed_messages[-1]["content"]) == question

    def test_raises_clean_413_when_question_and_system_prompt_exceed_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings, "GROQ_PREFLIGHT_PROMPT_TOKEN_BUDGET", 60)

        messages = _build_groq_messages(
            system_prompt=PromptBuilder.get_system_prompt(),
            history=None,
            context="",
            question="Important question " + ("?" * 500),
        )

        with pytest.raises(HTTPException) as exc_info:
            _preflight_trim_messages(messages)

        assert exc_info.value.status_code == 413
        assert exc_info.value.detail == PAYLOAD_TOO_LARGE_DETAIL

    def test_generate_returns_false_truncation_flag_after_non_truncating_call(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Second generate() must not inherit context_truncated=True from a prior trimmed call."""
        monkeypatch.setattr(settings, "GROQ_PREFLIGHT_PROMPT_TOKEN_BUDGET", 3000)

        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Answer"))]
        mock_client.chat.completions.create.return_value = mock_completion
        monkeypatch.setattr(GroqProvider, "_client_instance", mock_client)
        monkeypatch.setattr(
            "app.services.chat.groq_provider.groq_token_window.reserve",
            lambda **kwargs: (True, 0),
        )

        provider = GroqProvider()
        system_prompt = PromptBuilder.get_system_prompt()
        question = "What is the summary?"

        _, first_truncated = provider.generate(
            system_prompt=system_prompt,
            context="X" * settings.FULL_CONTEXT_MAX_CHARS,
            question=question,
            history=_make_history(9),
        )
        assert first_truncated is True

        _, second_truncated = provider.generate(
            system_prompt=system_prompt,
            context="Small context",
            question=question,
            history=None,
        )
        assert second_truncated is False
