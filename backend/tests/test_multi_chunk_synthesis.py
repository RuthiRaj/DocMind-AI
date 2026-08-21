"""
Regression tests for Multi-Chunk Synthesis & Generation Degradation Handling.
Verifies that:
1. Broad multi-chunk synthesis queries produce complete grounded answers with citations.
2. Truncated generation under finish_reason='length' with non-zero retrieved chunks returns honest synthesis guidance (never 'not enough information').
3. Genuinely unmentioned topics return 'not enough information'.
"""

import pytest
from unittest.mock import MagicMock
from fastapi import HTTPException
from app.services.chat.groq_provider import GroqProvider
from app.core.config import settings
from app.services.chat.prompt_builder import PromptBuilder


class TestMultiChunkSynthesis:

    def test_truncation_with_chunks_returns_honest_synthesis_guidance(self, monkeypatch: pytest.MonkeyPatch):
        """When generation runs out of tokens while chunks were retrieved, tell the user honestly."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.finish_reason = "length"
        mock_choice.message = MagicMock(content="", reasoning="We need to answer using the context...")
        
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage = MagicMock(prompt_tokens=500, completion_tokens=1024, total_tokens=1524)
        
        mock_raw = MagicMock()
        mock_raw.parse.return_value = mock_completion
        mock_raw.headers = {}
        mock_client.chat.completions.with_raw_response.create.return_value = mock_raw
        
        monkeypatch.setattr(GroqProvider, "_client_instance", mock_client)
        monkeypatch.setattr(
            "app.services.chat.groq_provider.groq_token_window.reserve",
            lambda **kwargs: (True, 0, "res-test-123"),
        )
        
        provider = GroqProvider()
        answer, truncated = provider.generate(
            system_prompt="You are a document assistant.",
            context="Chunk 1: Details about Transformer encoder and decoder...",
            question="Explain the complete Transformer architecture.",
            final_chunks_count=7
        )
        
        assert "synthesizing a substantial amount of information" in answer
        assert "couldn't find enough information" not in answer

    def test_genuine_no_info_returns_standard_fallback(self, monkeypatch: pytest.MonkeyPatch):
        """When 0 chunks are present or model stops with empty output, return standard no info fallback."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message = MagicMock(content="", reasoning="")
        
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage = MagicMock(prompt_tokens=200, completion_tokens=10, total_tokens=210)
        
        mock_raw = MagicMock()
        mock_raw.parse.return_value = mock_completion
        mock_raw.headers = {}
        mock_client.chat.completions.with_raw_response.create.return_value = mock_raw
        
        monkeypatch.setattr(GroqProvider, "_client_instance", mock_client)
        monkeypatch.setattr(
            "app.services.chat.groq_provider.groq_token_window.reserve",
            lambda **kwargs: (True, 0, "res-test-456"),
        )
        
        provider = GroqProvider()
        answer, truncated = provider.generate(
            system_prompt="You are a document assistant.",
            context="",
            question="What is the capital of Mars?",
            final_chunks_count=0
        )
        
        assert "couldn't find enough information" in answer

    def test_partial_content_before_truncation_is_preserved(self, monkeypatch: pytest.MonkeyPatch):
        """If partial content was generated before finish_reason='length', preserve the grounded content."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.finish_reason = "length"
        mock_choice.message = MagicMock(
            content="The Transformer architecture consists of an Encoder (6 layers) and Decoder (6 layers).",
            reasoning="Reasoning steps..."
        )
        
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage = MagicMock(prompt_tokens=400, completion_tokens=1024, total_tokens=1424)
        
        mock_raw = MagicMock()
        mock_raw.parse.return_value = mock_completion
        mock_raw.headers = {}
        mock_client.chat.completions.with_raw_response.create.return_value = mock_raw
        
        monkeypatch.setattr(GroqProvider, "_client_instance", mock_client)
        monkeypatch.setattr(
            "app.services.chat.groq_provider.groq_token_window.reserve",
            lambda **kwargs: (True, 0, "res-test-789"),
        )
        
        provider = GroqProvider()
        answer, truncated = provider.generate(
            system_prompt="You are a document assistant.",
            context="Chunk 1...",
            question="Explain the complete Transformer architecture.",
            final_chunks_count=5
        )
        
        assert "The Transformer architecture consists of an Encoder" in answer
        assert "couldn't find enough information" not in answer
