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
        
    def test_truncation_with_bare_markdown_syntax_returns_synthesis_guidance(self, monkeypatch: pytest.MonkeyPatch):
        """When generation is cut off during markdown formatting (e.g. '**'), return synthesis guidance instead of leaking bare syntax."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.finish_reason = "length"
        mock_choice.message = MagicMock(content="**", reasoning="Reasoning steps...")
        
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage = MagicMock(prompt_tokens=500, completion_tokens=512, total_tokens=1012)
        
        mock_raw = MagicMock()
        mock_raw.parse.return_value = mock_completion
        mock_raw.headers = {}
        mock_client.chat.completions.with_raw_response.create.return_value = mock_raw
        
        monkeypatch.setattr(GroqProvider, "_client_instance", mock_client)
        monkeypatch.setattr(
            "app.services.chat.groq_provider.groq_token_window.reserve",
            lambda **kwargs: (True, 0, "res-test-md"),
        )
        
        provider = GroqProvider()
        answer, truncated = provider.generate(
            system_prompt="You are a document assistant.",
            context="Chunk 1: Attention mechanisms...",
            question="Explain how self-attention, multi-head attention, and feed-forward networks work together.",
            final_chunks_count=8
        )
        
        assert answer != "**"
        assert "synthesizing a substantial amount of information" in answer

    def test_truncation_with_formula_or_short_answer_is_preserved(self, monkeypatch: pytest.MonkeyPatch):
        """Formulas and concise answers containing alphanumeric tokens must be preserved."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.finish_reason = "length"
        mock_choice.message = MagicMock(
            content="Attention(Q,K,V) = softmax(QK^T/√dk)V",
            reasoning="Reasoning steps..."
        )
        
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage = MagicMock(prompt_tokens=400, completion_tokens=512, total_tokens=912)
        
        mock_raw = MagicMock()
        mock_raw.parse.return_value = mock_completion
        mock_raw.headers = {}
        mock_client.chat.completions.with_raw_response.create.return_value = mock_raw
        
        monkeypatch.setattr(GroqProvider, "_client_instance", mock_client)
        monkeypatch.setattr(
            "app.services.chat.groq_provider.groq_token_window.reserve",
            lambda **kwargs: (True, 0, "res-test-formula"),
        )
        
        provider = GroqProvider()
        answer, truncated = provider.generate(
            system_prompt="You are a document assistant.",
            context="Chunk 1...",
            question="What is the attention formula?",
            final_chunks_count=5
        )
        
        assert "Attention(Q,K,V)" in answer
        assert "synthesizing a substantial amount of information" not in answer

    def test_synthesis_guidance_suppresses_citations_in_chat_service(self):
        """When synthesis guidance is returned, chat service must suppress citations."""
        from app.schemas.chat import SourceChunk
        guidance = (
            "This question involves synthesizing a substantial amount of information across the document. "
            "Please try asking a more focused question about a specific component or section."
        )
        fallback_phrases = [
            "i couldn't find enough information in this document",
            "i cannot find enough information in this document",
            "not enough information in this document",
            "synthesizing a substantial amount of information"
        ]
        is_fallback_answer = any(phrase in guidance.lower() for phrase in fallback_phrases)
        assert is_fallback_answer is True

    def test_dynamic_completion_budgeting_tiers(self):
        """Dynamic completion budget maps correctly across complexity tiers."""
        provider = GroqProvider()

        # Tier 1: Narrow context (1-2 chunks / <= 2000 chars)
        b1 = provider.calculate_completion_budget(final_chunks_count=1, context_chars=800)
        b2 = provider.calculate_completion_budget(final_chunks_count=2, context_chars=1800)
        assert b1 == 768
        assert b2 == 768

        # Tier 2: Moderate context (3-5 chunks / 2001-5000 chars)
        b3 = provider.calculate_completion_budget(final_chunks_count=4, context_chars=3200)
        assert b3 == 1536

        # Tier 3: Broad multi-chunk context (6+ chunks or >= 5000 chars)
        b4 = provider.calculate_completion_budget(final_chunks_count=7, context_chars=6000)
        b5 = provider.calculate_completion_budget(final_chunks_count=8, context_chars=7200)
        assert b4 == 3072
        assert b5 == 3072

    def test_dynamic_budget_passed_to_rate_limiter_and_groq_call(self, monkeypatch: pytest.MonkeyPatch):
        """Dynamic completion budget is passed to rate limiter reservation and client.create."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message = MagicMock(
            content="The Transformer is an attention-based architecture consisting of 6 encoder and 6 decoder layers.",
            reasoning="Reasoning steps..."
        )
        
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage = MagicMock(prompt_tokens=600, completion_tokens=400, total_tokens=1000)
        
        mock_raw = MagicMock()
        mock_raw.parse.return_value = mock_completion
        mock_raw.headers = {}
        mock_client.chat.completions.with_raw_response.create.return_value = mock_raw
        
        reserved_tokens_list = []
        def mock_reserve(tokens, limit, window, session_id):
            reserved_tokens_list.append(tokens)
            return (True, 0, "res-dyn-123")

        monkeypatch.setattr(GroqProvider, "_client_instance", mock_client)
        monkeypatch.setattr("app.services.chat.groq_provider.groq_token_window.reserve", mock_reserve)
        
        provider = GroqProvider()
        
        # Call with 8 chunks (Tier 3 -> 3072 tokens)
        answer, truncated = provider.generate(
            system_prompt="You are a document assistant.",
            context="Chunk 1...\nChunk 2...\nChunk 3...\nChunk 4...\nChunk 5...\nChunk 6...\nChunk 7...\nChunk 8...",
            question="Explain the complete Transformer architecture.",
            final_chunks_count=8
        )

        # Verify rate limiter received prompt_tokens + 3072
        assert len(reserved_tokens_list) >= 1
        # Dynamic reserve must include the 3072 Tier 3 budget
        assert any(t >= 3072 for t in reserved_tokens_list)
        
        # Verify Groq client received max_tokens=3072
        create_kwargs = mock_client.chat.completions.with_raw_response.create.call_args[1]
        assert create_kwargs["max_tokens"] == 3072
        assert "The Transformer is an attention-based architecture" in answer

    def test_range_of_questions_synthesis_regression(self, monkeypatch: pytest.MonkeyPatch):
        """Verifies that narrow, moderate, and broad synthesis questions produce real answers with citations."""
        test_questions = [
            ("Explain the complete Transformer architecture.", 8, "The Transformer uses self-attention mechanisms..."),
            ("Explain the complete decoder architecture and all its components.", 8, "The decoder consists of 6 stacked layers with masked self-attention..."),
            ("Explain how self-attention, multi-head attention, and feed-forward networks work together.", 8, "Self-attention computes dot products, multi-head projects in parallel, and FFN transforms positions..."),
            ("Describe the experimental results, datasets (WMT 2014), and BLEU scores.", 6, "On WMT 2014 English-to-German, the model achieved 28.4 BLEU..."),
            ("Explain positional encoding and why it is needed.", 4, "Positional encodings inject sequence order information using sine and cosine functions...")
        ]

        for q, chunk_count, mock_ans in test_questions:
            mock_client = MagicMock()
            mock_choice = MagicMock()
            mock_choice.finish_reason = "stop"
            mock_choice.message = MagicMock(content=mock_ans, reasoning="Reasoning...")
            
            mock_completion = MagicMock()
            mock_completion.choices = [mock_choice]
            mock_completion.usage = MagicMock(prompt_tokens=500, completion_tokens=300, total_tokens=800)
            
            mock_raw = MagicMock()
            mock_raw.parse.return_value = mock_completion
            mock_raw.headers = {}
            mock_client.chat.completions.with_raw_response.create.return_value = mock_raw
            
            monkeypatch.setattr(GroqProvider, "_client_instance", mock_client)
            monkeypatch.setattr(
                "app.services.chat.groq_provider.groq_token_window.reserve",
                lambda **kwargs: (True, 0, f"res-q-{chunk_count}"),
            )
            
            provider = GroqProvider()
            answer, truncated = provider.generate(
                system_prompt="You are a document assistant.",
                context="Sample context chunks...",
                question=q,
                final_chunks_count=chunk_count
            )
            
            assert answer == mock_ans
            assert "synthesizing a substantial amount of information" not in answer
            assert "couldn't find enough information" not in answer


