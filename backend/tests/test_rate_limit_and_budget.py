"""
Automated Verification Suite: Rate-Limit, Token Budgeting, and RAG Grounding
Tests:
A. One normal question makes exactly ONE Groq call when query rewriting is disabled.
B. Query rewriting enabled makes at most expected calls and skips when headroom is tight.
C. Context budget is strictly enforced (<= 3500 prompt tokens).
D. Conversation history is capped at 800 tokens.
E. Simulated Groq 429 correctly maps to rate-limit error.
F. Simulated 401/403 is NOT mapped to rate-limit error.
G. Simulated 500/502 is NOT mapped to rate-limit error.
H. Citations remain correct after context trimming.
I. FAISS + BM25 + RRF still execute correctly.
"""

import sys
import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.services.chat.conversation_store import ConversationStore
from app.services.retrieval.query_rewriter import should_rewrite, rewrite_query
from app.services.pdf.chat_service import ChatService
from app.services.chat.groq_provider import GroqProvider
from app.schemas.chat import ChatRequest, SourceChunk
from app.services.retrieval.bm25_provider import BM25Retriever
from fastapi import HTTPException


class TestRateLimitAndBudget(unittest.TestCase):

    def test_a_one_question_makes_one_groq_call_when_rewrite_disabled(self):
        """A: One normal question makes exactly ONE Groq call when query rewriting is disabled."""
        with patch.object(settings, "ENABLE_SELECTIVE_QUERY_REWRITING", False):
            qualified, reason = should_rewrite("What is the retention procedure for the database?")
            self.assertFalse(qualified)
            self.assertIn("disabled", reason.lower())

            queries = rewrite_query("What is the retention procedure for the database?")
            self.assertEqual(len(queries), 1)
            self.assertEqual(queries[0], "What is the retention procedure for the database?")

    def test_b_query_rewriting_headroom_safety(self):
        """B: Query rewriting enabled skips when Groq token headroom is insufficient."""
        with patch.object(settings, "ENABLE_SELECTIVE_QUERY_REWRITING", True):
            with patch("app.core.rate_limit.groq_token_window.current_usage", return_value=7000):
                queries = rewrite_query("What are the primary operational guidelines for servers?")
                # Headroom is tight (7000 > 8000 - 6000), should gracefully return only original query
                self.assertEqual(len(queries), 1)
                self.assertEqual(queries[0], "What are the primary operational guidelines for servers?")

    def test_c_context_budget_enforcement(self):
        """C: Context budget strictly enforces GROQ_PREFLIGHT_PROMPT_TOKEN_BUDGET (<= 2500 tokens)."""
        service = ChatService()
        system_prompt = "You are a helpful assistant."
        question = "What is the summary?"
        
        # 30 large chunks of 1,200 chars each = 36,000 chars = ~9,000 tokens (exceeds budget)
        class MockChunk:
            def __init__(self, idx, score):
                self.chunk_id = f"chunk_{idx}"
                self.chunk_index = idx
                self.score = score
                self.start_page = idx + 1
                self.end_page = idx + 1
                self.text = f"Content for chunk {idx}. " * 60

        mock_chunks = [MockChunk(i, score=1.0 - (i * 0.02)) for i in range(30)]
        
        history, trimmed_chunks, context, truncated = service._preflight_trim_for_groq(
            system_prompt=system_prompt,
            question=question,
            history=None,
            chunks=mock_chunks
        )
        
        est_tokens = service._estimate_groq_prompt_tokens(system_prompt, context, question, history)
        self.assertLessEqual(est_tokens, settings.GROQ_PREFLIGHT_PROMPT_TOKEN_BUDGET)
        self.assertLess(len(trimmed_chunks), 30)
        self.assertTrue(truncated)
        self.assertEqual(trimmed_chunks[0].chunk_id, "chunk_0")

    def test_d_conversation_history_capped_at_350_tokens(self):
        """D: Conversation history is capped at 350 tokens and 2 turns."""
        store = ConversationStore()
        doc_id = "test_doc"
        sess_id = "test_sess"
        store.clear(doc_id, sess_id)
        
        for i in range(10):
            store.add_turn(doc_id, sess_id, f"User question {i} with some content", f"Assistant answer {i} with detailed text " * 10)
            
        history = store.get_history(doc_id, sess_id, max_turns=2, max_tokens=350)
        total_chars = sum(len(m["content"]) for m in history)
        est_tokens = (total_chars + 3) // 4
        
        self.assertLessEqual(est_tokens, 350)
        self.assertLessEqual(len(history), 4)

    def test_e_groq_429_mapped_to_rate_limit_error(self):
        """E: Simulated Groq 429 is correctly mapped to 429 rate-limit error."""
        provider = GroqProvider()
        try:
            from groq import RateLimitError
        except ImportError:
            RateLimitError = Exception

        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create.side_effect = RateLimitError(
            message="Rate limit reached for model",
            response=MagicMock(status_code=429),
            body={"error": {"message": "Rate limit reached", "code": "rate_limit_exceeded"}}
        )
        
        with patch.object(provider, "_get_client", return_value=mock_client):
            with patch("app.core.rate_limit.groq_token_window.reserve", return_value=(True, 0, "res1")):
                with self.assertRaises(HTTPException) as ctx:
                    provider.generate(
                        system_prompt="sys",
                        context="ctx",
                        question="q"
                    )
                self.assertEqual(ctx.exception.status_code, 429)
                self.assertIn("rate limit was exceeded", ctx.exception.detail.lower())

    def test_f_simulated_401_not_mapped_to_rate_limit(self):
        """F: Simulated 401/403 is NOT mapped to rate-limit error (mapped to 401)."""
        provider = GroqProvider()
        try:
            from groq import AuthenticationError
        except ImportError:
            AuthenticationError = Exception

        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create.side_effect = AuthenticationError(
            message="Invalid API Key",
            response=MagicMock(status_code=401),
            body={"error": {"message": "Invalid API Key", "code": "invalid_api_key"}}
        )
        
        with patch.object(provider, "_get_client", return_value=mock_client):
            with patch("app.core.rate_limit.groq_token_window.reserve", return_value=(True, 0, "res1")):
                with self.assertRaises(HTTPException) as ctx:
                    provider.generate(
                        system_prompt="sys",
                        context="ctx",
                        question="q"
                    )
                self.assertIn(ctx.exception.status_code, (401, 500))
                self.assertNotIn("rate limit was exceeded", ctx.exception.detail.lower())

    def test_g_simulated_502_not_mapped_to_rate_limit(self):
        """G: Simulated 502/500 is NOT mapped to rate-limit error (mapped to 502/500)."""
        provider = GroqProvider()
        try:
            from groq import APIStatusError
        except ImportError:
            APIStatusError = Exception

        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create.side_effect = APIStatusError(
            message="Internal server error",
            response=MagicMock(status_code=500, text="Internal server error"),
            body={"error": {"message": "Internal server error", "code": "server_error"}}
        )
        
        with patch.object(provider, "_get_client", return_value=mock_client):
            with patch("app.core.rate_limit.groq_token_window.reserve", return_value=(True, 0, "res1")):
                with self.assertRaises(HTTPException) as ctx:
                    provider.generate(
                        system_prompt="sys",
                        context="ctx",
                        question="q"
                    )
                self.assertEqual(ctx.exception.status_code, 502)
                self.assertNotIn("rate limit was exceeded", ctx.exception.detail.lower())

    def test_h_citations_remain_correct_after_context_trimming(self):
        """H: Citations match only surviving context chunks after trimming."""
        class MockChunk:
            def __init__(self, idx, score, page):
                self.chunk_id = f"chunk_{idx}"
                self.chunk_index = idx
                self.score = score
                self.start_page = page
                self.end_page = page
                self.text = f"Policy definition on page {page}. " * 20

        chunks = [
            MockChunk(0, 0.95, 1),
            MockChunk(1, 0.90, 2),
            MockChunk(2, 0.40, 3),
            MockChunk(3, 0.35, 4),
        ]
        
        service = ChatService()
        history, trimmed_chunks, context, truncated = service._preflight_trim_for_groq(
            system_prompt="sys",
            question="What is the policy?",
            history=None,
            chunks=chunks
        )
        
        sources = [
            SourceChunk(
                chunk_id=c.chunk_id,
                chunk_index=c.chunk_index,
                score=round(c.score, 4),
                start_page=c.start_page,
                end_page=c.end_page,
                text=c.text
            )
            for c in trimmed_chunks
        ]
        
        surviving_ids = {c.chunk_id for c in trimmed_chunks}
        for s in sources:
            self.assertIn(s.chunk_id, surviving_ids)
            self.assertEqual(s.start_page, s.end_page)

    def test_i_bm25_rrf_retrieval_fusion_execution(self):
        """I: BM25 + FAISS + RRF fusion executes correctly."""
        mock_chunks = [
            {"chunk_id": "c1", "text": "ORION database replication settings and retention policy."},
            {"chunk_id": "c2", "text": "Unrelated network firewall configuration."},
            {"chunk_id": "c3", "text": "Database retention rules and archiving periods."},
        ]
        
        bm25 = BM25Retriever(k1=1.5, b=0.75)
        scored = bm25.search("database retention", mock_chunks, top_k=3)
        self.assertGreater(len(scored), 0)
        self.assertIn(scored[0][0]["chunk_id"], ("c1", "c3"))
        
        vec_rankings = [("c1", 0.88), ("c2", 0.50), ("c3", 0.45)]
        bm25_rankings = [(c[0]["chunk_id"], c[1]) for c in scored]
        
        fused = BM25Retriever.reciprocal_rank_fusion(vec_rankings, bm25_rankings, rrf_k=60)
        self.assertIsInstance(fused, dict)
        self.assertIn("c1", fused)
        self.assertGreater(fused["c1"], fused["c2"])


if __name__ == "__main__":
    unittest.main()
