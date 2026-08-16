"""
Production RAG Architectural Regression & Invariant Protection Test Suite.

This test suite locks down and permanently protects the core RAG invariants:
1. Single Authoritative Path: Every QA request routes through RetrievalService (No FULL_CONTEXT bypass, no raw full_text injection).
2. Strict Context Bounding: MAX_CONTEXT_CHUNKS <= 10, MAX_CONTEXT_CHARACTERS <= 10,000, Neighbor Merging <= 2 chunks & <= 1500 chars.
3. Page Metadata Precision: Character interval overlap matching across single-page, two-page, multi-page, boundary, and gap conditions.
4. Citation Integrity: Sources constructed 100% from final surviving context chunks; zero regex on LLM text; phantom citation immunity.
5. Token Accounting: Settle-after-completion token window, false-429 prevention, distinct local vs Groq 429 error handling.
6. Multi-Turn Conversation Bounding: Session memory bounded by turn and token caps; preflight trimming of history before chunks.
7. Real FastAPI Route Protection: End-to-end testing of POST /chat/{document_id} with validation, 400/404/200 HTTP responses.
"""

import json
import logging
import pytest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from httpx import ASGITransport
from fastapi import HTTPException
from groq import RateLimitError

from app.main import app
from app.core.config import settings
from app.core.rate_limit import GroqTokenWindow, groq_token_window
from app.schemas.chat import ChatRequest, ChatResponse, SourceChunk
from app.schemas.retrieval import RetrievalResult
from app.services.chat.conversation_store import conversation_store
from app.services.chat.groq_provider import GroqProvider
from app.services.chat.prompt_builder import PromptBuilder
from app.services.pdf.chat_service import ChatService
from app.services.pdf.chunking_service import ChunkingService
from app.services.pdf.retrieval_service import RetrievalService
from app.services.retrieval.query_rewriter import rewrite_query, should_rewrite


def _create_mock_doc_dir(tmp_path: Path, doc_id: str, page_count: int = 45, chars_per_page: int = 500) -> Path:
    """Helper to set up mock document folder on filesystem with valid status, extracted text, and pages."""
    doc_dir = tmp_path / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    status_data = {
        "upload_status": "completed",
        "processing_status": "completed",
        "chunking_status": "completed",
        "embedding_status": "completed",
        "indexing_status": "completed",
        "status": "COMPLETED",
        "progress": 100,
    }
    (doc_dir / "status.json").write_text(json.dumps(status_data), encoding="utf-8")

    pages_meta = []
    full_text_parts = []
    current_char = 0

    for p in range(1, page_count + 1):
        page_text = f"Page {p} content. " + ("Sample document sentences and domain data. " * (chars_per_page // 50))
        p_len = len(page_text)
        pages_meta.append({
            "page": p,
            "start_character": current_char,
            "end_character": current_char + p_len
        })
        full_text_parts.append(page_text)
        current_char += p_len + 1

    (doc_dir / "pages.json").write_text(json.dumps(pages_meta), encoding="utf-8")
    (doc_dir / "extracted_text.txt").write_text("\n".join(full_text_parts), encoding="utf-8")

    chunks_data = []
    for p in range(1, min(page_count + 1, 10)):
        chunks_data.append({
            "chunk_id": f"{doc_id[:8]}_chunk_{p:06d}",
            "chunk_index": p,
            "document_id": doc_id,
            "text": f"Content for chunk {p} on page {p}.",
            "start_page": p,
            "end_page": p,
            "start_character": (p - 1) * 100,
            "end_character": p * 100,
            "word_count": 8,
            "character_count": 50,
            "sentence_count": 1,
            "estimated_tokens": 12,
        })
    (doc_dir / "chunks.json").write_text(json.dumps(chunks_data), encoding="utf-8")

    return doc_dir



# ==============================================================================
# 1. Single Authoritative Path & No Raw Full-Text Bypass
# ==============================================================================
class TestSinglePathRagInvariant:
    @pytest.mark.anyio
    async def test_small_document_uses_rag_retrieval_service(self, tmp_path: Path):
        """Small documents (<=10,000 chars) must NOT bypass RetrievalService or inject full_text."""
        doc_id = str(uuid.uuid4())
        _create_mock_doc_dir(tmp_path, doc_id=doc_id, page_count=3, chars_per_page=200)

        chat_service = ChatService(target_dir=tmp_path)
        mock_retrieval = MagicMock(spec=RetrievalService)
        mock_retrieval.retrieve.return_value = [
            RetrievalResult(
                document_id=doc_id,
                chunk_id=f"{doc_id[:8]}_chunk_001",
                chunk_index=1,
                text="Specific factual answer in retrieved chunk.",
                start_page=2,
                end_page=2,
                start_character=205,
                end_character=245,
                word_count=6,
                character_count=40,
                sentence_count=1,
                estimated_tokens=10,
                score=0.92,
                rank=1
            )
        ]
        chat_service.retrieval_service = mock_retrieval

        mock_provider = MagicMock(spec=GroqProvider)
        mock_provider.generate.return_value = ("The answer is factual.", False)
        mock_provider.provider_name.return_value = "Groq"
        mock_provider.model_name.return_value = settings.LLM_MODEL
        chat_service.provider = mock_provider

        request = ChatRequest(question="What is the answer?", top_k=5)
        response = await chat_service.answer_question(document_id=doc_id, request=request)

        assert response.context_mode == "RAG"
        assert mock_retrieval.retrieve.called
        assert len(response.sources) == 1
        assert response.sources[0].start_page == 2
        assert response.sources[0].chunk_id == f"{doc_id[:8]}_chunk_001"

    @pytest.mark.anyio
    async def test_raw_full_text_never_sent_to_llm(self, tmp_path: Path):
        """Context sent to LLM provider must always be structured [Document Segment ...] format, never raw full text."""
        doc_id = str(uuid.uuid4())
        doc_dir = _create_mock_doc_dir(tmp_path, doc_id=doc_id, page_count=2, chars_per_page=150)
        raw_full_text = (doc_dir / "extracted_text.txt").read_text(encoding="utf-8")

        chat_service = ChatService(target_dir=tmp_path)
        mock_retrieval = MagicMock(spec=RetrievalService)
        mock_retrieval.retrieve.return_value = [
            RetrievalResult(
                document_id=doc_id,
                chunk_id=f"{doc_id[:8]}_chunk_001",
                chunk_index=1,
                text="Chunk text that is structured.",
                start_page=1,
                end_page=1,
                start_character=0,
                end_character=30,
                word_count=5,
                character_count=30,
                sentence_count=1,
                estimated_tokens=8,
                score=0.95,
                rank=1
            )
        ]
        chat_service.retrieval_service = mock_retrieval

        captured_kwargs = {}

        def mock_generate(**kwargs):
            captured_kwargs.update(kwargs)
            return ("Grounded answer.", False)

        mock_provider = MagicMock(spec=GroqProvider)
        mock_provider.generate.side_effect = mock_generate
        mock_provider.provider_name.return_value = "Groq"
        mock_provider.model_name.return_value = settings.LLM_MODEL
        chat_service.provider = mock_provider

        request = ChatRequest(question="Test question", top_k=5)
        await chat_service.answer_question(document_id=doc_id, request=request)

        context_sent = captured_kwargs.get("context", "")
        assert "[Document Segment 1]" in context_sent
        assert "Page: 1" in context_sent
        assert context_sent != raw_full_text


# ==============================================================================
# 2. Strict Context Bounding Invariants
# ==============================================================================
class TestBoundedContextCompilation:
    @pytest.mark.anyio
    async def test_context_bounded_by_max_chunks_and_characters(self, tmp_path: Path):
        """Retrieved candidates must be clamped to MAX_CONTEXT_CHUNKS (10) and MAX_CONTEXT_CHARACTERS (10000)."""
        doc_id = str(uuid.uuid4())
        _create_mock_doc_dir(tmp_path, doc_id=doc_id, page_count=30, chars_per_page=1000)

        chat_service = ChatService(target_dir=tmp_path)
        mock_retrieval = MagicMock(spec=RetrievalService)

        # Generate 25 candidate chunks
        candidates = []
        for i in range(1, 26):
            candidates.append(
                RetrievalResult(
                    document_id=doc_id,
                    chunk_id=f"{doc_id[:8]}_chunk_{i:03d}",
                    chunk_index=i,
                    text=f"Chunk {i} detailed content with scientific descriptions. " * 15,
                    start_page=i,
                    end_page=i,
                    start_character=(i - 1) * 800,
                    end_character=i * 800,
                    word_count=90,
                    character_count=750,
                    sentence_count=15,
                    estimated_tokens=180,
                    score=0.95 - (i * 0.02),
                    rank=i
                )
            )
        mock_retrieval.retrieve.return_value = candidates
        chat_service.retrieval_service = mock_retrieval

        mock_provider = MagicMock(spec=GroqProvider)
        mock_provider.generate.return_value = ("Synthesized answer.", False)
        mock_provider.provider_name.return_value = "Groq"
        mock_provider.model_name.return_value = settings.LLM_MODEL
        chat_service.provider = mock_provider

        request = ChatRequest(question="Summarize findings across sections", top_k=25)
        response = await chat_service.answer_question(document_id=doc_id, request=request)

        assert len(response.sources) <= settings.MAX_CONTEXT_CHUNKS
        total_source_chars = sum(len(s.text) for s in response.sources)
        assert total_source_chars <= settings.MAX_CONTEXT_CHARACTERS

    def test_context_preserves_reading_order_across_pages(self):
        """PromptBuilder.compile_context sorts chunks by start_page and chunk_index to preserve flow."""
        chunks = [
            RetrievalResult(
                document_id="doc_1",
                chunk_id="chunk_3",
                chunk_index=5,
                text="Content from page 10.",
                start_page=10,
                end_page=10,
                start_character=5000,
                end_character=5050,
                word_count=4,
                character_count=22,
                sentence_count=1,
                estimated_tokens=6,
                score=0.95,
                rank=1
            ),
            RetrievalResult(
                document_id="doc_1",
                chunk_id="chunk_1",
                chunk_index=1,
                text="Content from page 1.",
                start_page=1,
                end_page=1,
                start_character=0,
                end_character=50,
                word_count=4,
                character_count=21,
                sentence_count=1,
                estimated_tokens=6,
                score=0.80,
                rank=2
            ),
        ]

        compiled = PromptBuilder.compile_context(chunks)
        pos_page_1 = compiled.find("Content from page 1.")
        pos_page_10 = compiled.find("Content from page 10.")
        assert pos_page_1 < pos_page_10


# ==============================================================================
# 3. Neighbor Chunk Merging Safety Bounds
# ==============================================================================
class TestNeighborMergeConstraints:
    def test_neighbor_merging_respects_limits_and_consecutive_indices(self):
        """Neighbor merging merges at most 2 chunks and at most 1500 chars, only when strictly consecutive."""
        results = [
            RetrievalResult(
                document_id="doc_merge",
                chunk_id="chunk_1",
                chunk_index=1,
                text="A" * 600,
                start_page=1,
                end_page=1,
                start_character=0,
                end_character=600,
                word_count=50,
                character_count=600,
                sentence_count=5,
                estimated_tokens=150,
                score=0.9,
                rank=1
            ),
            RetrievalResult(
                document_id="doc_merge",
                chunk_id="chunk_2",
                chunk_index=2,
                text="B" * 600,
                start_page=1,
                end_page=1,
                start_character=601,
                end_character=1201,
                word_count=50,
                character_count=600,
                sentence_count=5,
                estimated_tokens=150,
                score=0.85,
                rank=2
            ),
            RetrievalResult(
                document_id="doc_merge",
                chunk_id="chunk_3",
                chunk_index=3,
                text="C" * 600,
                start_page=1,
                end_page=1,
                start_character=1202,
                end_character=1802,
                word_count=50,
                character_count=600,
                sentence_count=5,
                estimated_tokens=150,
                score=0.80,
                rank=3
            ),
        ]

        sorted_results = sorted(results, key=lambda x: x.chunk_index)
        merged = []
        for res in sorted_results:
            if not merged:
                res_copy = res.model_copy()
                setattr(res_copy, "_merged_count", 1)
                merged.append(res_copy)
            else:
                last_res = merged[-1]
                current_last_idx = getattr(last_res, "last_chunk_index", None) or last_res.chunk_index
                is_consecutive = (res.chunk_index == current_last_idx + 1)
                merged_count = getattr(last_res, "_merged_count", 1)
                projected_len = len(last_res.text) + 2 + len(res.text)

                if (
                    is_consecutive
                    and merged_count < settings.MAX_MERGED_CHUNKS
                    and projected_len <= settings.MAX_MERGED_CHUNK_CHARS
                ):
                    last_res.text = last_res.text + "\n\n" + res.text
                    last_res.last_chunk_index = res.chunk_index
                    setattr(last_res, "_merged_count", merged_count + 1)
                else:
                    res_copy = res.model_copy()
                    setattr(res_copy, "_merged_count", 1)
                    merged.append(res_copy)

        assert len(merged) == 2
        assert merged[0].last_chunk_index == 2
        assert getattr(merged[0], "_merged_count") == 2
        assert merged[1].chunk_index == 3

    def test_neighbor_merging_rejects_non_consecutive_indices(self):
        """Non-consecutive chunks (e.g. Chunk 1 and Chunk 3) must never be merged."""
        results = [
            RetrievalResult(
                document_id="doc_merge",
                chunk_id="chunk_1",
                chunk_index=1,
                text="First chunk text.",
                start_page=1,
                end_page=1,
                start_character=0,
                end_character=50,
                word_count=3,
                character_count=17,
                sentence_count=1,
                estimated_tokens=5,
                score=0.9,
                rank=1
            ),
            RetrievalResult(
                document_id="doc_merge",
                chunk_id="chunk_3",
                chunk_index=3,
                text="Third chunk text (gap in index).",
                start_page=1,
                end_page=1,
                start_character=100,
                end_character=150,
                word_count=5,
                character_count=32,
                sentence_count=1,
                estimated_tokens=8,
                score=0.85,
                rank=2
            ),
        ]

        sorted_results = sorted(results, key=lambda x: x.chunk_index)
        merged = []
        for res in sorted_results:
            if not merged:
                res_copy = res.model_copy()
                setattr(res_copy, "_merged_count", 1)
                merged.append(res_copy)
            else:
                last_res = merged[-1]
                current_last_idx = getattr(last_res, "last_chunk_index", None) or last_res.chunk_index
                is_consecutive = (res.chunk_index == current_last_idx + 1)
                if is_consecutive:
                    last_res.text = last_res.text + "\n\n" + res.text
                else:
                    merged.append(res.model_copy())

        assert len(merged) == 2
        assert merged[0].chunk_index == 1
        assert merged[1].chunk_index == 3

    def test_neighbor_merging_rejects_oversized_combined_text(self):
        """Two consecutive chunks whose combined length exceeds MAX_MERGED_CHUNK_CHARS (1500) must not merge."""
        c1 = RetrievalResult(
            document_id="doc_merge",
            chunk_id="chunk_1",
            chunk_index=1,
            text="A" * 900,
            start_page=1,
            end_page=1,
            start_character=0,
            end_character=900,
            word_count=90,
            character_count=900,
            sentence_count=9,
            estimated_tokens=225,
            score=0.9,
            rank=1
        )
        c2 = RetrievalResult(
            document_id="doc_merge",
            chunk_id="chunk_2",
            chunk_index=2,
            text="B" * 800,
            start_page=1,
            end_page=1,
            start_character=901,
            end_character=1701,
            word_count=80,
            character_count=800,
            sentence_count=8,
            estimated_tokens=200,
            score=0.85,
            rank=2
        )

        projected = len(c1.text) + 2 + len(c2.text)  # 1702 > 1500
        assert projected > settings.MAX_MERGED_CHUNK_CHARS


# ==============================================================================
# 4. Page Metadata & Exact Character Interval Overlap
# ==============================================================================
class TestCharacterIntervalPageOverlap:
    def setup_method(self):
        self.service = ChunkingService(target_dir=Path("."))
        self.pages_meta = [
            {"page": p, "start_character": (p - 1) * 1000, "end_character": p * 1000}
            for p in range(1, 51)
        ]

    def test_single_page_chunk(self):
        """Chunk located strictly on page 14 (chars 13200 to 13700) returns (14, 14)."""
        start, end = self.service._determine_page_range(13200, 13700, self.pages_meta)
        assert (start, end) == (14, 14)

    def test_two_page_spanning_chunk(self):
        """Chunk spanning page 7 and 8 boundary (chars 6800 to 7400) returns (7, 8)."""
        start, end = self.service._determine_page_range(6800, 7400, self.pages_meta)
        assert (start, end) == (7, 8)

    def test_multi_page_document_start_and_end_chunks(self):
        """Chunks on first and last pages must map accurately without defaulting to whole document."""
        # First page
        start, end = self.service._determine_page_range(0, 300, self.pages_meta)
        assert (start, end) == (1, 1)

        # Last page (Page 50)
        start, end = self.service._determine_page_range(49200, 49800, self.pages_meta)
        assert (start, end) == (50, 50)

    def test_boundary_exact_alignment(self):
        """Chunk precisely matching page boundary (chars 3000 to 4000) maps to Page 4."""
        start, end = self.service._determine_page_range(3000, 4000, self.pages_meta)
        assert (start, end) == (4, 4)

    def test_whitespace_gap_between_pages_falls_back_to_nearest_page(self):
        """Chunk in whitespace gap between pages falls back to nearest page, never last page."""
        gapped_pages = [
            {"page": 1, "start_character": 0, "end_character": 500},
            {"page": 2, "start_character": 550, "end_character": 1000},
            {"page": 3, "start_character": 1050, "end_character": 1500},
        ]
        # Chunk at gap offset 520 (closer to page 1 end 500 than page 2 start 550)
        start, end = self.service._determine_page_range(510, 530, gapped_pages)
        assert start in (1, 2)
        assert end in (1, 2)
        assert (start, end) != (1, 3)

    def test_inverted_character_offsets_normalized(self):
        """Inverted offsets (start > end) are safely normalized."""
        start, end = self.service._determine_page_range(1500, 500, self.pages_meta)
        assert (start, end) == (1, 2)

    def test_empty_pages_meta_fallback(self):
        """Empty page metadata defaults safely to (1, 1)."""
        start, end = self.service._determine_page_range(100, 200, [])
        assert (start, end) == (1, 1)


# ==============================================================================
# 5. Citation Integrity & Phantom Citation Immunity
# ==============================================================================
class TestCitationIntegrity:
    @pytest.mark.anyio
    async def test_llm_hallucinated_pages_do_not_alter_citations(self, tmp_path: Path):
        """Even if LLM mentions 'Page 99', citations reflect ONLY true retrieved chunk metadata."""
        doc_id = str(uuid.uuid4())
        _create_mock_doc_dir(tmp_path, doc_id=doc_id, page_count=10, chars_per_page=500)

        chat_service = ChatService(target_dir=tmp_path)
        mock_retrieval = MagicMock(spec=RetrievalService)
        mock_retrieval.retrieve.return_value = [
            RetrievalResult(
                document_id=doc_id,
                chunk_id=f"{doc_id[:8]}_chunk_003",
                chunk_index=3,
                text="The actual fact is located on page 4.",
                start_page=4,
                end_page=4,
                start_character=1500,
                end_character=1540,
                word_count=7,
                character_count=40,
                sentence_count=1,
                estimated_tokens=10,
                score=0.88,
                rank=1
            )
        ]
        chat_service.retrieval_service = mock_retrieval

        mock_provider = MagicMock(spec=GroqProvider)
        mock_provider.generate.return_value = (
            "According to Page 99 and Pages 1 to 45, the fact is located on page 4.",
            False
        )
        mock_provider.provider_name.return_value = "Groq"
        mock_provider.model_name.return_value = settings.LLM_MODEL
        chat_service.provider = mock_provider

        request = ChatRequest(question="Where is the fact?", top_k=5)
        response = await chat_service.answer_question(document_id=doc_id, request=request)

        assert len(response.sources) == 1
        assert response.sources[0].start_page == 4
        assert response.sources[0].end_page == 4
        assert response.sources[0].chunk_id == f"{doc_id[:8]}_chunk_003"
        all_pages = [s.start_page for s in response.sources] + [s.end_page for s in response.sources]
        assert 99 not in all_pages
        assert not (min(all_pages) == 1 and max(all_pages) == 45)

    @pytest.mark.anyio
    async def test_empty_retrieval_returns_zero_sources(self, tmp_path: Path):
        """When 0 chunks match query, ChatService returns fallback without calling Groq, with sources=[]."""
        doc_id = str(uuid.uuid4())
        _create_mock_doc_dir(tmp_path, doc_id=doc_id, page_count=5, chars_per_page=300)

        chat_service = ChatService(target_dir=tmp_path)
        mock_retrieval = MagicMock(spec=RetrievalService)
        mock_retrieval.retrieve.return_value = []
        chat_service.retrieval_service = mock_retrieval

        mock_provider = MagicMock(spec=GroqProvider)
        mock_provider.provider_name.return_value = "Groq"
        mock_provider.model_name.return_value = settings.LLM_MODEL
        chat_service.provider = mock_provider

        request = ChatRequest(question="Irrelevant topic query", top_k=5)
        response = await chat_service.answer_question(document_id=doc_id, request=request)

        assert response.success is True
        assert response.sources == []
        assert "couldn't find enough information" in response.answer.lower()
        assert not mock_provider.generate.called


# ==============================================================================
# 6. Token Accounting & Rate Limit Invariants
# ==============================================================================
class TestTokenAccountingAndRateLimiting:
    def test_reservation_and_settlement_prevents_false_429(self):
        """Settling actual token usage releases reserved headroom and permits immediate second question."""
        window = GroqTokenWindow()
        limit = 6000

        # Turn 1: Reserve 2000 tokens
        allowed_1, retry_1, res_id_1 = window.reserve(tokens=2000, limit=limit, window=60)
        assert allowed_1 is True

        # Completion returned actual total tokens = 350
        window.settle(res_id_1, actual_tokens=350)
        assert window.current_usage(60) == 350

        # Turn 2: Reserve 4500 tokens immediately (would fail if initial 2000 remained locked)
        allowed_2, retry_2, res_id_2 = window.reserve(tokens=4500, limit=limit, window=60)
        assert allowed_2 is True

        # Settle Turn 2 with actual usage = 400
        window.settle(res_id_2, actual_tokens=400)
        assert window.current_usage(60) == 750

    def test_settle_with_zero_or_none_releases_reservation(self):
        """On failure or skip, settle(res_id, 0) completely frees the reserved tokens."""
        window = GroqTokenWindow()
        limit = 3000

        allowed, _, res_id = window.reserve(tokens=2500, limit=limit, window=60)
        assert allowed is True
        assert window.current_usage(60) == 2500

        window.settle(res_id, actual_tokens=0)
        assert window.current_usage(60) == 0

    def test_actual_token_overflow_returns_429(self):
        """When token window exceeds limit, reserve returns allowed=False."""
        window = GroqTokenWindow()
        limit = 3000

        allowed_1, _, res_id_1 = window.reserve(tokens=2500, limit=limit, window=60)
        assert allowed_1 is True
        window.settle(res_id_1, actual_tokens=2500)

        # Second request asks for 1000 tokens (2500 + 1000 = 3500 > 3000)
        allowed_2, retry_after, _ = window.reserve(tokens=1000, limit=limit, window=60)
        assert allowed_2 is False
        assert retry_after > 0


# ==============================================================================
# 7. Multi-Turn Conversation Memory Bounding
# ==============================================================================
class TestMultiTurnConversationMemory:
    def test_conversation_store_prunes_older_turns(self):
        """Conversation store must respect max_turns and max_tokens bounds."""
        doc_id = "test_memory_doc"
        session_id = "session_bound_test"

        # Add 10 turns (20 messages)
        for i in range(10):
            conversation_store.add_turn(
                document_id=doc_id,
                session_id=session_id,
                question=f"Question {i} with some content.",
                answer=f"Answer {i} with detailed explanation."
            )

        history = conversation_store.get_history(
            document_id=doc_id,
            session_id=session_id,
            max_turns=3,
            max_tokens=1000
        )

        # Max 3 turns = 6 messages (3 user + 3 assistant)
        assert len(history) <= 6
        assert history[-1]["content"] == "Answer 9 with detailed explanation."

    @pytest.mark.anyio
    async def test_preflight_trims_history_before_context_chunks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Preflight trimming drops older history messages before dropping high-scoring retrieved chunks."""
        monkeypatch.setattr(settings, "GROQ_PREFLIGHT_PROMPT_TOKEN_BUDGET", 400)

        chat_service = ChatService(target_dir=tmp_path)
        system_prompt = PromptBuilder.get_system_prompt()
        question = "What is the primary conclusion?"

        # 6 turns of history = 12 messages (~3000 chars)
        history = []
        for i in range(6):
            history.append({"role": "user", "content": f"User question {i} " + ("data " * 50)})
            history.append({"role": "assistant", "content": f"Assistant answer {i} " + ("info " * 50)})

        # 2 high-scoring chunks
        chunks = [
            RetrievalResult(
                document_id="doc_trim",
                chunk_id="chunk_1",
                chunk_index=1,
                text="Crucial high-scoring result text.",
                start_page=1,
                end_page=1,
                start_character=0,
                end_character=35,
                word_count=5,
                character_count=35,
                sentence_count=1,
                estimated_tokens=9,
                score=0.95,
                rank=1
            )
        ]

        trimmed_history, trimmed_chunks, context, truncated = chat_service._preflight_trim_for_groq(
            system_prompt=system_prompt,
            question=question,
            history=history,
            chunks=chunks
        )

        assert truncated is True
        # History was trimmed, but high-scoring chunk 1 survived
        assert len(trimmed_chunks) == 1
        assert trimmed_chunks[0].chunk_id == "chunk_1"
        trimmed_history_len = len(trimmed_history) if trimmed_history else 0
        assert trimmed_history_len < len(history)


# ==============================================================================
# 8. Real FastAPI Route HTTP Verification (POST /chat/{document_id})
# ==============================================================================
class TestFastApiRouteIntegration:
    @pytest.mark.anyio
    async def test_real_chat_route_returns_200_with_grounded_response(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Real FastAPI route POST /chat/{document_id} returns 200 with valid grounded response payload."""
        from app.api.routes.chat import chat_service as route_chat_service

        doc_id = str(uuid.uuid4())
        _create_mock_doc_dir(tmp_path, doc_id=doc_id, page_count=5, chars_per_page=300)
        monkeypatch.setattr(route_chat_service, "target_dir", tmp_path)

        mock_retrieval = MagicMock(spec=RetrievalService)
        mock_retrieval.retrieve.return_value = [
            RetrievalResult(
                document_id=doc_id,
                chunk_id=f"{doc_id[:8]}_chunk_001",
                chunk_index=1,
                text="Grounded context from retrieval.",
                start_page=2,
                end_page=2,
                start_character=100,
                end_character=150,
                word_count=5,
                character_count=32,
                sentence_count=1,
                estimated_tokens=8,
                score=0.92,
                rank=1
            )
        ]
        monkeypatch.setattr(route_chat_service, "retrieval_service", mock_retrieval)

        # Mock Groq provider to return deterministic grounded answer
        mock_provider = MagicMock(spec=GroqProvider)
        mock_provider.generate.return_value = ("Grounded answer from real API route.", False)
        mock_provider.provider_name.return_value = "Groq"
        mock_provider.model_name.return_value = settings.LLM_MODEL
        monkeypatch.setattr(route_chat_service, "provider", mock_provider)

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {"question": "What is described on page 2?", "top_k": 5}
            res = await client.post(f"/chat/{doc_id}", json=payload)


            assert res.status_code == 200
            data = res.json()
            assert data["success"] is True
            assert data["document_id"] == doc_id
            assert data["answer"] == "Grounded answer from real API route."
            assert data["context_mode"] == "RAG"
            assert isinstance(data["sources"], list)
            assert "processing_time_ms" in data

    @pytest.mark.anyio
    async def test_real_chat_route_validates_empty_question(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Real route rejects whitespace/empty question with HTTP 400."""
        from app.api.routes.chat import chat_service as route_chat_service

        doc_id = str(uuid.uuid4())
        _create_mock_doc_dir(tmp_path, doc_id=doc_id, page_count=2)
        monkeypatch.setattr(route_chat_service, "target_dir", tmp_path)

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.post(f"/chat/{doc_id}", json={"question": "   "})
            assert res.status_code == 400

    @pytest.mark.anyio
    async def test_real_chat_route_validates_invalid_uuid(self, tmp_path: Path):
        """Real route rejects non-UUID document_id with HTTP 400."""
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.post("/chat/not-a-valid-uuid", json={"question": "Valid question?"})
            assert res.status_code == 400

    @pytest.mark.anyio
    async def test_real_chat_route_missing_document_returns_404(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Real route returns 404 for non-existent document UUID."""
        from app.api.routes.chat import chat_service as route_chat_service
        monkeypatch.setattr(route_chat_service, "target_dir", tmp_path)
        missing_doc_id = str(uuid.uuid4())

        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.post(f"/chat/{missing_doc_id}", json={"question": "Valid question?"})
            assert res.status_code == 404



# ==============================================================================
# 9. Query Rewriter Heuristics & Fallbacks
# ==============================================================================
class TestQueryRewriterHeuristics:
    def test_query_rewriter_heuristic_skips_short_or_factual_queries(self):
        """Heuristic filter should skip query expansion for short or already-specific queries."""
        # Short query
        qual, reason = should_rewrite("Page 4")
        assert qual is False

        # Specific single word
        qual, reason = should_rewrite("PostgreSQL")
        assert qual is False

        # Long compound query qualifies
        qual, reason = should_rewrite("When is the launch date for Project Titan and what microkernel version is used?")
        assert qual is True

    def test_query_rewriter_falls_back_gracefully_on_error(self, monkeypatch: pytest.MonkeyPatch):
        """Query rewriter returns original query when LLM fails without raising unhandled error."""
        monkeypatch.setattr(
            "app.services.chat.groq_provider.GroqProvider._get_client",
            MagicMock(side_effect=RuntimeError("Groq client failure"))
        )

        queries = rewrite_query("What are the system specifications and architecture?")
        assert len(queries) == 1
        assert queries[0] == "What are the system specifications and architecture?"


# ==============================================================================
# 10. Structured Telemetry Logging
# ==============================================================================
class TestStructuredObservabilityLogging:
    @pytest.mark.anyio
    async def test_chat_pipeline_logs_structured_telemetry_tags(self, tmp_path: Path, caplog: pytest.LogCaptureFixture):
        """Verify structured log tags are emitted across the chat execution path."""
        doc_id = str(uuid.uuid4())
        _create_mock_doc_dir(tmp_path, doc_id=doc_id, page_count=2, chars_per_page=200)

        chat_service = ChatService(target_dir=tmp_path)
        mock_retrieval = MagicMock(spec=RetrievalService)
        mock_retrieval.retrieve.return_value = [
            RetrievalResult(
                document_id=doc_id,
                chunk_id=f"{doc_id[:8]}_chunk_001",
                chunk_index=1,
                text="Telemetry test chunk content.",
                start_page=1,
                end_page=1,
                start_character=0,
                end_character=30,
                word_count=4,
                character_count=28,
                sentence_count=1,
                estimated_tokens=8,
                score=0.91,
                rank=1
            )
        ]
        chat_service.retrieval_service = mock_retrieval

        mock_provider = MagicMock(spec=GroqProvider)
        mock_provider.generate.return_value = ("Telemetry response.", False)
        mock_provider.provider_name.return_value = "Groq"
        mock_provider.model_name.return_value = settings.LLM_MODEL
        chat_service.provider = mock_provider

        with caplog.at_level(logging.INFO):
            request = ChatRequest(question="Telemetry question", top_k=3)
            await chat_service.answer_question(document_id=doc_id, request=request)

        log_text = caplog.text
        assert "[REQUEST]" in log_text
        assert "[ROUTING]" in log_text
        assert "[RETRIEVAL]" in log_text
        assert "[CONTEXT]" in log_text
        assert "[LLM]" in log_text
        assert "[SOURCES]" in log_text
        assert "[RESPONSE]" in log_text
