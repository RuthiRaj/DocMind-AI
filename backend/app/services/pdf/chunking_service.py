"""
Text Chunking Engine Service Layer.

Encapsulates business logic for preprocessing extracted text, performing
production-grade smart chunking with stable IDs, page mapping, character offsets,
token estimations, rigorous validation, and statistics persistence.
"""

import json
import logging
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
from fastapi import HTTPException, status

from app.core.config import settings
from app.schemas.chunking import ChunkItem, ChunkingResponse

# Initialize logger for chunking operations
logger = logging.getLogger(__name__)


class ChunkingService:
    """
    Service responsible for converting extracted document text into validated,
    semantic chunks with source location metadata and analytics persistence.
    """

    def __init__(self, target_dir: Path | None = None):
        """
        Initialize ChunkingService with target upload storage directory.
        """
        if target_dir is None:
            self.target_dir = Path(__file__).resolve().parent.parent.parent.parent / settings.UPLOAD_DIRECTORY
        else:
            self.target_dir = target_dir

    def preprocess_text(self, raw_text: str) -> str:
        """
        Cleans and normalizes extracted text before chunking.

        Args:
            raw_text (str): Raw extracted document text.

        Returns:
            str: Preprocessed, normalized text.
        """
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        text = re.sub(r" ?\n ?", "\n", text)
        text = re.sub(r"\n\n+", "\n\n", text)
        return text.strip()

    def _determine_page_range(
        self,
        start_char: int,
        end_char: int,
        pages_meta: List[Dict[str, int]]
    ) -> Tuple[int, int]:
        """
        Determines start_page and end_page for a given character range based on pages.json.

        Args:
            start_char (int): Starting character offset.
            end_char (int): Ending character offset.
            pages_meta (List[Dict]): List of page metadata entries with start_character and end_character.

        Returns:
            Tuple[int, int]: (start_page, end_page)
        """
        if not pages_meta:
            return (1, 1)

        start_page = pages_meta[0]["page"]
        end_page = pages_meta[-1]["page"]

        for meta in pages_meta:
            page_num = meta["page"]
            p_start = meta["start_character"]
            p_end = meta["end_character"]

            if p_start <= start_char <= p_end:
                start_page = page_num
            if p_start <= end_char <= p_end:
                end_page = page_num
                break
            if start_char >= p_start:
                end_page = page_num

        return (start_page, max(start_page, end_page))

    def _count_sentences(self, text: str) -> int:
        """
        Counts sentences in chunk text using regex boundary matching.

        Args:
            text (str): Chunk text.

        Returns:
            int: Number of sentences (at least 1 if non-empty).
        """
        if not text.strip():
            return 0
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        return max(1, len(sentences))

    def generate_chunks(
        self,
        document_id: str,
        full_text: str,
        pages_meta: List[Dict[str, int]],
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP
    ) -> List[ChunkItem]:
        """
        Executes sliding-window smart chunking with stable IDs, character offsets,
        and page mappings.

        Args:
            document_id (str): Unique document UUID.
            full_text (str): Cleaned full document text.
            pages_meta (List[Dict]): Page character offset mappings from pages.json.
            chunk_size (int): Max target character length.
            chunk_overlap (int): Desired character overlap.

        Returns:
            List[ChunkItem]: Generated list of enriched ChunkItem objects.
        """
        doc_prefix = document_id.split("-")[0] if "-" in document_id else document_id[:8]
        chunks: List[ChunkItem] = []
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Hierarchical split into units (paragraphs / sentences)
        paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
        units: List[str] = []

        for para in paragraphs:
            if len(para) <= chunk_size:
                units.append(para)
            else:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if len(sentence) <= chunk_size:
                        units.append(sentence)
                    else:
                        words = sentence.split(" ")
                        current_word_unit = []
                        current_len = 0
                        for word in words:
                            if current_len + len(word) + 1 <= chunk_size:
                                current_word_unit.append(word)
                                current_len += len(word) + 1
                            else:
                                if current_word_unit:
                                    units.append(" ".join(current_word_unit))
                                current_word_unit = [word]
                                current_len = len(word)
                        if current_word_unit:
                            units.append(" ".join(current_word_unit))

        logger.info(
            "Hierarchical text split completed for document_id '%s': %d paragraphs, %d semantic units",
            document_id,
            len(paragraphs),
            len(units)
        )

        current_units: List[str] = []
        current_char_count = 0
        chunk_index_counter = 1
        search_start_offset = 0

        for unit in units:
            unit_len = len(unit)

            if current_char_count + (2 if current_units else 0) + unit_len > chunk_size and current_units:
                chunk_text = "\n\n".join(current_units).strip()
                if chunk_text:
                    # Find character bounds within full_text
                    start_char = full_text.find(chunk_text[:30], search_start_offset)
                    if start_char == -1:
                        start_char = search_start_offset
                    end_char = start_char + len(chunk_text)

                    start_page, end_page = self._determine_page_range(start_char, end_char, pages_meta)
                    char_count = len(chunk_text)
                    word_count = len(chunk_text.split())
                    sentence_count = self._count_sentences(chunk_text)
                    estimated_tokens = math.ceil(char_count / settings.TOKEN_ESTIMATION_RATIO)
                    stable_id = f"{doc_prefix}_chunk_{chunk_index_counter:06d}"

                    chunks.append(
                        ChunkItem(
                            chunk_id=stable_id,
                            chunk_index=chunk_index_counter,
                            document_id=document_id,
                            start_character=start_char,
                            end_character=end_char,
                            start_page=start_page,
                            end_page=end_page,
                            character_count=char_count,
                            word_count=word_count,
                            sentence_count=sentence_count,
                            estimated_tokens=estimated_tokens,
                            embedding_status="pending",
                            embedding_model=None,
                            vector_dimension=None,
                            created_at=created_at,
                            text=chunk_text
                        )
                    )
                    chunk_index_counter += 1
                    search_start_offset = max(0, end_char - chunk_overlap)

                # Form overlap buffer for next chunk
                overlap_text = ""
                if chunk_overlap > 0 and len(chunk_text) > chunk_overlap:
                    raw_overlap = chunk_text[-chunk_overlap:]
                    space_idx = raw_overlap.find(" ")
                    if space_idx != -1 and space_idx < len(raw_overlap) - 10:
                        overlap_text = raw_overlap[space_idx + 1:].strip()
                    else:
                        overlap_text = raw_overlap.strip()

                if overlap_text:
                    current_units = [overlap_text, unit]
                    current_char_count = len(overlap_text) + 2 + unit_len
                else:
                    current_units = [unit]
                    current_char_count = unit_len
            else:
                current_units.append(unit)
                current_char_count += (2 if len(current_units) > 1 else 0) + unit_len

        # Flush final chunk
        if current_units:
            final_text = "\n\n".join(current_units).strip()
            if final_text:
                start_char = full_text.find(final_text[:30], search_start_offset)
                if start_char == -1:
                    start_char = search_start_offset
                end_char = start_char + len(final_text)

                start_page, end_page = self._determine_page_range(start_char, end_char, pages_meta)
                char_count = len(final_text)
                word_count = len(final_text.split())
                sentence_count = self._count_sentences(final_text)
                estimated_tokens = math.ceil(char_count / settings.TOKEN_ESTIMATION_RATIO)
                stable_id = f"{doc_prefix}_chunk_{chunk_index_counter:06d}"

                chunks.append(
                    ChunkItem(
                        chunk_id=stable_id,
                        chunk_index=chunk_index_counter,
                        document_id=document_id,
                        start_character=start_char,
                        end_character=end_char,
                        start_page=start_page,
                        end_page=end_page,
                        character_count=char_count,
                        word_count=word_count,
                        sentence_count=sentence_count,
                        estimated_tokens=estimated_tokens,
                        embedding_status="pending",
                        embedding_model=None,
                        vector_dimension=None,
                        created_at=created_at,
                        text=final_text
                    )
                )

        return chunks

    def validate_chunks(self, chunks: List[ChunkItem], chunk_size_limit: int = settings.CHUNK_SIZE) -> None:
        """
        Validates generated chunks against production quality criteria.

        Raises:
            HTTPException 400: If any validation rule fails.
        """
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Chunk validation failed: No chunks generated."
            )

        seen_ids = set()
        seen_texts = set()
        max_allowed_len = int(chunk_size_limit * 1.1)  # Allow max 10% overflow for sentence bounds

        for index, chunk in enumerate(chunks, start=1):
            # Rule 1: No empty or whitespace-only chunks
            if not chunk.text or not chunk.text.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Chunk validation failed: Chunk index {chunk.chunk_index} is empty."
                )

            # Rule 2: Unique and sequential IDs
            if chunk.chunk_id in seen_ids:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Chunk validation failed: Duplicate chunk_id '{chunk.chunk_id}' detected."
                )
            seen_ids.add(chunk.chunk_id)

            if chunk.chunk_index != index:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Chunk validation failed: Non-sequential chunk index {chunk.chunk_index} (expected {index})."
                )

            # Rule 3: No exact duplicate chunk texts
            if chunk.text in seen_texts:
                logger.warning("Duplicate chunk text detected at index %d", chunk.chunk_index)
            seen_texts.add(chunk.text)

            # Rule 4: Chunk size within limit (+10% margin)
            if chunk.character_count > max_allowed_len:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Chunk validation failed: Chunk {chunk.chunk_id} length ({chunk.character_count}) exceeds maximum allowed limit ({max_allowed_len})."
                )

        logger.info("Chunk validation completed successfully: %d chunks verified", len(chunks))

    async def chunk_document(self, document_id: str, force: bool = False) -> ChunkingResponse:
        """
        Orchestrates loading text & pages.json, executing chunking, validating chunks,
        persisting chunks.json & chunk_statistics.json, and updating status.json.

        Args:
            document_id (str): Unique UUID document identifier.
            force (bool): Force re-chunking even if chunking status is marked completed.

        Returns:
            ChunkingResponse: Summary metadata object detailing chunking execution.
        """
        safe_doc_id = Path(document_id).name
        doc_dir = self.target_dir / safe_doc_id
        text_path = doc_dir / "extracted_text.txt"
        pages_path = doc_dir / "pages.json"
        status_path = doc_dir / "status.json"
        chunks_path = doc_dir / "chunks.json"
        stats_path = doc_dir / "chunk_statistics.json"

        # Validate document directory existence
        if not doc_dir.exists():
            detail_msg = f"File not found for document_id: {document_id}"
            logger.warning("Chunking failed: %s", detail_msg)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=detail_msg
            )

        # Validate extracted_text.txt existence
        if not text_path.exists():
            detail_msg = f"Missing extracted_text.txt for document_id: {document_id}. Run process endpoint first."
            logger.warning("Chunking failed: %s", detail_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=detail_msg
            )

        # Check idempotency / already chunked status
        if status_path.exists() and not force:
            try:
                with open(status_path, "r", encoding="utf-8") as sf:
                    status_data = json.load(sf)
                if status_data.get("chunking_status") == "completed" and chunks_path.exists():
                    detail_msg = "Document has already been chunked. Use force=true to re-chunk."
                    logger.warning("Chunking rejected for document_id '%s': %s", safe_doc_id, detail_msg)
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=detail_msg
                    )
            except HTTPException:
                raise
            except Exception as err:
                logger.warning("Could not read status.json for document_id '%s': %s", safe_doc_id, str(err))

        logger.info("Chunking started for document_id: '%s'", safe_doc_id)
        start_time = time.perf_counter()

        # Read extracted_text.txt
        try:
            with open(text_path, "r", encoding="utf-8") as tf:
                raw_text = tf.read()
        except Exception as read_err:
            logger.error("Failed to read extracted_text.txt for document_id '%s': %s", safe_doc_id, str(read_err))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error reading extracted_text.txt: {str(read_err)}"
            )

        if not raw_text or not raw_text.strip():
            detail_msg = "extracted_text.txt is empty. Cannot generate chunks."
            logger.warning("Chunking failed for document_id '%s': %s", safe_doc_id, detail_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=detail_msg
            )

        # Read pages.json if available
        pages_meta = []
        if pages_path.exists():
            try:
                with open(pages_path, "r", encoding="utf-8") as pf:
                    pages_meta = json.load(pf)
            except Exception as pe:
                logger.warning("Could not read pages.json for document_id '%s': %s", safe_doc_id, str(pe))

        clean_text = self.preprocess_text(raw_text)
        logger.info("Text preprocessing completed for document_id '%s' (%d clean characters)", safe_doc_id, len(clean_text))

        # Generate chunks
        chunk_items = self.generate_chunks(
            document_id=safe_doc_id,
            full_text=clean_text,
            pages_meta=pages_meta,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

        # Validate generated chunks
        self.validate_chunks(chunk_items, settings.CHUNK_SIZE)

        total_chunks = len(chunk_items)
        avg_chunk_size = int(round(sum(c.character_count for c in chunk_items) / total_chunks)) if total_chunks > 0 else 0
        avg_words = int(round(sum(c.word_count for c in chunk_items) / total_chunks)) if total_chunks > 0 else 0
        avg_tokens = int(round(sum(c.estimated_tokens for c in chunk_items) / total_chunks)) if total_chunks > 0 else 0
        largest_chunk = max(c.character_count for c in chunk_items) if total_chunks > 0 else 0
        smallest_chunk = min(c.character_count for c in chunk_items) if total_chunks > 0 else 0

        # Save chunks.json
        chunks_payload = [item.model_dump() for item in chunk_items]
        try:
            with open(chunks_path, "w", encoding="utf-8") as cf:
                json.dump(chunks_payload, cf, indent=4)
            logger.info("chunks.json written to '%s'", chunks_path.name)
        except Exception as write_err:
            logger.error("Failed to write chunks.json for document_id '%s': %s", safe_doc_id, str(write_err))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error writing chunks.json: {str(write_err)}"
            )

        end_time = time.perf_counter()
        chunking_time_ms = int(round((end_time - start_time) * 1000))
        processed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Save chunk_statistics.json
        stats_payload = {
            "document_id": safe_doc_id,
            "total_chunks": total_chunks,
            "average_chunk_size": avg_chunk_size,
            "average_words": avg_words,
            "average_tokens": avg_tokens,
            "largest_chunk": largest_chunk,
            "smallest_chunk": smallest_chunk,
            "processing_time_ms": chunking_time_ms
        }
        try:
            with open(stats_path, "w", encoding="utf-8") as stf:
                json.dump(stats_payload, stf, indent=4)
            logger.info("Chunk statistics saved to '%s'", stats_path.name)
        except Exception as stats_err:
            logger.warning("Failed to save chunk_statistics.json for document_id '%s': %s", safe_doc_id, str(stats_err))

        # Update status.json (chunking_status = "completed", embedding_status = "pending", chunk_version = "1.0")
        if status_path.exists():
            try:
                with open(status_path, "r", encoding="utf-8") as sf:
                    status_data = json.load(sf)
                status_data["chunking_status"] = "completed"
                status_data["embedding_status"] = "pending"
                status_data["chunk_version"] = settings.CHUNK_VERSION
                status_data["updated_at"] = processed_at
                with open(status_path, "w", encoding="utf-8") as sf:
                    json.dump(status_data, sf, indent=4)
                logger.info("status.json updated for document_id '%s'", safe_doc_id)
            except Exception as status_err:
                logger.warning("Failed to update status.json for document_id '%s': %s", safe_doc_id, str(status_err))

        logger.info(
            "Chunking completed in %d ms for document_id: '%s' (%d total chunks generated)",
            chunking_time_ms,
            safe_doc_id,
            total_chunks
        )

        return ChunkingResponse(
            success=True,
            document_id=safe_doc_id,
            total_chunks=total_chunks,
            average_chunk_size=avg_chunk_size,
            average_tokens=avg_tokens,
            processing_time_ms=chunking_time_ms,
            chunk_version=settings.CHUNK_VERSION,
            message="Chunking completed successfully."
        )
