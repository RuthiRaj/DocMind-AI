"""
Text Chunking Engine Service Layer.

Encapsulates business logic for preprocessing extracted text, performing
production-grade smart chunking with stable IDs, page mapping, character offsets,
token estimations, rigorous validation, and statistics persistence.
"""

import json
import logging
import os
import math
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
from fastapi import HTTPException, status

from app.core.config import settings
from app.schemas.chunking import ChunkItem, ChunkingResponse
from app.services.pdf.pipeline_validator import PipelineLockManager, validate_process_artifacts, validate_chunk_artifacts

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
        Determines start_page and end_page for a given character range based on pages.json
        using exact character interval overlap.

        Args:
            start_char (int): Starting character offset of the chunk.
            end_char (int): Ending character offset of the chunk.
            pages_meta (List[Dict]): List of page metadata entries with start_character and end_character.

        Returns:
            Tuple[int, int]: (start_page, end_page) representing the actual pages overlapping this chunk.
        """
        if not pages_meta:
            return (1, 1)

        # Handle inverted offsets defensively
        if end_char < start_char:
            start_char, end_char = end_char, start_char

        overlapping_pages: List[int] = []

        for meta in pages_meta:
            page_num = meta.get("page", 1)
            p_start = meta.get("start_character", 0)
            p_end = meta.get("end_character", 0)

            # Positive overlap between intervals
            if start_char < end_char:
                if max(start_char, p_start) < min(end_char, p_end) and p_end > p_start:
                    overlapping_pages.append(page_num)
            else:
                if p_start <= start_char <= p_end:
                    overlapping_pages.append(page_num)

        if overlapping_pages:
            return (min(overlapping_pages), max(overlapping_pages))


        # Defensive fallback: If chunk falls in whitespace/gap between pages, find the closest page
        closest_page = pages_meta[0].get("page", 1)
        min_distance = float("inf")

        for meta in pages_meta:
            page_num = meta.get("page", 1)
            p_start = meta.get("start_character", 0)
            p_end = meta.get("end_character", 0)

            if p_start <= start_char <= p_end:
                return (page_num, page_num)

            dist = min(abs(start_char - p_start), abs(start_char - p_end))
            if dist < min_distance:
                min_distance = dist
                closest_page = page_num

        return (closest_page, closest_page)


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
        Executes sliding-window smart chunking with stable IDs, exact character offsets,
        and page mappings derived directly from page character boundaries recorded at ingestion.

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

        if not full_text or not full_text.strip():
            return chunks

        # 1. Extract semantic units with exact (start_char, end_char) offsets in full_text
        units: List[Tuple[str, int, int]] = []  # (unit_text, start_char, end_char)

        # Split into paragraph matches preserving exact character offsets
        para_pattern = re.compile(r"\S.*?(?=(?:\n\s*\n)|\Z)", re.DOTALL)
        for p_match in para_pattern.finditer(full_text):
            p_raw = p_match.group(0)
            p_start = p_match.start()
            p_end = p_match.end()
            p_text = p_raw.strip()
            if not p_text:
                continue

            # Adjust offsets to stripped content
            lead_ws = len(p_raw) - len(p_raw.lstrip())
            p_start += lead_ws
            p_end = p_start + len(p_text)

            if len(p_text) <= chunk_size:
                units.append((p_text, p_start, p_end))
            else:
                # Split paragraph into sentences preserving exact character offsets
                s_pattern = re.compile(r"[^.!?]+(?:[.!?]+|\Z)", re.DOTALL)
                for s_match in s_pattern.finditer(p_text):
                    s_raw = s_match.group(0)
                    s_text = s_raw.strip()
                    if not s_text:
                        continue
                    s_lead_ws = len(s_raw) - len(s_raw.lstrip())
                    s_start = p_start + s_match.start() + s_lead_ws
                    s_end = s_start + len(s_text)

                    if len(s_text) <= chunk_size:
                        units.append((s_text, s_start, s_end))
                    else:
                        # Split sentence into words preserving exact character offsets
                        w_pattern = re.compile(r"\S+")
                        cur_words = []
                        cur_w_start = None
                        cur_w_end = None
                        for w_match in w_pattern.finditer(s_text):
                            w_text = w_match.group(0)
                            w_start = s_start + w_match.start()
                            w_end = s_start + w_match.end()
                            if cur_words and (w_end - cur_w_start) > chunk_size:
                                units.append((" ".join(cur_words), cur_w_start, cur_w_end))
                                cur_words = [w_text]
                                cur_w_start = w_start
                                cur_w_end = w_end
                            else:
                                if cur_w_start is None:
                                    cur_w_start = w_start
                                cur_words.append(w_text)
                                cur_w_end = w_end
                        if cur_words and cur_w_start is not None and cur_w_end is not None:
                            units.append((" ".join(cur_words), cur_w_start, cur_w_end))

        logger.info(
            "Hierarchical text split completed for document_id '%s': %d semantic units with exact offsets",
            document_id,
            len(units)
        )

        if not units:
            return chunks

        # 2. Assemble chunks using exact offsets
        chunk_index_counter = 1
        unit_idx = 0
        total_units = len(units)

        while unit_idx < total_units:
            chunk_units: List[Tuple[str, int, int]] = []
            cur_len = 0
            cur_start_idx = unit_idx

            while unit_idx < total_units:
                u_text, u_start, u_end = units[unit_idx]
                added_len = len(u_text) + (2 if chunk_units else 0)
                if cur_len + added_len > chunk_size and chunk_units:
                    break
                chunk_units.append(units[unit_idx])
                cur_len += added_len
                unit_idx += 1

            if not chunk_units:
                # Single unit larger than chunk_size, consume it
                chunk_units.append(units[unit_idx])
                unit_idx += 1

            start_char = chunk_units[0][1]
            end_char = chunk_units[-1][2]
            chunk_text = full_text[start_char:end_char].strip()
            if not chunk_text:
                chunk_text = "\n\n".join(u[0] for u in chunk_units)

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
                    page_start=start_page,
                    page_end=end_page,
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

            if unit_idx >= total_units:
                break

            # Handle overlap: backtrack unit_idx to include units overlapping with end of chunk
            if chunk_overlap > 0 and len(chunk_units) > 1:
                overlap_cutoff = end_char - chunk_overlap
                backtrack_to = None
                for i in range(len(chunk_units) - 1, 0, -1):
                    if chunk_units[i][1] >= overlap_cutoff:
                        backtrack_to = cur_start_idx + i
                    else:
                        break
                if backtrack_to is not None and backtrack_to > cur_start_idx and backtrack_to < unit_idx:
                    unit_idx = backtrack_to

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

        if len(chunks) > settings.MAX_CHUNKS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Generated chunks count ({len(chunks)}) exceeds the maximum allowed limit of {settings.MAX_CHUNKS}."
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

    def _write_atomic(self, target_path: Path, content: any) -> None:
        """
        Atomically writes content to a target file.
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
            logger.exception("Atomic file write failed in chunking service: %s", str(exc))
            raise

    def _update_status(self, status_path: Path, new_status: str, extra_fields: dict | None = None) -> dict:
        """
        Helper method to update status.json securely and return the updated dictionary.
        """
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            with open(status_path, "r", encoding="utf-8") as sf:
                status_data = json.load(sf)
        except Exception as err:
            logger.warning("Failed to read status.json. Constructing default state tracker: %s", str(err))
            status_data = {}

        status_data["chunking_status"] = new_status
        status_data["updated_at"] = now_str

        if extra_fields:
            status_data.update(extra_fields)

        self._write_atomic(status_path, status_data)
        logger.info("Pipeline state updated to '%s' in status.json", new_status)
        return status_data

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

        # Prevent concurrent duplicate execution for the same document ID
        if not PipelineLockManager.acquire_stage(safe_doc_id, "chunk"):
            detail_msg = "Text chunking is currently running for this document. Duplicate execution rejected."
            logger.warning(detail_msg)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=detail_msg
            )

        try:
            import asyncio
            await asyncio.sleep(0.05)

            # 1. Dependency Validation (Verify previous stage artifacts)
            process_valid, process_msg = validate_process_artifacts(doc_dir)
            if not process_valid:
                detail_msg = f"Processing stage must be completed and valid before chunking. Reason: {process_msg}"
                logger.warning("Chunking dependency validation failed for document_id '%s': %s", safe_doc_id, detail_msg)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=detail_msg
                )

            # 2. Idempotency Check (Check if already chunked and valid)
            is_completed = False
            if status_path.exists():
                try:
                    with open(status_path, "r", encoding="utf-8") as sf:
                        status_data = json.load(sf)
                    if status_data.get("chunking_status") == "completed":
                        is_completed = True
                except Exception:
                    pass

            chunks_valid, chunk_msg = validate_chunk_artifacts(doc_dir)

            if not force and is_completed and chunks_valid:
                logger.info("Text chunking already completed for document_id '%s'. Reusing valid existing artifacts.", safe_doc_id)
                try:
                    with open(stats_path, "r", encoding="utf-8") as stf:
                        stats = json.load(stf)
                    return ChunkingResponse(
                        success=True,
                        document_id=safe_doc_id,
                        total_chunks=stats.get("total_chunks", 0),
                        average_chunk_size=stats.get("average_chunk_size", 0),
                        average_tokens=stats.get("average_tokens", 0),
                        processing_time_ms=stats.get("processing_time_ms", 0),
                        chunk_version=settings.CHUNK_VERSION,
                        message="Chunking completed successfully (cached result)."
                    )
                except Exception as e:
                    logger.warning("Failed to read chunk_statistics.json for document_id '%s': %s. Re-chunking...", safe_doc_id, str(e))

            request_id = str(uuid.uuid4())
            logger.info("[STAGE: CHUNK] request_id=%s document_id='%s' status=started", request_id, safe_doc_id)
            self._update_status(status_path, "running")
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

            clean_text = raw_text if pages_meta else self.preprocess_text(raw_text)
            logger.info("Text prepared for chunking for document_id '%s' (%d characters, %d pages meta)", safe_doc_id, len(clean_text), len(pages_meta))

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

            # Save chunks.json atomically
            chunks_payload = [item.model_dump() for item in chunk_items]
            try:
                self._write_atomic(chunks_path, chunks_payload)
                logger.info("chunks.json written atomically to '%s'", chunks_path.name)
            except Exception as write_err:
                logger.error("Failed to write chunks.json atomically for document_id '%s': %s", safe_doc_id, str(write_err))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error writing chunks.json: {str(write_err)}"
                )

            end_time = time.perf_counter()
            chunking_time_ms = int(round((end_time - start_time) * 1000))
            processed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            # Save chunk_statistics.json atomically
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
                self._write_atomic(stats_path, stats_payload)
                logger.info("Chunk statistics saved atomically to '%s'", stats_path.name)
            except Exception as stats_err:
                logger.warning("Failed to save chunk_statistics.json atomically for document_id '%s': %s", safe_doc_id, str(stats_err))

            # Update status.json with chunking completed and reset downstream statuses to pending
            self._update_status(
                status_path=status_path,
                new_status="completed",
                extra_fields={
                    "embedding_status": "pending",
                    "indexing_status": "pending",
                    "chunk_version": settings.CHUNK_VERSION
                }
            )

            logger.info(
                "[STAGE: CHUNK] request_id=%s document_id='%s' status=completed elapsed_ms=%d chunks=%d avg_tokens=%d",
                request_id,
                safe_doc_id,
                chunking_time_ms,
                total_chunks,
                avg_tokens
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

        except Exception as err:
            self._update_status(status_path, "failed")
            if isinstance(err, HTTPException):
                raise
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to complete text chunking: {str(err)}"
            )
        finally:
            PipelineLockManager.release_stage(safe_doc_id, "chunk")
