import json
import logging
import threading
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, Any, List
import numpy as np
import fitz # PyMuPDF
from fastapi import HTTPException, status
from app.core.config import settings

logger = logging.getLogger(__name__)

class PipelineLockManager:
    """
    Thread-safe in-memory manager to prevent concurrent execution of the same pipeline stage
    for a specific document_id.
    """
    _running_stages = set() # Set of Tuple[str, str] representing (document_id, stage)
    _lock = threading.Lock()

    @classmethod
    def acquire_stage(cls, document_id: str, stage_name: str) -> bool:
        """
        Attempts to acquire the lock for a document ID and stage.
        Returns True if successful, False if the stage is already running.
        """
        key = (document_id, stage_name)
        with cls._lock:
            if key in cls._running_stages:
                return False
            cls._running_stages.add(key)
            return True

    @classmethod
    def release_stage(cls, document_id: str, stage_name: str) -> None:
        """
        Releases the running lock for a document ID and stage.
        """
        key = (document_id, stage_name)
        with cls._lock:
            cls._running_stages.discard(key)


def validate_process_artifacts(doc_dir: Path) -> Tuple[bool, str]:
    """
    Validates all artifacts produced during the PDF processing stage.
    """
    pdf_path = doc_dir / "original.pdf"
    text_path = doc_dir / "extracted_text.txt"
    pages_path = doc_dir / "pages.json"
    metadata_path = doc_dir / "metadata.json"

    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        return False, "Original PDF file is missing or empty."
    if not text_path.exists() or text_path.stat().st_size == 0:
        return False, "Extracted text file is missing or empty."
    
    # Validate pages.json
    if not pages_path.exists() or pages_path.stat().st_size == 0:
        return False, "Pages index file is missing or empty."
    try:
        with open(pages_path, "r", encoding="utf-8") as f:
            pages_data = json.load(f)
        if not isinstance(pages_data, list):
            return False, "Pages index is not a valid list."
    except Exception as err:
        return False, f"Pages index file is corrupted: {str(err)}"

    # Validate metadata.json
    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        return False, "Metadata file is missing or empty."
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if "total_pages" not in meta:
            return False, "Metadata is missing 'total_pages' field."
    except Exception as err:
        return False, f"Metadata file is corrupted: {str(err)}"

    return True, "Process artifacts are valid."


def validate_chunk_artifacts(doc_dir: Path) -> Tuple[bool, str]:
    """
    Validates all artifacts produced during the text chunking stage.
    """
    chunks_path = doc_dir / "chunks.json"
    stats_path = doc_dir / "chunk_statistics.json"

    if not chunks_path.exists() or chunks_path.stat().st_size == 0:
        return False, "Chunks file is missing or empty."
    
    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        if not isinstance(chunks_data, list):
            return False, "Chunks file is not a valid list."
        if not chunks_data:
            return False, "Chunks file contains no entries."
        for idx, chunk in enumerate(chunks_data):
            if not isinstance(chunk, dict) or "chunk_id" not in chunk or "text" not in chunk:
                return False, f"Chunk at index {idx} is invalid or missing fields."
    except Exception as err:
        return False, f"Chunks file is corrupted: {str(err)}"

    if not stats_path.exists() or stats_path.stat().st_size == 0:
        return False, "Chunk statistics file is missing or empty."
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            json.load(f)
    except Exception as err:
        return False, f"Chunk statistics file is corrupted: {str(err)}"

    return True, "Chunk artifacts are valid."


def validate_embedding_artifacts(doc_dir: Path) -> Tuple[bool, str]:
    """
    Validates all artifacts produced during the embedding stage.
    """
    embeddings_path = doc_dir / "embeddings.npy"
    meta_path = doc_dir / "embedding_metadata.json"
    stats_path = doc_dir / "embedding_statistics.json"
    chunks_path = doc_dir / "chunks.json"

    if not embeddings_path.exists():
        return False, "Embeddings npy array file is missing."

    # Validate embeddings loading and shapes
    try:
        embeddings = np.load(embeddings_path)
        if embeddings.size == 0 or embeddings.ndim != 2:
            return False, f"Invalid embedding matrix shape: {embeddings.shape}."
        if embeddings.shape[1] != settings.EMBEDDING_DIMENSION:
            return False, f"Embedding dimension mismatch: got {embeddings.shape[1]}, expected {settings.EMBEDDING_DIMENSION}."
    except Exception as err:
        return False, f"Embeddings array is corrupted: {str(err)}"

    # Validate embedding metadata
    if not meta_path.exists() or meta_path.stat().st_size == 0:
        return False, "Embedding metadata file is missing or empty."
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            emb_meta = json.load(f)
    except Exception as err:
        return False, f"Embedding metadata is corrupted: {str(err)}"

    # Validate embedding statistics
    if not stats_path.exists() or stats_path.stat().st_size == 0:
        return False, "Embedding statistics file is missing or empty."
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            json.load(f)
    except Exception as err:
        return False, f"Embedding statistics is corrupted: {str(err)}"

    # Consistency Check: compare against chunks.json
    if chunks_path.exists():
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            if len(chunks_data) != embeddings.shape[0]:
                return False, f"Embeddings count ({embeddings.shape[0]}) does not match chunks count ({len(chunks_data)})."
        except Exception:
            pass

    return True, "Embedding artifacts are valid."


def validate_indexing_artifacts(doc_dir: Path) -> Tuple[bool, str]:
    """
    Validates all artifacts produced during the indexing stage.
    """
    index_path = doc_dir / "index.faiss"
    meta_path = doc_dir / "index_metadata.json"
    stats_path = doc_dir / "index_statistics.json"
    chunks_path = doc_dir / "chunks.json"

    if not index_path.exists() or index_path.stat().st_size == 0:
        return False, "FAISS index file is missing or empty."

    # Validate metadata
    if not meta_path.exists() or meta_path.stat().st_size == 0:
        return False, "Index metadata file is missing or empty."
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            idx_meta = json.load(f)
        indexed_vectors = idx_meta.get("indexed_vectors", 0)
    except Exception as err:
        return False, f"Index metadata is corrupted: {str(err)}"

    # Validate statistics
    if not stats_path.exists() or stats_path.stat().st_size == 0:
        return False, "Index statistics file is missing or empty."
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            json.load(f)
    except Exception as err:
        return False, f"Index statistics is corrupted: {str(err)}"

    # Consistency Check: compare against chunks.json
    if chunks_path.exists():
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            if len(chunks_data) != indexed_vectors:
                return False, f"Indexed vectors count ({indexed_vectors}) does not match chunks count ({len(chunks_data)})."
        except Exception:
            pass

    return True, "Indexing artifacts are valid."
