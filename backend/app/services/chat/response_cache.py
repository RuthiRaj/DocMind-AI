"""
In-Memory Response Cache for Document-Grounded RAG Queries.

Features LRU eviction, deterministic question normalization, thread safety,
and cache performance telemetry metrics.
"""

import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def normalize_question(question: str) -> str:
    """
    Normalizes question string for robust semantic cache key generation.
    - Converts to lowercase
    - Strips punctuation and symbols
    - Collapses multiple whitespace characters
    """
    if not question:
        return ""
    q = question.lower().strip()
    # Replace punctuation characters with space
    q = re.sub(r"[^\w\s]", " ", q)
    # Collapse multiple whitespace into a single space
    q = re.sub(r"\s+", " ", q).strip()
    return q


@dataclass
class CacheEntry:
    answer: str
    sources: List[Dict[str, Any]]
    context_mode: str
    timestamp: float


class ResponseCache:
    """
    Thread-safe LRU Response Cache for Chat Completions.
    """

    def __init__(self, max_entries: int = 200):
        self._max_entries = max_entries
        self._cache: OrderedDict[Tuple[str, str], CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, document_id: str, question: str) -> Optional[CacheEntry]:
        """
        Lookup cached response for a document_id and question.
        Returns CacheEntry if found, None otherwise.
        """
        normalized_q = normalize_question(question)
        if not normalized_q:
            return None

        key = (document_id, normalized_q)
        with self._lock:
            if key in self._cache:
                self._hits += 1
                # Move to end for LRU recency
                self._cache.move_to_end(key)
                entry = self._cache[key]
                hit_rate = (self._hits / (self._hits + self._misses)) * 100.0
                logger.info(
                    "[CACHE_HIT] doc=%s query='%s' (hits=%d, misses=%d, hit_rate=%.1f%%)",
                    document_id[:8] if document_id else "N/A",
                    normalized_q[:50],
                    self._hits,
                    self._misses,
                    hit_rate,
                )
                return entry

            self._misses += 1
            logger.info(
                "[CACHE_MISS] doc=%s query='%s' (hits=%d, misses=%d)",
                document_id[:8] if document_id else "N/A",
                normalized_q[:50],
                self._hits,
                self._misses,
            )
            return None

    def put(
        self,
        document_id: str,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]],
        context_mode: str = "RAG",
    ) -> None:
        """
        Stores an answer in the cache with LRU eviction.
        """
        normalized_q = normalize_question(question)
        if not normalized_q or not answer:
            return

        key = (document_id, normalized_q)
        entry = CacheEntry(
            answer=answer,
            sources=sources,
            context_mode=context_mode,
            timestamp=time.time(),
        )

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = entry

            # Evict oldest entry if size limit exceeded
            if len(self._cache) > self._max_entries:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug("[CACHE_EVICT] Evicted oldest key: doc=%s query='%s'", evicted_key[0][:8], evicted_key[1][:30])

    def clear(self) -> None:
        """Clears all cached entries and resets statistics."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Returns cache telemetry metrics."""
        with self._lock:
            total = self._hits + self._misses
            rate = (self._hits / total * 100.0) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_entries": self._max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "total_queries": total,
                "hit_rate_pct": round(rate, 2),
            }


# Global singleton cache instance
response_cache = ResponseCache(max_entries=200)
