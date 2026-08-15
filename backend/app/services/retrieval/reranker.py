"""
Score Fusion Reranker for DocMind AI Retrieval Engine.

Provides fine-grained candidate reranking over retrieved vector/BM25 chunks
to optimize context selection before prompt compilation.
"""

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)


class ScoreFusionReranker:
    """
    Reranker service for scoring candidate chunks against the user query.
    """

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[float, dict]],
        top_k: int = 10
    ) -> List[Tuple[float, dict]]:
        """
        Reranks candidate chunks using exact term overlap and semantic density.

        Args:
            query (str): User query string.
            candidates (List[Tuple[float, dict]]): List of (score, chunk) tuples.
            top_k (int): Target number of top reranked candidates to return.

        Returns:
            List[Tuple[float, dict]]: Reranked candidates list.
        """
        if not query or not candidates:
            return candidates[:top_k]

        q_terms = set(re.findall(r"\b\w+\b", query.lower()))

        scored_candidates = []
        for base_score, chunk in candidates:
            text = chunk.get("text", "") if isinstance(chunk, dict) else getattr(chunk, "text", "")
            c_terms = set(re.findall(r"\b\w+\b", text.lower()))

            # Term overlap ratio
            overlap_ratio = len(q_terms.intersection(c_terms)) / float(len(q_terms)) if q_terms else 0.0

            # Composite rerank score (70% base similarity/RRF + 30% exact term overlap boost)
            rerank_score = (0.7 * base_score) + (0.3 * overlap_ratio)
            scored_candidates.append((rerank_score, chunk))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return scored_candidates[:top_k]
