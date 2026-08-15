"""
BM25 Keyword Search Provider for DocMind AI.

Provides BM25 Okapi scoring and Reciprocal Rank Fusion (RRF)
to combine keyword matching with dense vector search results.
"""

import math
import re
from typing import Dict, List, Tuple


class BM25Retriever:
    """
    In-memory BM25 Okapi Retriever for document chunk corpus.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize and normalize text into lowercase terms.
        """
        return re.findall(r"\b\w+\b", text.lower())

    def search(
        self,
        query: str,
        chunks: List[dict],
        top_k: int = 25
    ) -> List[Tuple[dict, float]]:
        """
        Calculates BM25 Okapi score for each chunk against the query.

        Args:
            query (str): Query string.
            chunks (List[dict]): List of chunk dicts (or objects with .text).
            top_k (int): Number of top scored chunks to return.

        Returns:
            List[Tuple[dict, float]]: List of (chunk, bm25_score) sorted descending by score.
        """
        if not query or not chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        N = len(chunks)
        corpus_tokens = []
        doc_lengths = []

        for c in chunks:
            text = c.text if hasattr(c, "text") else c.get("text", "")
            tokens = self._tokenize(text)
            corpus_tokens.append(tokens)
            doc_lengths.append(len(tokens))

        avgdl = sum(doc_lengths) / N if N > 0 else 1.0
        if avgdl == 0:
            avgdl = 1.0

        # Calculate Document Frequency (DF) for query tokens
        df = {}
        for qt in set(query_tokens):
            df[qt] = sum(1 for tokens in corpus_tokens if qt in set(tokens))

        # Calculate Inverse Document Frequency (IDF) for query tokens
        idf = {}
        for qt, doc_freq in df.items():
            idf[qt] = math.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

        # Score each chunk
        scored_chunks = []
        for idx, c in enumerate(chunks):
            tokens = corpus_tokens[idx]
            doc_len = doc_lengths[idx]
            token_counts = {}
            for t in tokens:
                token_counts[t] = token_counts.get(t, 0) + 1

            score = 0.0
            for qt in query_tokens:
                if qt not in token_counts:
                    continue
                tf = token_counts[qt]
                num = tf * (self.k1 + 1.0)
                den = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avgdl))
                score += idf.get(qt, 0.0) * (num / den)

            if score > 0.0:
                scored_chunks.append((c, score))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

    @staticmethod
    def reciprocal_rank_fusion(
        vector_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
        rrf_k: int = 60
    ) -> Dict[str, float]:
        """
        Combines two ranked lists using Reciprocal Rank Fusion (RRF).

        RRF Score = 1 / (k + rank_vec) + 1 / (k + rank_bm25)

        Returns:
            Dict[str, float]: Mapping of chunk_id -> fused RRF score.
        """
        rrf_scores: Dict[str, float] = {}

        for rank, (chunk_id, _) in enumerate(vector_results, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))

        for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))

        return rrf_scores
