"""
FAISS Vector Similarity Search Provider.

Implements the RetrievalProvider interface using local FAISS CPU execution.
Features thread-safe index loading, similarity query matching, and validation checkups.
"""

import logging
import threading
from pathlib import Path
import faiss
import numpy as np

from app.services.retrieval.provider import RetrievalProvider

logger = logging.getLogger(__name__)


class FaissRetrievalProvider(RetrievalProvider):
    """
    FAISS-powered similarity retrieval provider.
    """

    _lock = threading.Lock()

    def search(self, query_vector: np.ndarray, top_k: int, index_path: Path) -> list[tuple[float, int]]:
        """
        Performs vector similarity search on a local FAISS index.

        Args:
            query_vector (np.ndarray): 1D float32 query embedding vector.
            top_k (int): Number of top results to retrieve.
            index_path (Path): Path to FAISS index file.

        Returns:
            list[tuple[float, int]]: Ordered list of (similarity_score, vector_id) tuples.
        """
        logger.info("Executing FAISS search on index: '%s'", index_path.name)
        
        # Ensure query_vector has shape (1, dimension)
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        # Normalize query vector for Cosine similarity compatibility
        query_vector_norm = query_vector.copy()
        faiss.normalize_L2(query_vector_norm)

        # Load index thread-safely
        with self._lock:
            try:
                index = faiss.read_index(str(index_path))
            except Exception as exc:
                logger.error("Failed to load FAISS index from disk path '%s': %s", index_path, str(exc))
                raise RuntimeError(f"FAISS index load failure: {str(exc)}")

        # Execute search
        try:
            distances, indices = index.search(query_vector_norm, top_k)
        except Exception as exc:
            logger.error("FAISS index search query execution failed: %s", str(exc))
            raise RuntimeError(f"FAISS search execution failure: {str(exc)}")

        # Map results to (distance, index) tuple list
        results = []
        if distances.shape[0] > 0:
            for dist, idx in zip(distances[0], indices[0]):
                # FAISS returns -1 index for padding if fewer vectors exist than top_k
                if idx != -1:
                    results.append((float(dist), int(idx)))

        logger.info("FAISS search query returned %d matches.", len(results))
        return results

    def validate(self, index_path: Path) -> bool:
        """
        Validates index structure and counts.

        Args:
            index_path (Path): Path to saved index binary.

        Returns:
            bool: True if index file loads and passes test queries, False otherwise.
        """
        if not index_path.exists():
            return False

        with self._lock:
            try:
                index = faiss.read_index(str(index_path))
                if index is None or index.ntotal == 0:
                    return False
                
                # Verify basic query test
                k = 1
                q = np.random.randn(1, index.d).astype(np.float32)
                faiss.normalize_L2(q)
                index.search(q, k)
                return True
            except Exception as exc:
                logger.warning("FAISS retrieval validation failed for path '%s': %s", index_path, str(exc))
                return False
