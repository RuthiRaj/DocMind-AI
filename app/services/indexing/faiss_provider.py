"""
FAISS Vector Index Provider.

Implements the IndexProvider interface using the local FAISS library.
Creates Flat Inner Product (IP) index, normalizes vectors, and verifies search integrity.
"""

import logging
import threading
from pathlib import Path
import faiss
import numpy as np

from app.core.config import settings
from app.services.indexing.provider import IndexProvider

logger = logging.getLogger(__name__)


class FaissIndexProvider(IndexProvider):
    """
    FAISS index building and serialization provider.
    """

    _lock = threading.Lock()

    def __init__(self):
        """
        Initialize the provider using core settings configuration.
        """
        self._index_type = settings.INDEX_TYPE
        self._distance_metric = settings.VECTOR_DISTANCE
        self._dimension = settings.EMBEDDING_DIMENSION

    def create_index(self, vectors: np.ndarray) -> faiss.IndexFlatIP:
        """
        Creates and populates a FAISS IndexFlatIP.
        Normalizes vectors if cosine similarity is configured.

        Args:
            vectors (np.ndarray): 2D float32 NumPy array.

        Returns:
            faiss.IndexFlatIP: populated index instance.
        """
        if vectors.size == 0:
            raise ValueError("Cannot create index with empty vector array.")

        dim = vectors.shape[1]
        logger.info("Initializing FAISS IndexFlatIP with dimension: %d", dim)

        # Work within thread lock for thread-safety during creation
        with self._lock:
            # IndexFlatIP calculates inner product
            index = faiss.IndexFlatIP(dim)
            
            # For cosine similarity, inputs must be L2 normalized
            if self._distance_metric == "cosine":
                logger.info("L2 normalizing vectors for cosine similarity compatibility...")
                # Normalize copy of vectors to avoid mutating source array directly
                norm_vectors = vectors.copy()
                faiss.normalize_L2(norm_vectors)
                index.add(norm_vectors)
            else:
                index.add(vectors)

        logger.info("FAISS Index created and loaded with %d vectors.", index.ntotal)
        return index

    def save(self, index: faiss.IndexFlatIP, path: Path) -> None:
        """
        Writes index instance to disk using faiss.write_index.

        Args:
            index (faiss.IndexFlatIP): index instance.
            path (Path): Destination file path.
        """
        logger.info("Saving FAISS index to disk: '%s'", path)
        try:
            # faiss.write_index takes index instance and string path
            faiss.write_index(index, str(path))
        except Exception as exc:
            logger.error("Failed to write FAISS index to disk path '%s': %s", path, str(exc))
            raise RuntimeError(f"FAISS save operation failed: {str(exc)}")

    def load(self, path: Path) -> faiss.IndexFlatIP:
        """
        Loads index instance from disk using faiss.read_index.

        Args:
            path (Path): File path.

        Returns:
            faiss.IndexFlatIP: Loaded index instance.
        """
        logger.info("Loading FAISS index from disk: '%s'", path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {path}")

        try:
            index = faiss.read_index(str(path))
            logger.info("FAISS index loaded successfully. Vector count: %d", index.ntotal)
            return index
        except Exception as exc:
            logger.error("Failed to load FAISS index from disk path '%s': %s", path, str(exc))
            raise RuntimeError(f"FAISS load operation failed: {str(exc)}")

    def validate(self, index: faiss.IndexFlatIP, expected_count: int, expected_dim: int) -> bool:
        """
        Validates index properties and performs a test lookup querying the index.

        Args:
            index (faiss.IndexFlatIP): Loaded index instance.
            expected_count (int): Expected indexed vector count.
            expected_dim (int): Expected vector dimension.

        Returns:
            bool: True if index is valid and search integrity passes.
        """
        if index is None:
            logger.warning("Index validation failed: Index object is None.")
            return False

        # 1. Count checks
        current_count = index.ntotal
        if current_count != expected_count:
            logger.warning("Index validation failed: count mismatch (expected %d, got %d).", expected_count, current_count)
            return False

        # 2. Dimension checks
        current_dim = index.d
        if current_dim != expected_dim:
            logger.warning("Index validation failed: dimension mismatch (expected %d, got %d).", expected_dim, current_dim)
            return False

        # 3. Test lookup query check
        try:
            # Query with a random unit vector matching expected dimensions
            query_vector = np.random.randn(1, expected_dim).astype(np.float32)
            if self._distance_metric == "cosine":
                faiss.normalize_L2(query_vector)

            # Search top 1 nearest neighbor
            k = 1
            distances, indices = index.search(query_vector, k)

            if indices.shape != (1, k) or distances.shape != (1, k):
                logger.warning("Index validation failed: search query returned invalid shapes.")
                return False

            logger.info("Index validation integrity check passed successfully.")
            return True

        except Exception as err:
            logger.exception("Index validation failed due to query lookup exception: %s", str(err))
            return False

    def vector_count(self, index: faiss.IndexFlatIP) -> int:
        """
        Returns number of indexed vectors.
        """
        return index.ntotal if index else 0

    def dimension(self, index: faiss.IndexFlatIP) -> int:
        """
        Returns dimension of index.
        """
        return index.d if index else self._dimension

    def index_type(self) -> str:
        """
        Returns index configuration type.
        """
        return self._index_type
