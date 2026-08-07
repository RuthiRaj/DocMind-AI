"""
SentenceTransformer Embedding Provider.

Implements the EmbeddingProvider interface using local sentence-transformers models.
Features singleton lazy-loading model instances with thread-safe double-checked locking and caching logs.
"""

import logging
import threading
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.services.embeddings.provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class SentenceTransformerProvider(EmbeddingProvider):
    """
    SentenceTransformer embedding generation implementation.
    """

    _model_instance = None
    _lock = threading.Lock()

    def __init__(self):
        """
        Initialize the provider using application core configurations.
        """
        self._model_name = settings.EMBEDDING_MODEL
        self._dimension = settings.EMBEDDING_DIMENSION
        self._normalize = settings.EMBEDDING_NORMALIZE
        self._batch_size = settings.EMBEDDING_BATCH_SIZE

    def _get_model(self) -> SentenceTransformer:
        """
        Thread-safe lazy-loaded singleton retriever for the SentenceTransformer model.
        Uses double-checked locking to prevent duplicate model loading under concurrent requests.
        """
        if SentenceTransformerProvider._model_instance is None:
            with SentenceTransformerProvider._lock:
                if SentenceTransformerProvider._model_instance is None:
                    logger.info("SentenceTransformer model cache miss. Loading model '%s' from disk/hub...", self._model_name)
                    try:
                        SentenceTransformerProvider._model_instance = SentenceTransformer(self._model_name)
                        logger.info("SentenceTransformer model '%s' loaded and cached successfully.", self._model_name)
                    except Exception as exc:
                        logger.error("Failed to load SentenceTransformer model '%s' from disk/hub: %s", self._model_name, str(exc))
                        raise
                else:
                    logger.info("SentenceTransformer model '%s' retrieved from double-checked locked cached instance.", self._model_name)
        else:
            logger.info("SentenceTransformer model '%s' retrieved from cached instance.", self._model_name)
            
        return SentenceTransformerProvider._model_instance

    def initialize_model(self) -> None:
        """
        Explicitly pre-initializes and caches the model instance during application startup.
        """
        self._get_model()

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Generates dense vector embeddings using local inference.

        Args:
            texts (list[str]): List of texts.

        Returns:
            np.ndarray: float32 2D NumPy array.
        """
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        try:
            model = self._get_model()
        except Exception as load_err:
            raise RuntimeError(f"Could not load SentenceTransformer model: {str(load_err)}")

        logger.info("Generating embeddings for %d text chunks in batches of %d...", len(texts), self._batch_size)
        
        try:
            embeddings = model.encode(
                sentences=texts,
                batch_size=self._batch_size,
                show_progress_bar=False,
                normalize_embeddings=self._normalize,
                convert_to_numpy=True
            )
        except Exception as encode_err:
            logger.exception("SentenceTransformer encode execution failed: %s", str(encode_err))
            raise RuntimeError(f"Failed to generate embeddings: {str(encode_err)}")
        
        return np.asarray(embeddings, dtype=np.float32)

    def dimension(self) -> int:
        """
        Returns model output vector dimension.
        """
        return self._dimension

    def model_name(self) -> str:
        """
        Returns model identifier.
        """
        return self._model_name
