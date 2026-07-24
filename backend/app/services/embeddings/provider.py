"""
Abstract Embedding Provider Interface.

Defines the base contract for all vector generation models and providers
ensuring dependency inversion.
"""

from abc import ABC, abstractmethod
import numpy as np


class EmbeddingProvider(ABC):
    """
    Abstract interface defining methods required for embedding generation.
    """

    @abstractmethod
    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Generates vector embeddings for a list of input texts.

        Args:
            texts (list[str]): List of raw chunk texts.

        Returns:
            np.ndarray: 2D float32 NumPy array where each row is an embedding vector.
        """
        pass

    @abstractmethod
    def dimension(self) -> int:
        """
        Returns the dimensional size of generated embedding vectors.

        Returns:
            int: Dimension size.
        """
        pass

    @abstractmethod
    def model_name(self) -> str:
        """
        Returns the exact model identifier or identifier name.

        Returns:
            str: Model identifier string.
        """
        pass
