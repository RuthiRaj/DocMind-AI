"""
Abstract Retrieval Provider Interface.

Defines the base contract for all vector similarity search engines
ensuring dependency inversion.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class RetrievalProvider(ABC):
    """
    Abstract interface defining search and load validation contract.
    """

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int, index_path: Path) -> list[tuple[float, int]]:
        """
        Queries a persisted vector index to retrieve the top_k nearest neighbors.

        Args:
            query_vector (np.ndarray): 1D float32 normalized vector.
            top_k (int): Maximum matches to return.
            index_path (Path): Path to saved index binary.

        Returns:
            list[tuple[float, int]]: List of (similarity_score, vector_id_index) matches.
        """
        pass

    @abstractmethod
    def validate(self, index_path: Path) -> bool:
        """
        Validates index structure and integrity on disk.

        Args:
            index_path (Path): Path to saved index binary.

        Returns:
            bool: True if the index is valid, False otherwise.
        """
        pass
