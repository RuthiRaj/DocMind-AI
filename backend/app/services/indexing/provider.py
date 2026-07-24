"""
Abstract Vector Index Provider Interface.

Defines the base contract for all vector search index implementations
ensuring dependency inversion.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class IndexProvider(ABC):
    """
    Abstract interface defining methods required for index building and management.
    """

    @abstractmethod
    def create_index(self, vectors: np.ndarray) -> any:
        """
        Creates and populates a search index from a NumPy array of vectors.

        Args:
            vectors (np.ndarray): 2D float32 array of vectors.

        Returns:
            any: Instantiated index instance.
        """
        pass

    @abstractmethod
    def save(self, index: any, path: Path) -> None:
        """
        Saves the index instance to disk.

        Args:
            index (any): The active index instance.
            path (Path): Target file path.
        """
        pass

    @abstractmethod
    def load(self, path: Path) -> any:
        """
        Loads the index instance from disk.

        Args:
            path (Path): File path.

        Returns:
            any: Loaded index instance.
        """
        pass

    @abstractmethod
    def validate(self, index: any, expected_count: int, expected_dim: int) -> bool:
        """
        Validates index properties and verifies basic search integrity.

        Args:
            index (any): Loaded index instance.
            expected_count (int): Expected number of indexed vectors.
            expected_dim (int): Expected dimension.

        Returns:
            bool: True if the index is valid, False otherwise.
        """
        pass

    @abstractmethod
    def vector_count(self, index: any) -> int:
        """
        Returns the total count of indexed vectors.

        Args:
            index (any): Loaded index instance.

        Returns:
            int: Number of vectors.
        """
        pass

    @abstractmethod
    def dimension(self, index: any) -> int:
        """
        Returns vector dimension expected by the index.

        Args:
            index (any): Loaded index instance.

        Returns:
            int: Dimension.
        """
        pass

    @abstractmethod
    def index_type(self) -> str:
        """
        Returns the identifier string of the index type.

        Returns:
            str: Index type name.
        """
        pass
