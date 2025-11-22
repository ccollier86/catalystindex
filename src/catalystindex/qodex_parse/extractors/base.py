"""
Base classes for content extraction
"""

from abc import ABC, abstractmethod
from typing import List, Any


class ProcessingStep(ABC):
    """
    Base class for processing steps in the pipeline.
    Similar to OpenParse's ProcessingStep but simpler.
    """

    @abstractmethod
    def process(self, nodes: List[Any]) -> List[Any]:
        """
        Process a list of nodes and return the transformed list.

        Args:
            nodes: List of OpenParse nodes or QodexChunks

        Returns:
            Transformed list of nodes
        """
        pass
