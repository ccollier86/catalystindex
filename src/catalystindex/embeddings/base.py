from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence


class EmbeddingProvider(ABC):
    """Interface for generating embeddings."""

    dimension: int

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> Iterable[Sequence[float]]:
        raise NotImplementedError


__all__ = ["EmbeddingProvider"]
