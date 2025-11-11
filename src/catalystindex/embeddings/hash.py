from __future__ import annotations

import hashlib
import math
from typing import Iterable, Sequence

from .base import EmbeddingProvider


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for testing and offline use."""

    def __init__(self, dimension: int = 128) -> None:
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> Iterable[Sequence[float]]:
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            expanded = (digest * (self.dimension // len(digest) + 1))[: self.dimension]
            vector = [float(byte) for byte in expanded]
            norm = math.sqrt(sum(value * value for value in vector))
            if not math.isclose(norm, 0.0):
                vector = [value / norm for value in vector]
            yield vector


__all__ = ["HashEmbeddingProvider"]
