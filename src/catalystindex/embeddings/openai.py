from __future__ import annotations

from typing import Iterable, Sequence

from openai import OpenAI

from .base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by OpenAI's embeddings API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAIEmbeddingProvider requires an API key")
        if not model:
            raise ValueError("OpenAIEmbeddingProvider requires a model name")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        # dimension is not enforced by OpenAI client; leave as informational
        self.dimension = 0

    def embed(self, texts: Sequence[str]) -> Iterable[Sequence[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=list(texts))
        vectors = []
        for item in response.data:
            vector = list(getattr(item, "embedding", []))
            vectors.append(vector)
        return vectors


__all__ = ["OpenAIEmbeddingProvider"]
