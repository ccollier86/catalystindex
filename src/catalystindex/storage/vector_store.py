from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from ..models.common import ChunkRecord, RetrievalResult, Tenant


@dataclass(slots=True)
class VectorDocument:
    chunk: ChunkRecord
    vector: Sequence[float]
    track: str


class VectorStoreClient:
    """Interface for vector store operations."""

    def upsert(self, tenant: Tenant, documents: Iterable[VectorDocument]) -> None:
        raise NotImplementedError

    def query(
        self,
        tenant: Tenant,
        vector: Sequence[float],
        *,
        track: str,
        limit: int,
        filters: Dict[str, object] | None = None,
    ) -> List[RetrievalResult]:
        raise NotImplementedError


class InMemoryVectorStore(VectorStoreClient):
    """Tenant-aware in-memory vector store supporting cosine similarity."""

    def __init__(self) -> None:
        self._store: Dict[str, List[VectorDocument]] = {}

    def _key(self, tenant: Tenant) -> str:
        return f"{tenant.org_id}:{tenant.workspace_id}"

    def upsert(self, tenant: Tenant, documents: Iterable[VectorDocument]) -> None:
        docs = self._store.setdefault(self._key(tenant), [])
        docs.extend(documents)

    def query(
        self,
        tenant: Tenant,
        vector: Sequence[float],
        *,
        track: str,
        limit: int,
        filters: Dict[str, object] | None = None,
    ) -> List[RetrievalResult]:
        docs = self._store.get(self._key(tenant), [])
        if not docs:
            return []
        results: List[RetrievalResult] = []
        for doc in docs:
            if doc.track != track:
                continue
            if filters and not _matches_filters(doc.chunk.metadata, filters):
                continue
            denom = float(_norm(doc.vector) * _norm(vector))
            score = _dot(doc.vector, vector) / denom if not math.isclose(denom, 0.0) else 0.0
            results.append(
                RetrievalResult(chunk=doc.chunk, score=score, track=doc.track, vision_context=doc.chunk.metadata.get("vision"))
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]


def _matches_filters(metadata: Dict[str, object], filters: Dict[str, object]) -> bool:
    for key, value in filters.items():
        meta_value = metadata.get(key)
        if isinstance(value, (list, tuple, set)):
            if meta_value not in value:
                return False
        elif str(meta_value) != str(value):
            return False
    return True


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(l * r for l, r in zip(left, right)))


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


__all__ = ["VectorDocument", "VectorStoreClient", "InMemoryVectorStore"]
