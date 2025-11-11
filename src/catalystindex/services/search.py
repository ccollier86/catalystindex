from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from ..embeddings.base import EmbeddingProvider
from ..models.common import RetrievalResult, Tenant
from ..storage.vector_store import VectorStoreClient
from ..telemetry.logger import AuditLogger, MetricsRecorder


@dataclass(slots=True)
class SearchOptions:
    economy_mode: bool = False
    track: str = "text"
    filters: Dict[str, str] | None = None
    limit: int = 6


class SearchService:
    """Hybrid retrieval service with optional economy mode."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStoreClient,
        audit_logger: AuditLogger,
        metrics: MetricsRecorder,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._audit_logger = audit_logger
        self._metrics = metrics

    def retrieve(self, tenant: Tenant, *, query: str, options: SearchOptions | None = None) -> List[RetrievalResult]:
        options = options or SearchOptions()
        embedding = next(iter(self._embedding_provider.embed([query])))
        limit = options.limit
        results = self._vector_store.query(
            tenant,
            embedding,
            track=options.track,
            limit=limit,
            filters=options.filters,
        )
        self._metrics.record_search(len(results), economy=options.economy_mode)
        self._audit_logger.search_executed(tenant, query=query, result_count=len(results))
        return results


__all__ = ["SearchService", "SearchOptions"]
