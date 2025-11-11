from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from ..embeddings.base import EmbeddingProvider
from ..models.common import RetrievalResult, Tenant
from ..storage.vector_store import VectorStoreClient
from ..storage.term_index import TermIndex
from ..telemetry.logger import AuditLogger, MetricsRecorder


@dataclass(slots=True)
class SearchOptions:
    economy_mode: bool = False
    track: str = "text"
    filters: Dict[str, object] | None = None
    limit: int = 6
    alias_limit: int = 5


class SearchService:
    """Hybrid retrieval service with optional economy mode."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStoreClient,
        term_index: TermIndex | None,
        audit_logger: AuditLogger,
        metrics: MetricsRecorder,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._term_index = term_index
        self._audit_logger = audit_logger
        self._metrics = metrics

    def retrieve(self, tenant: Tenant, *, query: str, options: SearchOptions | None = None) -> List[RetrievalResult]:
        options = options or SearchOptions()
        expanded_query = self._expand_query(query, options)
        embedding = next(iter(self._embedding_provider.embed([expanded_query])))
        limit = options.limit
        results = self._vector_store.query(
            tenant,
            embedding,
            track=options.track,
            limit=limit,
            filters=options.filters,
        )
        self._metrics.record_search(len(results), economy=options.economy_mode)
        self._audit_logger.search_executed(tenant, query=expanded_query, result_count=len(results))
        return results

    def _expand_query(self, query: str, options: SearchOptions) -> str:
        if not self._term_index:
            return query
        aliases = self._term_index.expand_query(query, limit=options.alias_limit)
        if not aliases:
            return query
        alias_text = " ".join(sorted(set(aliases)))
        return f"{query} {alias_text}".strip()


__all__ = ["SearchService", "SearchOptions"]
