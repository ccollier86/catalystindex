from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Dict, List, Sequence

from ..models.common import RetrievalResult, Tenant
from ..telemetry.logger import AuditLogger, MetricsRecorder

if TYPE_CHECKING:
    from ..services.semantic_cache import SemanticCache


@dataclass(slots=True)
class GenerationResponse:
    query: str
    summary: str
    citations: Dict[str, Dict[str, object]]
    results: List[RetrievalResult]


class GenerationService:
    """Lightweight generation wrapper using retrieval results with semantic caching."""

    def __init__(
        self,
        *,
        search_service,
        metrics: MetricsRecorder,
        audit_logger: AuditLogger,
        semantic_cache: "SemanticCache | None" = None,
    ) -> None:
        self._search_service = search_service
        self._metrics = metrics
        self._audit_logger = audit_logger
        self._semantic_cache = semantic_cache

    def summarize(
        self,
        tenant: Tenant,
        *,
        query: str,
        knowledge_base_ids: Sequence[str],
        limit: int = 6,
    ) -> GenerationResponse:
        from .search import SearchOptions

        start = perf_counter()

        # Check semantic cache for similar query (global cache, not KB-scoped)
        cached_response = None
        if self._semantic_cache and self._semantic_cache.enabled:
            cached_response = self._semantic_cache.lookup(query)

        if cached_response:
            # Cache hit - return cached response immediately
            duration_ms = (perf_counter() - start) * 1000.0
            self._metrics.record_generation(latency_ms=duration_ms)
            self._audit_logger.generation_completed(tenant, query=query, chunk_ids={})
            # Cached response is just the summary text, reconstruct GenerationResponse
            return GenerationResponse(query=query, summary=cached_response, citations={}, results=[])

        # Cache miss - generate fresh response
        execution = self._search_service.retrieve(
            tenant,
            query=query,
            options=SearchOptions(limit=limit, knowledge_base_ids=tuple(knowledge_base_ids)),
        )
        limited = execution.results[:limit]
        summary_lines = []
        citations: Dict[str, Dict[str, object]] = {}
        for index, result in enumerate(limited, start=1):
            chunk = result.chunk
            snippet = chunk.summary or chunk.text[:200]
            summary_lines.append(f"[{index}] {snippet}")
            citations[str(index)] = {
                "chunk_id": chunk.chunk_id,
                "score": result.score,
                "section": chunk.section_slug,
                "pages": f"{chunk.start_page}-{chunk.end_page}",
            }
        summary = "\n\n".join(summary_lines)

        # Store in semantic cache (global cache)
        if self._semantic_cache and self._semantic_cache.enabled:
            self._semantic_cache.store(query, summary)

        duration_ms = (perf_counter() - start) * 1000.0
        self._metrics.record_generation(latency_ms=duration_ms)
        self._audit_logger.generation_completed(tenant, query=query, chunk_ids=citations)
        return GenerationResponse(query=query, summary=summary, citations=citations, results=limited)


__all__ = ["GenerationService", "GenerationResponse"]
