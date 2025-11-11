from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..models.common import RetrievalResult, Tenant
from ..telemetry.logger import AuditLogger, MetricsRecorder


@dataclass(slots=True)
class GenerationResponse:
    query: str
    summary: str
    citations: Dict[str, float]
    results: List[RetrievalResult]


class GenerationService:
    """Lightweight generation wrapper using retrieval results."""

    def __init__(self, *, search_service, metrics: MetricsRecorder, audit_logger: AuditLogger) -> None:
        self._search_service = search_service
        self._metrics = metrics
        self._audit_logger = audit_logger

    def summarize(self, tenant: Tenant, *, query: str, limit: int = 6) -> GenerationResponse:
        from .search import SearchOptions

        results = self._search_service.retrieve(tenant, query=query, options=SearchOptions(limit=limit))
        limited = results[:limit]
        combined_text = "\n".join(result.chunk.text for result in limited)
        summary = combined_text[:500] + ("..." if len(combined_text) > 500 else "")
        citations = {result.chunk.chunk_id: result.score for result in limited}
        self._metrics.record_generation()
        self._audit_logger.generation_completed(tenant, query=query, chunk_ids=citations)
        return GenerationResponse(query=query, summary=summary, citations=citations, results=limited)


__all__ = ["GenerationService", "GenerationResponse"]
