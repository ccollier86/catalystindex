from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List

from ..models.common import RetrievalResult, Tenant
from ..telemetry.logger import AuditLogger, MetricsRecorder


@dataclass(slots=True)
class GenerationResponse:
    query: str
    summary: str
    citations: Dict[str, Dict[str, object]]
    results: List[RetrievalResult]


class GenerationService:
    """Lightweight generation wrapper using retrieval results."""

    def __init__(self, *, search_service, metrics: MetricsRecorder, audit_logger: AuditLogger) -> None:
        self._search_service = search_service
        self._metrics = metrics
        self._audit_logger = audit_logger

    def summarize(self, tenant: Tenant, *, query: str, limit: int = 6) -> GenerationResponse:
        from .search import SearchOptions

        start = perf_counter()
        execution = self._search_service.retrieve(tenant, query=query, options=SearchOptions(limit=limit))
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
        duration_ms = (perf_counter() - start) * 1000.0
        self._metrics.record_generation(latency_ms=duration_ms)
        self._audit_logger.generation_completed(tenant, query=query, chunk_ids=citations)
        return GenerationResponse(query=query, summary=summary, citations=citations, results=limited)


__all__ = ["GenerationService", "GenerationResponse"]
