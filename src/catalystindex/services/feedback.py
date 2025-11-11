from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Dict, Iterable, Tuple

from ..models.common import Tenant
from ..storage.term_index import TermIndex
from ..telemetry.logger import AuditLogger, MetricsRecorder


@dataclass(slots=True)
class FeedbackRecord:
    """Structured record returned after feedback submission."""

    query: str
    chunk_ids: Tuple[str, ...]
    positive: bool
    recorded_at: datetime
    comment: str | None = None
    metadata: Dict[str, object] = field(default_factory=dict)


class FeedbackService:
    """Coordinates feedback persistence and telemetry hooks."""

    def __init__(
        self,
        *,
        term_index: TermIndex,
        metrics: MetricsRecorder,
        audit_logger: AuditLogger,
    ) -> None:
        self._term_index = term_index
        self._metrics = metrics
        self._audit_logger = audit_logger

    def submit(
        self,
        tenant: Tenant,
        *,
        query: str,
        chunk_ids: Iterable[str],
        positive: bool,
        comment: str | None = None,
        metadata: Dict[str, object] | None = None,
    ) -> FeedbackRecord:
        start = perf_counter()
        normalized_chunk_ids = tuple(chunk_id for chunk_id in chunk_ids if chunk_id)
        if not normalized_chunk_ids:
            raise ValueError("Feedback submissions must include at least one chunk identifier")

        metadata = dict(metadata or {})
        self._term_index.record_feedback(
            tenant,
            query,
            normalized_chunk_ids,
            positive=positive,
        )

        recorded_at = datetime.utcnow()
        self._metrics.record_feedback(
            positive=positive,
            count=len(normalized_chunk_ids),
            latency_ms=(perf_counter() - start) * 1000.0,
        )
        self._audit_logger.feedback_recorded(
            tenant,
            query=query,
            chunk_ids=list(normalized_chunk_ids),
            positive=positive,
            comment=comment,
            metadata=metadata if metadata else None,
        )
        return FeedbackRecord(
            query=query,
            chunk_ids=normalized_chunk_ids,
            positive=positive,
            recorded_at=recorded_at,
            comment=comment,
            metadata=metadata,
        )


__all__ = ["FeedbackService", "FeedbackRecord"]
