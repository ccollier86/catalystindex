from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Dict, Iterable, Tuple

from ..models.common import Tenant
from ..storage.term_index import TermIndex
from ..storage.vector_store import VectorStoreClient
from ..telemetry.logger import AuditLogger, MetricsRecorder


LOGGER = logging.getLogger("catalystindex")


@dataclass(slots=True)
class FeedbackRecord:
    """Structured record returned after feedback submission."""

    query: str
    chunk_ids: Tuple[str, ...]
    positive: bool
    recorded_at: datetime
    comment: str | None = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class FeedbackCounter:
    positive: int = 0
    negative: int = 0
    last_feedback_at: datetime | None = None
    last_comment: str | None = None

    def apply(
        self,
        *,
        positive: bool,
        recorded_at: datetime,
        weight: int = 1,
        comment: str | None = None,
    ) -> None:
        if positive:
            self.positive += weight
        else:
            self.negative += weight
        self.last_feedback_at = recorded_at
        if comment:
            self.last_comment = comment

    @property
    def score(self) -> int:
        return self.positive - self.negative


@dataclass(slots=True)
class FeedbackItemSnapshot:
    identifier: str
    positive: int
    negative: int
    score: int
    last_feedback_at: datetime | None
    last_comment: str | None = None


@dataclass(slots=True)
class FeedbackAnalyticsSnapshot:
    total_positive: int
    total_negative: int
    feedback_ratio: float
    generated_at: datetime
    chunks: Tuple[FeedbackItemSnapshot, ...]
    queries: Tuple[FeedbackItemSnapshot, ...]


@dataclass(slots=True)
class _FeedbackAnalyticsState:
    totals: FeedbackCounter = field(default_factory=FeedbackCounter)
    queries: Dict[str, FeedbackCounter] = field(default_factory=dict)
    chunks: Dict[str, FeedbackCounter] = field(default_factory=dict)

    def record(
        self,
        *,
        query: str,
        chunk_ids: Tuple[str, ...],
        positive: bool,
        recorded_at: datetime,
        comment: str | None,
    ) -> None:
        weight = len(chunk_ids) or 1
        self.totals.apply(positive=positive, recorded_at=recorded_at, weight=weight)
        query_counter = self.queries.setdefault(query, FeedbackCounter())
        query_counter.apply(positive=positive, recorded_at=recorded_at)
        for chunk_id in chunk_ids:
            counter = self.chunks.setdefault(chunk_id, FeedbackCounter())
            counter.apply(positive=positive, recorded_at=recorded_at, comment=comment)

    def snapshot(self, *, generated_at: datetime) -> FeedbackAnalyticsSnapshot:
        total_positive = self.totals.positive
        total_negative = self.totals.negative
        total = total_positive + total_negative
        ratio = (total_positive / total) if total else 0.0
        chunk_snapshots = tuple(
            FeedbackItemSnapshot(
                identifier=chunk_id,
                positive=counter.positive,
                negative=counter.negative,
                score=counter.score,
                last_feedback_at=counter.last_feedback_at,
                last_comment=counter.last_comment,
            )
            for chunk_id, counter in sorted(
                self.chunks.items(),
                key=lambda item: (-item[1].score, item[0]),
            )
        )
        query_snapshots = tuple(
            FeedbackItemSnapshot(
                identifier=query,
                positive=counter.positive,
                negative=counter.negative,
                score=counter.score,
                last_feedback_at=counter.last_feedback_at,
                last_comment=None,
            )
            for query, counter in sorted(
                self.queries.items(),
                key=lambda item: (-item[1].score, item[0]),
            )
        )
        return FeedbackAnalyticsSnapshot(
            total_positive=total_positive,
            total_negative=total_negative,
            feedback_ratio=ratio,
            generated_at=generated_at,
            chunks=chunk_snapshots,
            queries=query_snapshots,
        )


class FeedbackService:
    """Coordinates feedback persistence and telemetry hooks."""

    def __init__(
        self,
        *,
        term_index: TermIndex,
        metrics: MetricsRecorder,
        audit_logger: AuditLogger,
        vector_store: VectorStoreClient | None = None,
    ) -> None:
        self._term_index = term_index
        self._metrics = metrics
        self._audit_logger = audit_logger
        self._vector_store = vector_store
        self._analytics: Dict[str, _FeedbackAnalyticsState] = {}

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
        if self._vector_store:
            try:
                self._vector_store.apply_feedback(
                    tenant,
                    normalized_chunk_ids,
                    positive=positive,
                )
            except Exception:  # pragma: no cover - defensive guard for optional backends
                LOGGER.exception("feedback.vector_store_update_failed")

        recorded_at = datetime.utcnow()
        self._metrics.record_feedback(
            positive=positive,
            count=len(normalized_chunk_ids),
            latency_ms=(perf_counter() - start) * 1000.0,
        )
        analytics_state = self._analytics.setdefault(_tenant_key(tenant), _FeedbackAnalyticsState())
        analytics_state.record(
            query=query,
            chunk_ids=normalized_chunk_ids,
            positive=positive,
            recorded_at=recorded_at,
            comment=comment,
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

    def analytics(self, tenant: Tenant) -> FeedbackAnalyticsSnapshot:
        generated_at = datetime.utcnow()
        state = self._analytics.get(_tenant_key(tenant))
        if not state:
            return FeedbackAnalyticsSnapshot(
                total_positive=0,
                total_negative=0,
                feedback_ratio=0.0,
                generated_at=generated_at,
                chunks=tuple(),
                queries=tuple(),
            )
        return state.snapshot(generated_at=generated_at)


def _tenant_key(tenant: Tenant) -> str:
    return f"{tenant.org_id}:{tenant.workspace_id}"


__all__ = [
    "FeedbackService",
    "FeedbackRecord",
    "FeedbackAnalyticsSnapshot",
    "FeedbackItemSnapshot",
]
