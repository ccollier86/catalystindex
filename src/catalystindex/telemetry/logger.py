from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List

from ..models.common import Tenant

LOGGER = logging.getLogger("catalystindex")


@dataclass(slots=True)
class MetricsRecorder:
    """In-memory metrics recorder suitable for testing."""

    ingestion_count: int = 0
    search_count: int = 0
    generation_count: int = 0
    economy_requests: int = 0
    premium_requests: int = 0
    feedback_positive: int = 0
    feedback_negative: int = 0
    ingestion_latency_ms: List[float] = field(default_factory=list)
    search_latency_ms: List[float] = field(default_factory=list)
    generation_latency_ms: List[float] = field(default_factory=list)
    feedback_latency_ms: List[float] = field(default_factory=list)
    dependency_failures: Dict[str, int] = field(default_factory=dict)
    dependency_retries: Dict[str, int] = field(default_factory=dict)

    def record_ingestion(self, chunk_count: int, *, latency_ms: float | None = None) -> None:
        LOGGER.info("ingestion.completed", extra={"chunk_count": chunk_count, "latency_ms": latency_ms})
        self.ingestion_count += chunk_count
        if latency_ms is not None:
            self.ingestion_latency_ms.append(latency_ms)

    def record_ingestion_job(self, document_count: int, *, status: str) -> None:
        LOGGER.info(
            "ingestion.job",
            extra={"document_count": document_count, "status": status},
        )

    def record_search(self, results_count: int, *, economy: bool, latency_ms: float | None = None) -> None:
        LOGGER.info(
            "search.completed",
            extra={"results_count": results_count, "economy": economy, "latency_ms": latency_ms},
        )
        self.search_count += 1
        if economy:
            self.economy_requests += 1
        else:
            self.premium_requests += 1
        if latency_ms is not None:
            self.search_latency_ms.append(latency_ms)

    def record_generation(self, *, latency_ms: float | None = None) -> None:
        LOGGER.info("generation.completed", extra={"latency_ms": latency_ms})
        self.generation_count += 1
        if latency_ms is not None:
            self.generation_latency_ms.append(latency_ms)

    def record_feedback(self, *, positive: bool, count: int, latency_ms: float | None = None) -> None:
        LOGGER.info(
            "feedback.recorded",
            extra={"positive": positive, "count": count, "latency_ms": latency_ms},
        )
        if positive:
            self.feedback_positive += count
        else:
            self.feedback_negative += count
        if latency_ms is not None:
            self.feedback_latency_ms.append(latency_ms)

    def record_dependency_failure(self, dependency: str) -> None:
        LOGGER.warning("dependency.failure", extra={"dependency": dependency})
        self.dependency_failures[dependency] = self.dependency_failures.get(dependency, 0) + 1

    def record_dependency_retry(self, dependency: str) -> None:
        LOGGER.info("dependency.retry", extra={"dependency": dependency})
        self.dependency_retries[dependency] = self.dependency_retries.get(dependency, 0) + 1

    @property
    def feedback_positive_ratio(self) -> float:
        total = self.feedback_positive + self.feedback_negative
        if not total:
            return 0.0
        return self.feedback_positive / total


class AuditLogger:
    """Structured audit logger writing to application logs."""

    def ingest_completed(
        self,
        tenant: Tenant,
        *,
        document_id: str,
        chunk_count: int,
        policy: str,
        job_id: str | None = None,
        source_type: str | None = None,
        metadata: Dict[str, object] | None = None,
    ) -> None:
        extra = {
            "tenant": asdict(tenant),
            "document_id": document_id,
            "chunk_count": chunk_count,
            "policy": policy,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if job_id:
            extra["job_id"] = job_id
        if source_type:
            extra["source_type"] = source_type
        if metadata:
            extra["metadata"] = metadata
        LOGGER.info(
            "audit.ingest_completed",
            extra=extra,
        )

    def search_executed(self, tenant: Tenant, *, query: str, result_count: int) -> None:
        LOGGER.info(
            "audit.search_executed",
            extra={
                "tenant": asdict(tenant),
                "query_hash": hash(query),
                "result_count": result_count,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    def generation_completed(self, tenant: Tenant, *, query: str, chunk_ids: Dict[str, object]) -> None:
        LOGGER.info(
            "audit.generation_completed",
            extra={
                "tenant": asdict(tenant),
                "query_hash": hash(query),
                "chunk_ids": chunk_ids,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    def feedback_recorded(
        self,
        tenant: Tenant,
        *,
        query: str,
        chunk_ids: List[str],
        positive: bool,
        comment: str | None = None,
        metadata: Dict[str, object] | None = None,
    ) -> None:
        payload: Dict[str, object] = {
            "tenant": asdict(tenant),
            "query_hash": hash(query),
            "chunk_ids": chunk_ids,
            "positive": positive,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if comment:
            payload["comment"] = comment
        if metadata:
            payload["metadata"] = metadata
        LOGGER.info("audit.feedback_recorded", extra=payload)


__all__ = ["MetricsRecorder", "AuditLogger"]
