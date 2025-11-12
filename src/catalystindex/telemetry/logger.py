from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from statistics import mean
from typing import Dict, Iterable, List

try:  # pragma: no cover - optional dependency
    from prometheus_client import Counter as PrometheusCounter
    from prometheus_client import Histogram as PrometheusHistogram
    from prometheus_client import start_http_server
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    PrometheusCounter = None
    PrometheusHistogram = None
    start_http_server = None

from ..models.common import Tenant

LOGGER = logging.getLogger("catalystindex")


@dataclass(slots=True)
class _PrometheusHandles:
    ingestion_counter: object
    ingestion_latency: object
    search_counter: object
    search_latency: object
    generation_counter: object
    generation_latency: object
    feedback_counter: object
    feedback_latency: object
    dependency_failures: object
    dependency_retries: object
    firecrawl_failures: object


@dataclass(slots=True)
class MetricsRecorder:
    """In-memory metrics recorder with optional Prometheus export."""

    namespace: str = "catalystindex"
    enable_prometheus: bool = False
    exporter_port: int | None = None
    exporter_address: str = "0.0.0.0"
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
    _prometheus: _PrometheusHandles | None = field(init=False, default=None)
    _exporter_started: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if not self.enable_prometheus:
            return
        if PrometheusCounter is None or PrometheusHistogram is None:
            LOGGER.warning(
                "prometheus.unavailable",
                extra={"namespace": self.namespace},
            )
            return
        metric_prefix = self.namespace.replace("-", "_")
        self._prometheus = _PrometheusHandles(
            ingestion_counter=PrometheusCounter(
                f"{metric_prefix}_ingestion_chunks_total",
                "Total chunks processed during ingestion.",
            ),
            ingestion_latency=PrometheusHistogram(
                f"{metric_prefix}_ingestion_latency_seconds",
                "Ingestion latency distribution.",
                buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            ),
            search_counter=PrometheusCounter(
                f"{metric_prefix}_search_requests_total",
                "Total search requests processed.",
            ),
            search_latency=PrometheusHistogram(
                f"{metric_prefix}_search_latency_seconds",
                "Search latency distribution.",
                buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
            ),
            generation_counter=PrometheusCounter(
                f"{metric_prefix}_generation_requests_total",
                "Total generation requests processed.",
            ),
            generation_latency=PrometheusHistogram(
                f"{metric_prefix}_generation_latency_seconds",
                "Generation latency distribution.",
                buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            ),
            feedback_counter=PrometheusCounter(
                f"{metric_prefix}_feedback_events_total",
                "Total feedback events recorded.",
            ),
            feedback_latency=PrometheusHistogram(
                f"{metric_prefix}_feedback_latency_seconds",
                "Feedback submission latency distribution.",
                buckets=(0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
            ),
            dependency_failures=PrometheusCounter(
                f"{metric_prefix}_dependency_failures_total",
                "Total dependency failures recorded.",
                labelnames=("dependency",),
            ),
            dependency_retries=PrometheusCounter(
                f"{metric_prefix}_dependency_retries_total",
                "Total dependency retry attempts recorded.",
                labelnames=("dependency",),
            ),
            firecrawl_failures=PrometheusCounter(
                f"{metric_prefix}_firecrawl_failures_total",
                "Total Firecrawl acquisition failures.",
            ),
        )
        if self.exporter_port is not None and start_http_server is not None:
            try:
                start_http_server(self.exporter_port, addr=self.exporter_address)
            except OSError as exc:  # pragma: no cover - I/O errors are environment specific
                LOGGER.warning(
                    "prometheus.exporter_start_failed",
                    extra={"error": str(exc), "port": self.exporter_port},
                )
            else:
                self._exporter_started = True

    def record_ingestion(self, chunk_count: int, *, latency_ms: float | None = None) -> None:
        LOGGER.info("ingestion.completed", extra={"chunk_count": chunk_count, "latency_ms": latency_ms})
        self.ingestion_count += chunk_count
        if self._prometheus:
            self._prometheus.ingestion_counter.inc(chunk_count)
        if latency_ms is not None:
            self.ingestion_latency_ms.append(latency_ms)
            if self._prometheus:
                self._prometheus.ingestion_latency.observe(latency_ms / 1000.0)

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
        if self._prometheus:
            self._prometheus.search_counter.inc()
        if economy:
            self.economy_requests += 1
        else:
            self.premium_requests += 1
        if latency_ms is not None:
            self.search_latency_ms.append(latency_ms)
            if self._prometheus:
                self._prometheus.search_latency.observe(latency_ms / 1000.0)

    def record_generation(self, *, latency_ms: float | None = None) -> None:
        LOGGER.info("generation.completed", extra={"latency_ms": latency_ms})
        self.generation_count += 1
        if self._prometheus:
            self._prometheus.generation_counter.inc()
        if latency_ms is not None:
            self.generation_latency_ms.append(latency_ms)
            if self._prometheus:
                self._prometheus.generation_latency.observe(latency_ms / 1000.0)

    def record_feedback(self, *, positive: bool, count: int, latency_ms: float | None = None) -> None:
        LOGGER.info(
            "feedback.recorded",
            extra={"positive": positive, "count": count, "latency_ms": latency_ms},
        )
        if positive:
            self.feedback_positive += count
        else:
            self.feedback_negative += count
        if self._prometheus:
            self._prometheus.feedback_counter.inc(count)
        if latency_ms is not None:
            self.feedback_latency_ms.append(latency_ms)
            if self._prometheus:
                self._prometheus.feedback_latency.observe(latency_ms / 1000.0)

    def record_dependency_failure(self, dependency: str) -> None:
        LOGGER.warning("dependency.failure", extra={"dependency": dependency})
        self.dependency_failures[dependency] = self.dependency_failures.get(dependency, 0) + 1
        if self._prometheus:
            self._prometheus.dependency_failures.labels(dependency=dependency).inc()
            if dependency.lower() == "firecrawl":
                self._prometheus.firecrawl_failures.inc()

    def record_dependency_retry(self, dependency: str) -> None:
        LOGGER.info("dependency.retry", extra={"dependency": dependency})
        self.dependency_retries[dependency] = self.dependency_retries.get(dependency, 0) + 1
        if self._prometheus:
            self._prometheus.dependency_retries.labels(dependency=dependency).inc()

    @property
    def feedback_positive_ratio(self) -> float:
        total = self.feedback_positive + self.feedback_negative
        if not total:
            return 0.0
        return self.feedback_positive / total

    @property
    def exporter_running(self) -> bool:
        return self._exporter_started

    def snapshot(self) -> Dict[str, object]:
        return {
            "ingestion": {
                "chunks": self.ingestion_count,
                "latency_ms": self._latency_summary(self.ingestion_latency_ms),
            },
            "search": {
                "requests": self.search_count,
                "economy_requests": self.economy_requests,
                "premium_requests": self.premium_requests,
                "latency_ms": self._latency_summary(self.search_latency_ms),
            },
            "generation": {
                "requests": self.generation_count,
                "latency_ms": self._latency_summary(self.generation_latency_ms),
            },
            "feedback": {
                "positive": self.feedback_positive,
                "negative": self.feedback_negative,
                "ratio": self.feedback_positive_ratio,
                "latency_ms": self._latency_summary(self.feedback_latency_ms),
            },
            "dependencies": {
                "failures": dict(self.dependency_failures),
                "retries": dict(self.dependency_retries),
            },
            "exporter": {
                "enabled": bool(self._prometheus),
                "running": self._exporter_started,
                "address": self.exporter_address if self._exporter_started else None,
                "port": self.exporter_port if self._exporter_started else None,
            },
        }

    def _latency_summary(self, values: Iterable[float]) -> Dict[str, float | int]:
        data = list(values)
        if not data:
            return {"count": 0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        data.sort()
        return {
            "count": len(data),
            "avg": float(mean(data)),
            "p50": float(self._percentile(data, 0.5)),
            "p95": float(self._percentile(data, 0.95)),
            "max": float(data[-1]),
        }

    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        if not data:
            return 0.0
        if percentile <= 0:
            return float(data[0])
        if percentile >= 1:
            return float(data[-1])
        index = percentile * (len(data) - 1)
        lower = int(index)
        upper = min(lower + 1, len(data) - 1)
        weight = index - lower
        return float(data[lower] * (1 - weight) + data[upper] * weight)


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
