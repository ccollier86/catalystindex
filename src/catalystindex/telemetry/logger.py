from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict

from ..models.common import Tenant

LOGGER = logging.getLogger("catalystindex")


@dataclass(slots=True)
class MetricsRecorder:
    """In-memory metrics recorder suitable for testing."""

    ingestion_count: int = 0
    search_count: int = 0
    generation_count: int = 0

    def record_ingestion(self, chunk_count: int) -> None:
        LOGGER.info("ingestion.completed", extra={"chunk_count": chunk_count})
        self.ingestion_count += chunk_count

    def record_ingestion_job(self, document_count: int, *, status: str) -> None:
        LOGGER.info(
            "ingestion.job",
            extra={"document_count": document_count, "status": status},
        )

    def record_search(self, results_count: int, *, economy: bool) -> None:
        LOGGER.info("search.completed", extra={"results_count": results_count, "economy": economy})
        self.search_count += 1

    def record_generation(self) -> None:
        LOGGER.info("generation.completed")
        self.generation_count += 1


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


__all__ = ["MetricsRecorder", "AuditLogger"]
