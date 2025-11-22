from __future__ import annotations

from typing import Dict, Sequence

from ..models.common import Tenant
from ..services.ingestion_jobs import DocumentSubmission, IngestionTaskDispatcher

try:  # pragma: no cover - optional dependency import guards
    import redis  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency import guards
    redis = None  # type: ignore

try:  # pragma: no cover - optional dependency import guards
    from rq import Queue  # type: ignore
    from rq.job import Retry  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency import guards
    Queue = None  # type: ignore
    Retry = None  # type: ignore


class RQIngestionTaskDispatcher(IngestionTaskDispatcher):
    """Dispatch ingestion document tasks to an RQ queue."""

    def __init__(
        self,
        *,
        redis_url: str,
        queue_name: str = "ingestion",
        default_timeout: int = 900,
        max_retries: int = 3,
        retry_intervals: Sequence[int] | None = None,
        max_queue_length: int | None = None,
    ) -> None:
        missing: list[str] = []
        if redis is None:
            missing.append("redis")
        if Queue is None or Retry is None:
            missing.append("rq")
        if missing:
            raise ModuleNotFoundError(" or ".join(missing))
        self._redis = redis.Redis.from_url(redis_url)
        self._queue = Queue(queue_name, connection=self._redis, default_timeout=default_timeout)
        self._max_retries = max_retries
        self._retry_intervals = tuple(retry_intervals or ())
        self._max_queue_length = max_queue_length if max_queue_length and max_queue_length > 0 else None

    def can_accept(self, count: int = 1) -> bool:
        if self._max_queue_length is None:
            return True
        try:
            queued = int(self._queue.count)
        except Exception:
            return True
        return queued + max(count, 0) <= self._max_queue_length

    def enqueue(
        self,
        job_id: str,
        tenant: Tenant,
        submission: DocumentSubmission,
        *,
        retry_intervals: Sequence[int] | None = None,
    ) -> None:
        if self._max_queue_length is not None and not self.can_accept(1):
            raise RuntimeError("ingestion queue backpressure: too many queued documents")
        intervals = tuple(retry_intervals or self._retry_intervals)
        retry = None
        if intervals or self._max_retries:
            retry = Retry(max(self._max_retries, 0), interval=list(intervals) or None)
        task_id = f"{job_id}-{submission.document_id}"
        tenant_payload = _serialize_tenant(tenant)
        submission_payload = _serialize_submission(submission)
        self._queue.enqueue(
            "catalystindex.workers.tasks.process_ingestion_document",
            kwargs={
                "job_id": job_id,
                "tenant": tenant_payload,
                "submission": submission_payload,
            },
            job_id=task_id,
            retry=retry,
            meta={
                "tenant": tenant_payload,
                "document_id": submission.document_id,
                "retry_intervals": list(intervals),
            },
        )


def _serialize_tenant(tenant: Tenant) -> Dict[str, str]:
    return {
        "org_id": tenant.org_id,
        "workspace_id": tenant.workspace_id,
        "user_id": tenant.user_id,
    }


def _serialize_submission(submission: DocumentSubmission) -> Dict[str, object]:
    return {
        "document_id": submission.document_id,
        "document_title": submission.document_title,
        "knowledge_base_id": submission.knowledge_base_id,
        "schema": submission.schema,
        "source_type": submission.source_type,
        "parser_hint": submission.parser_hint,
        "metadata": dict(submission.metadata),
        "content": submission.content,
        "content_uri": submission.content_uri,
    }


__all__ = ["RQIngestionTaskDispatcher"]
