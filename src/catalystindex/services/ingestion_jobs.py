from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any, Dict, List, Optional, Protocol, Sequence
from uuid import uuid4

from ..acquisition.service import AcquisitionService
from ..artifacts.store import ArtifactStore
from ..models.common import ChunkRecord, Tenant
from ..telemetry.logger import AuditLogger, MetricsRecorder
from .ingestion import IngestionService
from .policy_advisor import PolicyAdvisor, PolicyAdvice


class IngestionJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PARTIAL = "partial"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class DocumentStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(slots=True)
class DocumentSubmission:
    document_id: str
    document_title: str
    schema: str | None
    source_type: str
    parser_hint: str | None
    metadata: Dict[str, object]
    content: bytes | str | None
    content_uri: str | None


@dataclass(slots=True)
class JobDocumentResult:
    document_id: str
    status: DocumentStatus
    policy: str | None
    chunk_count: int
    artifact_uri: str | None
    artifact_content_type: str | None
    artifact_metadata: Dict[str, object]
    parser: str | None
    metadata: Dict[str, object]
    error: str | None = None
    chunks: Sequence[ChunkRecord] = field(default_factory=tuple)


@dataclass(slots=True)
class PersistedJobDocument(JobDocumentResult):
    job_id: str | None = None


@dataclass(slots=True)
class IngestionJobRecord:
    job_id: str
    tenant_key: str
    status: IngestionJobStatus
    created_at: datetime
    updated_at: datetime
    documents: List[JobDocumentResult]
    error: str | None = None


class IngestionJobStore:
    """Persistence abstraction for ingestion jobs."""

    def create(self, tenant: Tenant, submissions: Sequence[DocumentSubmission]) -> IngestionJobRecord:
        raise NotImplementedError

    def mark_document_running(self, job_id: str, document_id: str) -> IngestionJobRecord:
        raise NotImplementedError

    def complete_document(self, job_id: str, document: JobDocumentResult) -> IngestionJobRecord:
        raise NotImplementedError

    def fail_document(
        self,
        job_id: str,
        document_id: str,
        error: str,
        *,
        metadata: Optional[Dict[str, object]] = None,
        parser: str | None = None,
    ) -> IngestionJobRecord:
        raise NotImplementedError

    def get(self, tenant: Tenant, job_id: str) -> IngestionJobRecord | None:
        raise NotImplementedError

    def list(self, tenant: Tenant) -> List[IngestionJobRecord]:
        raise NotImplementedError


class IngestionTaskDispatcher(Protocol):
    def enqueue(
        self,
        job_id: str,
        tenant: Tenant,
        submission: DocumentSubmission,
        *,
        retry_intervals: Sequence[int] | None = None,
    ) -> None:
        ...


class RedisPostgresIngestionJobStore(IngestionJobStore):
    """Stores ingestion jobs in a relational database with Redis-backed status cache."""

    def __init__(
        self,
        *,
        connection: Any,
        redis_client: Optional[object] = None,
        namespace: str = "catalystindex",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._connection = connection
        self._redis = redis_client
        self._namespace = namespace.rstrip(":") or "catalystindex"
        self._logger = logger or logging.getLogger(__name__)
        self._lock = RLock()
        module_name = type(connection).__module__
        self._placeholder = "%s" if "psycopg" in module_name else "?"
        self._ensure_schema()

    # -- public API -----------------------------------------------------

    def create(self, tenant: Tenant, submissions: Sequence[DocumentSubmission]) -> IngestionJobRecord:
        job_id = uuid4().hex
        tenant_key = _tenant_key(tenant)
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        documents: List[PersistedJobDocument] = [
            PersistedJobDocument(
                job_id=job_id,
                document_id=sub.document_id,
                status=DocumentStatus.QUEUED,
                policy=None,
                chunk_count=0,
                artifact_uri=None,
                artifact_content_type=None,
                artifact_metadata={},
                parser=sub.parser_hint,
                metadata=dict(sub.metadata),
                error=None,
                chunks=tuple(),
            )
            for sub in submissions
        ]
        with self._lock:
            self._execute(
                """
                INSERT INTO ingestion_jobs (job_id, tenant_key, status, created_at, updated_at, error)
                VALUES (?, ?, ?, ?, ?, NULL)
                """,
                (
                    job_id,
                    tenant_key,
                    IngestionJobStatus.QUEUED.value,
                    now_iso,
                    now_iso,
                ),
            )
            self._executemany(
                """
                INSERT INTO ingestion_job_documents (
                    job_id,
                    document_id,
                    status,
                    policy,
                    chunk_count,
                    artifact_uri,
                    artifact_content_type,
                    artifact_metadata,
                    parser,
                    metadata,
                    error,
                    chunks
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        job_id,
                        doc.document_id,
                        doc.status.value,
                        doc.policy,
                        doc.chunk_count,
                        doc.artifact_uri,
                        doc.artifact_content_type,
                        _serialize_json(doc.artifact_metadata),
                        doc.parser,
                        _serialize_json(doc.metadata),
                        None,
                        _serialize_json([]),
                    )
                    for doc in documents
                ],
            )
            self._connection.commit()
        record = IngestionJobRecord(
            job_id=job_id,
            tenant_key=tenant_key,
            status=IngestionJobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            documents=[self._to_result(doc) for doc in documents],
            error=None,
        )
        self._cache_job(record)
        return record

    def mark_document_running(self, job_id: str, document_id: str) -> IngestionJobRecord:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._execute(
                """
                UPDATE ingestion_job_documents
                SET status = ?, error = NULL
                WHERE job_id = ? AND document_id = ?
                """,
                (DocumentStatus.RUNNING.value, job_id, document_id),
            )
            self._execute(
                """
                UPDATE ingestion_jobs
                SET status = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (IngestionJobStatus.RUNNING.value, now, job_id),
            )
            self._connection.commit()
        record = self._refresh_job_status(job_id)
        return record

    def complete_document(self, job_id: str, document: JobDocumentResult) -> IngestionJobRecord:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._execute(
                """
                UPDATE ingestion_job_documents
                SET
                    status = ?,
                    policy = ?,
                    chunk_count = ?,
                    artifact_uri = ?,
                    artifact_content_type = ?,
                    artifact_metadata = ?,
                    parser = ?,
                    metadata = ?,
                    error = NULL,
                    chunks = ?
                WHERE job_id = ? AND document_id = ?
                """,
                (
                    DocumentStatus.SUCCEEDED.value,
                    document.policy,
                    document.chunk_count,
                    document.artifact_uri,
                    document.artifact_content_type,
                    _serialize_json(document.artifact_metadata),
                    document.parser,
                    _serialize_json(document.metadata),
                    _serialize_chunks(document.chunks),
                    job_id,
                    document.document_id,
                ),
            )
            self._execute(
                """
                UPDATE ingestion_jobs SET updated_at = ? WHERE job_id = ?
                """,
                (now_iso, job_id),
            )
            self._connection.commit()
        record = self._refresh_job_status(job_id)
        return record

    def fail_document(
        self,
        job_id: str,
        document_id: str,
        error: str,
        *,
        metadata: Optional[Dict[str, object]] = None,
        parser: str | None = None,
    ) -> IngestionJobRecord:
        now_iso = datetime.now(timezone.utc).isoformat()
        existing = self._load_document(job_id, document_id)
        metadata_payload = metadata if metadata is not None else existing.metadata
        parser_value = parser or existing.parser
        with self._lock:
            self._execute(
                """
                UPDATE ingestion_job_documents
                SET status = ?, error = ?, artifact_metadata = ?, metadata = ?, parser = ?, chunks = ?
                WHERE job_id = ? AND document_id = ?
                """,
                (
                    DocumentStatus.FAILED.value,
                    error,
                    _serialize_json(existing.artifact_metadata),
                    _serialize_json(metadata_payload),
                    parser_value,
                    _serialize_chunks(existing.chunks),
                    job_id,
                    document_id,
                ),
            )
            self._execute(
                """
                UPDATE ingestion_jobs SET updated_at = ? WHERE job_id = ?
                """,
                (now_iso, job_id),
            )
            self._connection.commit()
        record = self._refresh_job_status(job_id)
        return record

    def get(self, tenant: Tenant, job_id: str) -> IngestionJobRecord | None:
        record = self._load_job(job_id)
        if record and record.tenant_key == _tenant_key(tenant):
            return record
        return None

    def list(self, tenant: Tenant) -> List[IngestionJobRecord]:
        tenant_key = _tenant_key(tenant)
        cursor = self._execute(
            """
            SELECT job_id
            FROM ingestion_jobs
            WHERE tenant_key = ?
            ORDER BY created_at DESC
            """,
            (tenant_key,),
        )
        job_ids = [row[0] for row in cursor.fetchall()]
        return [self._load_job(job_id) for job_id in job_ids]

    # -- internal helpers ----------------------------------------------

    def _ensure_schema(self) -> None:
        with self._lock:
            self._execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    job_id TEXT PRIMARY KEY,
                    tenant_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT
                )
                """,
            )
            self._execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_job_documents (
                    job_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    policy TEXT,
                    chunk_count INTEGER,
                    artifact_uri TEXT,
                    artifact_content_type TEXT,
                    artifact_metadata TEXT,
                    parser TEXT,
                    metadata TEXT,
                    error TEXT,
                    chunks TEXT,
                    PRIMARY KEY (job_id, document_id)
                )
                """,
            )
            self._connection.commit()
        try:
            self._execute("ALTER TABLE ingestion_job_documents ADD COLUMN artifact_metadata TEXT", ())
            self._connection.commit()
        except Exception:  # pragma: no cover - best-effort schema upgrade
            try:
                self._connection.rollback()
            except Exception:  # pragma: no cover - defensive cleanup
                pass

    def _refresh_job_status(self, job_id: str) -> IngestionJobRecord:
        job_row = self._execute(
            "SELECT tenant_key, created_at FROM ingestion_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if job_row is None:
            raise KeyError(f"job {job_id} not found")
        tenant_key, created_at = job_row
        documents = self._load_documents(job_id)
        status_counts = {
            status: sum(1 for doc in documents if doc.status == status)
            for status in DocumentStatus
        }
        queued = status_counts[DocumentStatus.QUEUED]
        running = status_counts[DocumentStatus.RUNNING]
        succeeded = status_counts[DocumentStatus.SUCCEEDED]
        failed = status_counts[DocumentStatus.FAILED]
        total = queued + running + succeeded + failed
        if total == 0 or queued == total:
            job_status = IngestionJobStatus.QUEUED
        elif running > 0 or queued > 0:
            job_status = IngestionJobStatus.PARTIAL if failed > 0 else IngestionJobStatus.RUNNING
        else:
            if failed == 0:
                job_status = IngestionJobStatus.SUCCEEDED
            elif failed == total:
                job_status = IngestionJobStatus.FAILED
            else:
                job_status = IngestionJobStatus.PARTIAL
        updated_at = datetime.now(timezone.utc)
        errors = [doc.error for doc in documents if doc.error]
        error_value = "; ".join(errors) if errors else None
        with self._lock:
            self._execute(
                """
                UPDATE ingestion_jobs
                SET status = ?, error = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    job_status.value,
                    error_value,
                    updated_at.isoformat(),
                    job_id,
                ),
            )
            self._connection.commit()
        record = IngestionJobRecord(
            job_id=job_id,
            tenant_key=tenant_key,
            status=job_status,
            created_at=datetime.fromisoformat(created_at),
            updated_at=updated_at,
            documents=documents,
            error=error_value,
        )
        self._cache_job(record)
        return record

    def _load_documents(self, job_id: str) -> List[JobDocumentResult]:
        cursor = self._execute(
            """
            SELECT document_id, status, policy, chunk_count, artifact_uri,
                   artifact_content_type, artifact_metadata, parser, metadata, error, chunks
            FROM ingestion_job_documents
            WHERE job_id = ?
            ORDER BY document_id
            """,
            (job_id,),
        )
        documents: List[JobDocumentResult] = []
        for row in cursor.fetchall():
            (
                document_id,
                status,
                policy,
                chunk_count,
                artifact_uri,
                artifact_content_type,
                artifact_metadata,
                parser,
                metadata,
                error,
                chunks,
            ) = row
            documents.append(
                JobDocumentResult(
                    document_id=document_id,
                    status=DocumentStatus(status),
                    policy=policy,
                    chunk_count=chunk_count or 0,
                    artifact_uri=artifact_uri,
                    artifact_content_type=artifact_content_type,
                    artifact_metadata=_deserialize_json(artifact_metadata),
                    parser=parser,
                    metadata=_deserialize_json(metadata),
                    error=error,
                    chunks=_deserialize_chunks(chunks),
                )
            )
        return documents

    def _load_job(self, job_id: str) -> IngestionJobRecord | None:
        cursor = self._execute(
            """
            SELECT tenant_key, status, created_at, updated_at, error
            FROM ingestion_jobs
            WHERE job_id = ?
            """,
            (job_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        tenant_key, status, created_at, updated_at, error = row
        documents = self._load_documents(job_id)
        return IngestionJobRecord(
            job_id=job_id,
            tenant_key=tenant_key,
            status=IngestionJobStatus(status),
            created_at=datetime.fromisoformat(created_at),
            updated_at=datetime.fromisoformat(updated_at),
            documents=documents,
            error=error,
        )

    def _load_document(self, job_id: str, document_id: str) -> JobDocumentResult:
        cursor = self._execute(
            """
            SELECT status, policy, chunk_count, artifact_uri,
                   artifact_content_type, artifact_metadata, parser, metadata, error, chunks
            FROM ingestion_job_documents
            WHERE job_id = ? AND document_id = ?
            """,
            (job_id, document_id),
        )
        row = cursor.fetchone()
        if row is None:
            raise KeyError(f"document {document_id} not found in job {job_id}")
        (
            status,
            policy,
            chunk_count,
            artifact_uri,
            artifact_content_type,
            artifact_metadata,
            parser,
            metadata,
            error,
            chunks,
        ) = row
        return JobDocumentResult(
            document_id=document_id,
            status=DocumentStatus(status),
            policy=policy,
            chunk_count=chunk_count or 0,
            artifact_uri=artifact_uri,
            artifact_content_type=artifact_content_type,
            artifact_metadata=_deserialize_json(artifact_metadata),
            parser=parser,
            metadata=_deserialize_json(metadata),
            error=error,
            chunks=_deserialize_chunks(chunks),
        )

    def _execute(self, sql: str, parameters: Sequence[object] | None = None):
        cursor = self._connection.cursor()
        statement = self._prepare_sql(sql)
        if parameters is None:
            cursor.execute(statement)
        else:
            cursor.execute(statement, parameters)
        return cursor

    def _executemany(self, sql: str, seq_of_parameters: Sequence[Sequence[object]]):
        cursor = self._connection.cursor()
        statement = self._prepare_sql(sql)
        cursor.executemany(statement, seq_of_parameters)
        return cursor

    def _prepare_sql(self, sql: str) -> str:
        if self._placeholder == "?":
            return sql
        return sql.replace("?", self._placeholder)

    def _to_result(self, document: PersistedJobDocument) -> JobDocumentResult:
        return JobDocumentResult(
            document_id=document.document_id,
            status=document.status,
            policy=document.policy,
            chunk_count=document.chunk_count,
            artifact_uri=document.artifact_uri,
            artifact_content_type=document.artifact_content_type,
            artifact_metadata=dict(document.artifact_metadata),
            parser=document.parser,
            metadata=dict(document.metadata),
            error=document.error,
            chunks=document.chunks,
        )

    def _cache_job(self, record: IngestionJobRecord) -> None:
        if self._redis is None:
            return
        job_key = f"{self._namespace}:ingest:{record.job_id}"
        mapping: Dict[str, str] = {
            "status": record.status.value,
            "updated_at": record.updated_at.isoformat(),
        }
        if record.error:
            mapping["error"] = record.error
        try:
            self._redis.hset(job_key, mapping=mapping)
            for document in record.documents:
                doc_key = f"{job_key}:doc:{document.document_id}"
                doc_mapping: Dict[str, str] = {
                    "status": document.status.value,
                    "chunk_count": str(document.chunk_count),
                }
                if document.error:
                    doc_mapping["error"] = document.error
                self._redis.hset(doc_key, mapping=doc_mapping)
        except Exception:  # pragma: no cover - defensive logging
            self._logger.debug("failed to push job state to redis", exc_info=True)


def _serialize_json(payload: object) -> str:
    try:
        return json.dumps(payload)
    except TypeError:
        return json.dumps(json.loads(json.dumps(payload, default=str)))


def _serialize_chunks(chunks: Sequence[ChunkRecord]) -> str:
    if not chunks:
        return _serialize_json([])
    return json.dumps([asdict(chunk) for chunk in chunks])


def _deserialize_json(raw: Any) -> Dict[str, object]:
    if not raw:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
        return {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _deserialize_chunks(raw: Any) -> Sequence[ChunkRecord]:
    if not raw:
        return tuple()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return tuple()
    if not isinstance(payload, list):
        return tuple()
    chunks: List[ChunkRecord] = []
    for item in payload:
        if isinstance(item, dict):
            chunks.append(ChunkRecord(**item))
    return tuple(chunks)


def _tenant_key(tenant: Tenant) -> str:
    return f"{tenant.org_id}:{tenant.workspace_id}"


class IngestionCoordinator:
    """Coordinates ingestion submissions, acquisition, and job lifecycle."""

    def __init__(
        self,
        *,
        ingestion_service: IngestionService,
        acquisition: AcquisitionService,
        artifact_store: ArtifactStore,
        job_store: IngestionJobStore,
        metrics: MetricsRecorder,
        audit_logger: AuditLogger,
        policy_resolver,
        task_dispatcher: IngestionTaskDispatcher | None = None,
        retry_intervals: Sequence[int] | None = None,
        policy_advisor: PolicyAdvisor | None = None,
    ) -> None:
        self._ingestion_service = ingestion_service
        self._acquisition = acquisition
        self._artifact_store = artifact_store
        self._job_store = job_store
        self._metrics = metrics
        self._audit_logger = audit_logger
        self._resolve_policy = policy_resolver
        self._task_dispatcher = task_dispatcher
        self._retry_intervals = tuple(retry_intervals or (15, 30, 60, 120))
        self._policy_advisor = policy_advisor

    def ingest_document(self, tenant: Tenant, submission: DocumentSubmission) -> IngestionJobRecord:
        record = self._job_store.create(tenant, [submission])
        record = self._job_store.mark_document_running(record.job_id, submission.document_id)
        record = self._process_document(tenant, record.job_id, submission)
        return record

    def ingest_bulk(self, tenant: Tenant, submissions: Sequence[DocumentSubmission]) -> IngestionJobRecord:
        record = self._job_store.create(tenant, submissions)
        if not submissions:
            return record
        if self._task_dispatcher:
            for submission in submissions:
                self._task_dispatcher.enqueue(
                    record.job_id,
                    tenant,
                    submission,
                    retry_intervals=self._retry_intervals,
                )
        else:
            for submission in submissions:
                record = self._job_store.mark_document_running(record.job_id, submission.document_id)
                record = self._process_document(tenant, record.job_id, submission)
        return record

    def process_document_task(
        self,
        tenant: Tenant,
        job_id: str,
        submission: DocumentSubmission,
    ) -> IngestionJobRecord:
        record = self._job_store.mark_document_running(job_id, submission.document_id)
        record = self._process_document(tenant, job_id, submission)
        return record

    def get_job(self, tenant: Tenant, job_id: str) -> IngestionJobRecord | None:
        return self._job_store.get(tenant, job_id)

    def list_jobs(self, tenant: Tenant) -> List[IngestionJobRecord]:
        records = self._job_store.list(tenant)
        return sorted(records, key=lambda record: record.created_at, reverse=True)

    @property
    def retry_intervals(self) -> Sequence[int]:
        return self._retry_intervals

    def _process_document(
        self,
        tenant: Tenant,
        job_id: str,
        submission: DocumentSubmission,
    ) -> IngestionJobRecord:
        try:
            result = self._process_single_document(tenant, job_id, submission)
        except Exception as exc:  # pragma: no cover - defensive guard
            record = self._job_store.fail_document(
                job_id,
                submission.document_id,
                str(exc),
                metadata=dict(submission.metadata),
                parser=submission.parser_hint,
            )
            self._record_job_metrics_if_finished(record)
            return record
        record = self._job_store.complete_document(job_id, result)
        self._record_job_metrics_if_finished(record)
        return record

    def _process_single_document(
        self,
        tenant: Tenant,
        job_id: str,
        submission: DocumentSubmission,
    ) -> JobDocumentResult:
        try:
            acquisition = self._acquisition.acquire(
                source_type=submission.source_type,
                content=submission.content,
                content_uri=submission.content_uri,
                metadata=submission.metadata,
                parser_hint=submission.parser_hint,
            )
        except Exception:
            self._metrics.record_dependency_failure("acquisition")
            raise
        artifact = self._artifact_store.store_document(
            tenant,
            job_id=job_id,
            document_id=submission.document_id,
            content=acquisition.content,
            content_type=acquisition.content_type,
            metadata=acquisition.metadata,
        )
        advisor_policy = None
        advisor_metadata: Dict[str, object] = {}
        if self._policy_advisor and not submission.schema:
            preview = self._preview_text(acquisition.content)
            advice = self._policy_advisor.advise(
                title=submission.document_title,
                schema=submission.schema,
                content=preview,
            )
            if advice.policy_name:
                advisor_policy = advice.policy_name
            if advice.confidence is not None:
                advisor_metadata["advisor_confidence"] = advice.confidence
            if advice.notes:
                advisor_metadata["advisor_notes"] = advice.notes
            if advice.tags:
                advisor_metadata["advisor_tags"] = advice.tags
        policy_hint = submission.schema or advisor_policy
        policy = self._resolve_policy(submission.document_title, policy_hint)
        parser_name = submission.parser_hint or acquisition.parser_hint or "plain_text"
        document_metadata = {
            **acquisition.metadata,
            **submission.metadata,
            "artifact_uri": artifact.uri,
            "source_type": submission.source_type,
        }
        if advisor_policy:
            document_metadata["advisor_policy"] = advisor_policy
        document_metadata.update(advisor_metadata)
        if acquisition.content_type and "content_type" not in document_metadata:
            document_metadata["content_type"] = acquisition.content_type
        ingestion_result = self._ingestion_service.ingest(
            tenant=tenant,
            document_id=submission.document_id,
            document_title=submission.document_title,
            content=acquisition.content,
            policy=policy,
            parser_name=parser_name,
            document_metadata=document_metadata,
        )
        self._audit_logger.ingest_completed(
            tenant,
            document_id=submission.document_id,
            chunk_count=len(ingestion_result.chunks),
            policy=ingestion_result.policy.policy_name,
            job_id=job_id,
            source_type=submission.source_type,
        )
        return JobDocumentResult(
            document_id=submission.document_id,
            status=DocumentStatus.SUCCEEDED,
            policy=ingestion_result.policy.policy_name,
            chunk_count=len(ingestion_result.chunks),
            artifact_uri=artifact.uri,
            artifact_content_type=artifact.content_type,
            artifact_metadata=dict(artifact.metadata),
            parser=parser_name,
            metadata=document_metadata,
            chunks=ingestion_result.chunks,
        )

    def _record_job_metrics_if_finished(self, job: IngestionJobRecord) -> None:
        if job.status not in {
            IngestionJobStatus.SUCCEEDED,
            IngestionJobStatus.PARTIAL,
            IngestionJobStatus.FAILED,
        }:
            return
        if not job.documents:
            return
        if any(doc.status in {DocumentStatus.QUEUED, DocumentStatus.RUNNING} for doc in job.documents):
            return
        failed_documents = sum(1 for doc in job.documents if doc.status == DocumentStatus.FAILED)
        self._metrics.record_ingestion_job(
            len(job.documents),
            status=job.status.value,
            failed_documents=failed_documents,
        )

    def _preview_text(self, content: bytes | str, limit: int = 4000) -> str:
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="ignore")
        else:
            text = content
        return text[:limit]


__all__ = [
    "DocumentStatus",
    "DocumentSubmission",
    "IngestionCoordinator",
    "IngestionJobRecord",
    "IngestionJobStatus",
    "IngestionJobStore",
    "IngestionTaskDispatcher",
    "JobDocumentResult",
    "RedisPostgresIngestionJobStore",
]
