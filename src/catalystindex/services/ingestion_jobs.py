from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any, Dict, List, Optional, Protocol, Sequence
from uuid import uuid4

from ..acquisition.service import AcquisitionService
from ..artifacts.store import ArtifactStore
from ..models.common import ChunkRecord, Tenant
from ..parsers.registry import ParserRegistry, default_registry
from ..policies.resolver import ChunkingPolicy
from ..telemetry.logger import AuditLogger, MetricsRecorder
from .ingestion import IngestionService
from .knowledge_base import KnowledgeBaseStore
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


DOCUMENT_STAGE_SEQUENCE: tuple[str, ...] = (
    "acquired",
    "parsed",
    "chunked",
    "embedded",
    "uploaded",
)


@dataclass(slots=True)
class DocumentSubmission:
    document_id: str
    document_title: str
    knowledge_base_id: str
    schema: str | None
    source_type: str
    parser_hint: str | None
    metadata: Dict[str, object]
    content: bytes | str | None
    content_uri: str | None


@dataclass(slots=True)
class JobDocumentResult:
    document_id: str
    knowledge_base_id: str
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
    progress: Dict[str, Dict[str, object]] = field(default_factory=dict)


@dataclass(slots=True)
class PersistedJobDocument(JobDocumentResult):
    job_id: str | None = None


def _initial_progress() -> Dict[str, Dict[str, object]]:
    return {
        stage: {
            "status": "pending",
            "updated_at": None,
            "details": None,
        }
        for stage in DOCUMENT_STAGE_SEQUENCE
    }


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

    def update_progress(
        self,
        job_id: str,
        document_id: str,
        stage: str,
        status: str,
        details: Optional[Dict[str, object]] = None,
    ) -> None:
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
                knowledge_base_id=sub.knowledge_base_id,
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
                progress=_initial_progress(),
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
                    knowledge_base_id,
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
                    progress
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        job_id,
                        doc.document_id,
                        doc.knowledge_base_id,
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
                        _serialize_json(doc.progress),
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

    def update_progress(
        self,
        job_id: str,
        document_id: str,
        stage: str,
        status: str,
        details: Optional[Dict[str, object]] = None,
    ) -> None:
        normalized_stage = stage.lower()
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            row = self._execute(
                """
                SELECT progress
                FROM ingestion_job_documents
                WHERE job_id = ? AND document_id = ?
                """,
                (job_id, document_id),
            ).fetchone()
            if row is None:
                return
            progress = _deserialize_json(row[0]) or _initial_progress()
            entry = progress.get(normalized_stage, {"status": "pending", "updated_at": None, "details": None})
            entry.update(
                {
                    "status": status,
                    "updated_at": now_iso,
                    "details": details,
                }
            )
            progress[normalized_stage] = entry
            self._execute(
                """
                UPDATE ingestion_job_documents
                SET progress = ?
                WHERE job_id = ? AND document_id = ?
                """,
                (
                    _serialize_json(progress),
                    job_id,
                    document_id,
                ),
            )
            self._connection.commit()

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
                    knowledge_base_id TEXT NOT NULL,
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
                    progress TEXT,
                    PRIMARY KEY (job_id, document_id)
                )
                """,
            )
            self._connection.commit()
        self._maybe_add_column("knowledge_base_id TEXT")
        self._maybe_add_column("artifact_metadata TEXT")
        self._maybe_add_column("progress TEXT")

    def _maybe_add_column(self, column_def: str) -> None:
        try:
            self._execute(f"ALTER TABLE ingestion_job_documents ADD COLUMN {column_def}", ())
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
            SELECT document_id, knowledge_base_id, status, policy, chunk_count, artifact_uri,
                   artifact_content_type, artifact_metadata, parser, metadata, error, chunks, progress
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
                knowledge_base_id,
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
                progress,
            ) = row
            documents.append(
                JobDocumentResult(
                    document_id=document_id,
                    knowledge_base_id=knowledge_base_id or "",
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
                    progress=_deserialize_json(progress) or _initial_progress(),
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
            SELECT knowledge_base_id, status, policy, chunk_count, artifact_uri,
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
            knowledge_base_id,
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
            knowledge_base_id=knowledge_base_id or "",
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
            knowledge_base_id=document.knowledge_base_id,
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
        parser_registry: ParserRegistry | None = None,
        knowledge_base_store: KnowledgeBaseStore | None = None,
    ) -> None:
        if knowledge_base_store is None:
            raise ValueError("knowledge_base_store is required")
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
        self._parser_registry = parser_registry or default_registry()
        self._default_parser = "plain_text"
        self._logger = logging.getLogger(__name__)
        self._knowledge_bases = knowledge_base_store

    def ingest_document(self, tenant: Tenant, submission: DocumentSubmission) -> IngestionJobRecord:
        self._ensure_knowledge_bases(tenant, [submission])
        record = self._job_store.create(tenant, [submission])
        record = self._job_store.mark_document_running(record.job_id, submission.document_id)
        record = self._process_document(tenant, record.job_id, submission)
        return record

    def ingest_bulk(self, tenant: Tenant, submissions: Sequence[DocumentSubmission]) -> IngestionJobRecord:
        self._ensure_knowledge_bases(tenant, submissions)
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
        self._ensure_knowledge_bases(tenant, [submission])
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

    def _ensure_knowledge_bases(self, tenant: Tenant, submissions: Sequence[DocumentSubmission]) -> None:
        for submission in submissions:
            kb_id = (submission.knowledge_base_id or "").strip()
            if not kb_id:
                raise ValueError("knowledge_base_id is required for ingestion submissions")
            metadata = submission.metadata or {}
            description = metadata.get("knowledge_base_description")
            description_text = description if isinstance(description, str) else None
            keywords_value = metadata.get("knowledge_base_keywords")
            keywords: List[str] | None = None
            if isinstance(keywords_value, (list, tuple, set)):
                keywords = [str(value) for value in keywords_value if value]
            self._knowledge_bases.ensure(
                tenant,
                kb_id,
                description=description_text,
                keywords=keywords,
            )

    def _select_parser(
        self,
        submission: DocumentSubmission,
        advisor_parser_hint: str | None,
        acquisition_parser_hint: str | None,
        content_type: str | None,
    ) -> tuple[str, str]:
        candidates = [
            ("submission", submission.parser_hint),
            ("advisor", advisor_parser_hint),
            ("acquisition", acquisition_parser_hint),
            ("mime", self._infer_parser_from_content_type(content_type)),
            ("default", self._default_parser),
        ]
        for source, parser_name in candidates:
            if parser_name:
                if source not in {"submission", "advisor"}:
                    self._logger.warning(
                        "parser.selection.fallback",
                        extra={
                            "document_id": submission.document_id,
                            "knowledge_base_id": submission.knowledge_base_id,
                            "source": source,
                            "parser": parser_name,
                        },
                    )
                return parser_name, source
        return self._default_parser, "default"

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
            self._record_stage(
                job_id,
                submission.document_id,
                "acquired",
                "failed",
                {"error": "acquisition_failed"},
            )
            raise
        self._record_stage(
            job_id,
            submission.document_id,
            "acquired",
            "succeeded",
            {
                "content_type": acquisition.content_type,
                "parser_hint": acquisition.parser_hint,
            },
        )
        artifact = self._artifact_store.store_document(
            tenant,
            job_id=job_id,
            document_id=submission.document_id,
            content=acquisition.content,
            content_type=acquisition.content_type,
            metadata=acquisition.metadata,
        )
        advisor_policy = None
        advisor_parser_hint = None
        advisor_overrides: Dict[str, object] = {}
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
            if advice.parser_hint:
                advisor_parser_hint = advice.parser_hint
            if advice.chunk_overrides:
                advisor_overrides = advice.chunk_overrides
        policy_hint = submission.schema or advisor_policy
        policy = self._resolve_policy(policy_hint)
        if advisor_overrides:
            policy = _apply_policy_overrides(policy, advisor_overrides)
            advisor_metadata.setdefault("advisor_policy_overrides", advisor_overrides)
        parser_name, parser_source = self._select_parser(
            submission,
            advisor_parser_hint,
            acquisition.parser_hint,
            acquisition.content_type,
        )
        document_metadata = {
            **acquisition.metadata,
            **submission.metadata,
            "artifact_uri": artifact.uri,
            "source_type": submission.source_type,
            "knowledge_base_id": submission.knowledge_base_id,
            "parser_source": parser_source,
        }
        if advisor_policy:
            document_metadata["advisor_policy"] = advisor_policy
        document_metadata.update(advisor_metadata)
        if acquisition.content_type and "content_type" not in document_metadata:
            document_metadata["content_type"] = acquisition.content_type
        if advisor_parser_hint:
            document_metadata.setdefault("advisor_parser", advisor_parser_hint)
        ingestion_result = self._ingestion_service.ingest(
            tenant=tenant,
            document_id=submission.document_id,
            document_title=submission.document_title,
            content=acquisition.content,
            policy=policy,
            parser_name=parser_name,
            document_metadata=document_metadata,
            progress_callback=lambda stage, status, info: self._record_stage(
                job_id,
                submission.document_id,
                stage,
                status,
                info,
            ),
        )
        self._persist_pipeline_artifacts(
            tenant,
            job_id,
            submission.document_id,
            policy=ingestion_result.policy,
            chunks=ingestion_result.chunks,
            embeddings=ingestion_result.embeddings,
        )
        keywords = self._collect_keywords(ingestion_result.chunks)
        try:
            self._knowledge_bases.record_document_ingested(
                tenant,
                submission.knowledge_base_id,
                document_title=submission.document_title,
                keywords=keywords,
            )
        except Exception:  # pragma: no cover - catalog updates must not break ingestion
            self._logger.warning(
                "knowledge_base.update_failed",
                extra={"knowledge_base_id": submission.knowledge_base_id, "document_id": submission.document_id},
                exc_info=True,
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
            knowledge_base_id=submission.knowledge_base_id,
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

    def _record_stage(
        self,
        job_id: str,
        document_id: str,
        stage: str,
        status: str,
        details: Optional[Dict[str, object]] = None,
    ) -> None:
        try:
            self._job_store.update_progress(job_id, document_id, stage, status, details)
        except Exception:  # pragma: no cover - best effort
            self._logger.debug(
                "progress.update_failed",
                extra={"job_id": job_id, "document_id": document_id, "stage": stage, "status": status},
                exc_info=True,
            )

    def _collect_keywords(self, chunks: Sequence[ChunkRecord], limit: int = 50) -> List[str]:
        counter: Counter[str] = Counter()
        canonical: Dict[str, str] = {}
        for chunk in chunks:
            for term in chunk.key_terms:
                if not term:
                    continue
                normalized = term.strip()
                if not normalized:
                    continue
                lowered = normalized.lower()
                counter[lowered] += 1
                canonical.setdefault(lowered, normalized)
        return [canonical[key] for key, _ in counter.most_common(limit)]

    def _infer_parser_from_content_type(self, content_type: str | None) -> str | None:
        if not content_type:
            return None
        mime = content_type.split(";")[0].strip().lower()
        if not mime:
            return None
        if "pdf" in mime:
            return "pdf"
        if "word" in mime or "msword" in mime:
            return "docx"
        if "powerpoint" in mime or "presentation" in mime:
            return "pptx"
        if "excel" in mime or "spreadsheet" in mime:
            return "xlsx"
        for name, metadata in self._parser_registry.list_parsers().items():
            if not metadata:
                continue
            normalized = {value.lower() for value in metadata.content_types}
            if mime in normalized:
                return name
        if "pdf" in mime:
            return "pdf"
        if "html" in mime:
            return "html"
        return None

    def _persist_pipeline_artifacts(
        self,
        tenant: Tenant,
        job_id: str,
        document_id: str,
        *,
        policy: ChunkingPolicy,
        chunks: Sequence[ChunkRecord],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        payloads: Dict[str, object] = {
            "policy": _policy_to_dict(policy),
            "chunks": _serialize_chunk_records(chunks),
            "embeddings": _serialize_embeddings(chunks, embeddings),
            "term_index": _build_term_index_snapshot(chunks),
        }
        if len(embeddings) != len(chunks):  # pragma: no cover - diagnostic guard
            self._logger.warning(
                "artifact_store.embedding_mismatch",
                extra={"document_id": document_id, "chunk_count": len(chunks), "embedding_count": len(embeddings)},
            )
        for name, payload in payloads.items():
            try:
                self._artifact_store.store_json_artifact(
                    tenant,
                    job_id=job_id,
                    document_id=document_id,
                    name=name,
                    payload=payload,
                )
            except Exception:  # pragma: no cover - defensive logging
                self._logger.warning(
                    "artifact_store.persist_failed",
                    extra={"document_id": document_id, "artifact": name},
                    exc_info=True,
                )


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


def _policy_to_dict(policy: ChunkingPolicy) -> Dict[str, object]:
    return {
        "policy_name": policy.policy_name,
        "chunk_modes": list(policy.chunk_modes),
        "window_size": policy.window_size,
        "window_overlap": policy.window_overlap,
        "max_chunk_tokens": policy.max_chunk_tokens,
        "chunk_tiers": list(policy.chunk_tiers),
        "highlight_phrases": list(policy.highlight_phrases),
        "required_metadata": list(policy.required_metadata),
        "llm_metadata": {
            "enabled": policy.llm_metadata.enabled,
            "model": policy.llm_metadata.model,
            "summary_length": policy.llm_metadata.summary_length,
            "max_terms": policy.llm_metadata.max_terms,
        },
    }


def _serialize_chunk_records(chunks: Sequence[ChunkRecord]) -> List[Dict[str, object]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "section_slug": chunk.section_slug,
            "text": chunk.text,
            "chunk_tier": chunk.chunk_tier,
            "start_page": chunk.start_page,
            "end_page": chunk.end_page,
            "bbox_pointer": chunk.bbox_pointer,
            "summary": chunk.summary,
            "key_terms": list(chunk.key_terms),
            "requires_previous": chunk.requires_previous,
            "prev_chunk_id": chunk.prev_chunk_id,
            "confidence_note": chunk.confidence_note,
            "metadata": dict(chunk.metadata),
        }
        for chunk in chunks
    ]


def _serialize_embeddings(
    chunks: Sequence[ChunkRecord],
    embeddings: Sequence[Sequence[float]],
) -> List[Dict[str, object]]:
    return [
        {"chunk_id": chunk.chunk_id, "vector": [float(value) for value in vector]}
        for chunk, vector in zip(chunks, embeddings)
    ]


def _build_term_index_snapshot(chunks: Sequence[ChunkRecord]) -> Dict[str, Dict[str, object]]:
    snapshot: Dict[str, Dict[str, object]] = {}
    for chunk in chunks:
        snapshot[chunk.chunk_id] = {
            "aliases": list(chunk.key_terms),
            "section_slug": chunk.section_slug,
            "chunk_tier": chunk.chunk_tier,
        }
    return snapshot


def _apply_policy_overrides(policy: ChunkingPolicy, overrides: Dict[str, object]) -> ChunkingPolicy:
    updates: Dict[str, object] = {}
    if "chunk_modes" in overrides:
        try:
            updates["chunk_modes"] = tuple(overrides["chunk_modes"])
        except TypeError:
            pass
    if "chunk_tiers" in overrides:
        try:
            updates["chunk_tiers"] = tuple(overrides["chunk_tiers"])
        except TypeError:
            pass
    for key in ("window_size", "window_overlap", "max_chunk_tokens"):
        if key in overrides and isinstance(overrides[key], int):
            updates[key] = overrides[key]
    if not updates:
        return policy
    return replace(policy, **updates)
