from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence
from uuid import uuid4

from ..acquisition.service import AcquisitionService
from ..artifacts.store import ArtifactStore
from ..models.common import ChunkRecord, Tenant
from ..telemetry.logger import AuditLogger, MetricsRecorder
from .ingestion import IngestionService


class IngestionJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
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
    parser: str | None
    metadata: Dict[str, object]
    error: str | None = None
    chunks: Sequence[ChunkRecord] = field(default_factory=tuple)


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

    def update(self, record: IngestionJobRecord) -> None:
        raise NotImplementedError

    def get(self, tenant: Tenant, job_id: str) -> IngestionJobRecord | None:
        raise NotImplementedError

    def list(self, tenant: Tenant) -> List[IngestionJobRecord]:
        raise NotImplementedError


class InMemoryIngestionJobStore(IngestionJobStore):
    """Stores ingestion jobs in memory for development and tests."""

    def __init__(self) -> None:
        self._records: Dict[str, IngestionJobRecord] = {}

    def create(self, tenant: Tenant, submissions: Sequence[DocumentSubmission]) -> IngestionJobRecord:
        job_id = uuid4().hex
        now = datetime.utcnow()
        documents = [
            JobDocumentResult(
                document_id=sub.document_id,
                status=DocumentStatus.PENDING,
                policy=None,
                chunk_count=0,
                artifact_uri=None,
                artifact_content_type=None,
                parser=sub.parser_hint,
                metadata=dict(sub.metadata),
            )
            for sub in submissions
        ]
        record = IngestionJobRecord(
            job_id=job_id,
            tenant_key=_tenant_key(tenant),
            status=IngestionJobStatus.PENDING,
            created_at=now,
            updated_at=now,
            documents=documents,
        )
        self._records[job_id] = record
        return record

    def update(self, record: IngestionJobRecord) -> None:
        self._records[record.job_id] = record

    def get(self, tenant: Tenant, job_id: str) -> IngestionJobRecord | None:
        record = self._records.get(job_id)
        if record and record.tenant_key == _tenant_key(tenant):
            return record
        return None

    def list(self, tenant: Tenant) -> List[IngestionJobRecord]:
        tenant_key = _tenant_key(tenant)
        return [record for record in self._records.values() if record.tenant_key == tenant_key]


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
    ) -> None:
        self._ingestion_service = ingestion_service
        self._acquisition = acquisition
        self._artifact_store = artifact_store
        self._job_store = job_store
        self._metrics = metrics
        self._audit_logger = audit_logger
        self._resolve_policy = policy_resolver

    def ingest_document(self, tenant: Tenant, submission: DocumentSubmission) -> IngestionJobRecord:
        record = self._job_store.create(tenant, [submission])
        record = replace(record, status=IngestionJobStatus.RUNNING, updated_at=datetime.utcnow())
        self._job_store.update(record)
        record = self._process_documents(tenant, record, [submission])
        return record

    def ingest_bulk(self, tenant: Tenant, submissions: Sequence[DocumentSubmission]) -> IngestionJobRecord:
        record = self._job_store.create(tenant, submissions)
        record = replace(record, status=IngestionJobStatus.RUNNING, updated_at=datetime.utcnow())
        self._job_store.update(record)
        record = self._process_documents(tenant, record, submissions)
        return record

    def get_job(self, tenant: Tenant, job_id: str) -> IngestionJobRecord | None:
        return self._job_store.get(tenant, job_id)

    def list_jobs(self, tenant: Tenant) -> List[IngestionJobRecord]:
        records = self._job_store.list(tenant)
        return sorted(records, key=lambda record: record.created_at, reverse=True)

    def _process_documents(
        self,
        tenant: Tenant,
        record: IngestionJobRecord,
        submissions: Sequence[DocumentSubmission],
    ) -> IngestionJobRecord:
        updated_documents: List[JobDocumentResult] = []
        failures: List[str] = []
        for submission in submissions:
            try:
                doc_result = self._process_single_document(tenant, record.job_id, submission)
                updated_documents.append(doc_result)
                if doc_result.status == DocumentStatus.FAILED and doc_result.error:
                    failures.append(doc_result.error)
            except Exception as exc:  # pragma: no cover - defensive guard
                failures.append(str(exc))
                updated_documents.append(
                    JobDocumentResult(
                        document_id=submission.document_id,
                        status=DocumentStatus.FAILED,
                        policy=None,
                        chunk_count=0,
                        artifact_uri=None,
                        parser=submission.parser_hint,
                        metadata=dict(submission.metadata),
                        error=str(exc),
                    )
                )
        status = IngestionJobStatus.COMPLETED if not failures else IngestionJobStatus.FAILED
        record = replace(
            record,
            status=status,
            updated_at=datetime.utcnow(),
            error="; ".join(failures) if failures else None,
            documents=updated_documents,
        )
        self._job_store.update(record)
        self._metrics.record_ingestion_job(len(submissions), status=status.value)
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
        policy = self._resolve_policy(submission.document_title, submission.schema)
        parser_name = submission.parser_hint or acquisition.parser_hint or "plain_text"
        document_metadata = {
            **acquisition.metadata,
            **submission.metadata,
            "artifact_uri": artifact.uri,
            "source_type": submission.source_type,
        }
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
            status=DocumentStatus.COMPLETED,
            policy=ingestion_result.policy.policy_name,
            chunk_count=len(ingestion_result.chunks),
            artifact_uri=artifact.uri,
            artifact_content_type=artifact.content_type,
            parser=parser_name,
            metadata=document_metadata,
            chunks=ingestion_result.chunks,
        )


__all__ = [
    "DocumentStatus",
    "DocumentSubmission",
    "IngestionCoordinator",
    "IngestionJobRecord",
    "IngestionJobStatus",
    "IngestionJobStore",
    "InMemoryIngestionJobStore",
    "JobDocumentResult",
]

