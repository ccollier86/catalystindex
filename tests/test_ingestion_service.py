import sqlite3

import pytest

from catalystindex.acquisition.service import AcquisitionService
from catalystindex.artifacts.store import InMemoryArtifactStore, LocalArtifactStore
from catalystindex.chunking.engine import ChunkingEngine
from catalystindex.embeddings.hash import HashEmbeddingProvider
from catalystindex.models.common import SectionText, Tenant
from catalystindex.parsers.base import ParserAdapter, ParserMetadata
from catalystindex.parsers.registry import ParserRegistry, default_registry
from catalystindex.policies.resolver import resolve_policy
from catalystindex.services.ingestion import IngestionService
from catalystindex.services.ingestion_jobs import (
    DocumentSubmission,
    DocumentStatus,
    IngestionCoordinator,
    IngestionJobStatus,
    RedisPostgresIngestionJobStore,
)
from catalystindex.services.knowledge_base import KnowledgeBaseStore
from catalystindex.storage.term_index import InMemoryTermIndex
from catalystindex.storage.vector_store import InMemoryVectorStore
from catalystindex.telemetry.logger import AuditLogger, MetricsRecorder


def test_ingestion_generates_chunks():
    service = IngestionService(
        parser_registry=default_registry(),
        chunking_engine=ChunkingEngine(namespace="test"),
        embedding_provider=HashEmbeddingProvider(dimension=64),
        vector_store=InMemoryVectorStore(),
        term_index=InMemoryTermIndex(),
        audit_logger=AuditLogger(),
        metrics=MetricsRecorder(),
    )

    tenant = Tenant(org_id="org", workspace_id="ws", user_id="user")
    policy = resolve_policy("DSM Criteria", "dsm5")
    result = service.ingest(
        tenant=tenant,
        document_id="doc-1",
        document_title="DSM Criteria",
        content="Criterion A. Exposure to trauma Criterion B. Intrusion",
        policy=policy,
        document_metadata={"knowledge_base_id": "kb-test"},
    )

    assert result.document_id == "doc-1"
    assert len(result.chunks) > 0
    assert all(chunk.metadata["policy"] == policy.policy_name for chunk in result.chunks)
    assert any(chunk.summary for chunk in result.chunks)
    assert any(chunk.key_terms for chunk in result.chunks)


def test_ingestion_coordinator_creates_job():
    service = IngestionService(
        parser_registry=default_registry(),
        chunking_engine=ChunkingEngine(namespace="test"),
        embedding_provider=HashEmbeddingProvider(dimension=64),
        vector_store=InMemoryVectorStore(),
        term_index=InMemoryTermIndex(),
        audit_logger=AuditLogger(),
        metrics=MetricsRecorder(),
    )
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    job_store = RedisPostgresIngestionJobStore(connection=connection)
    kb_store = KnowledgeBaseStore(connection=connection)
    coordinator = IngestionCoordinator(
        ingestion_service=service,
        acquisition=AcquisitionService(),
        artifact_store=InMemoryArtifactStore(),
        job_store=job_store,
        metrics=MetricsRecorder(),
        audit_logger=AuditLogger(),
        policy_resolver=resolve_policy,
        parser_registry=default_registry(),
        knowledge_base_store=kb_store,
    )
    tenant = Tenant(org_id="org", workspace_id="ws", user_id="user")
    submission = DocumentSubmission(
        document_id="doc-2",
        document_title="Treatment Plan",
        knowledge_base_id="kb-treatment",
        schema="treatment_planner",
        source_type="inline",
        parser_hint="plain_text",
        metadata={"source_label": "unit-test"},
        content="Patient treatment objectives and plans",
        content_uri=None,
    )
    job = coordinator.ingest_document(tenant, submission)

    assert job.status == IngestionJobStatus.SUCCEEDED
    assert len(job.documents) == 1
    document = job.documents[0]
    assert document.status == DocumentStatus.SUCCEEDED
    assert document.chunk_count > 0
    assert document.metadata["source_type"] == "inline"
    assert document.metadata["source_label"] == "unit-test"
    assert document.metadata["parser_source"] == "submission"
    assert document.artifact_uri is not None
    assert document.artifact_metadata["payload_size"] > 0
    assert document.progress["acquired"]["status"] == "succeeded"
    assert document.progress["uploaded"]["status"] == "succeeded"


def test_ingest_bulk_enqueues_tasks():
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    job_store = RedisPostgresIngestionJobStore(connection=connection)
    kb_store = KnowledgeBaseStore(connection=connection)
    service = IngestionService(
        parser_registry=default_registry(),
        chunking_engine=ChunkingEngine(namespace="test"),
        embedding_provider=HashEmbeddingProvider(dimension=64),
        vector_store=InMemoryVectorStore(),
        term_index=InMemoryTermIndex(),
        audit_logger=AuditLogger(),
        metrics=MetricsRecorder(),
    )

    class DummyDispatcher:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, tuple[int, ...]]] = []

        def enqueue(
            self,
            job_id: str,
            tenant,
            submission: DocumentSubmission,
            *,
            retry_intervals,
        ) -> None:
            self.calls.append((job_id, submission.document_id, tuple(retry_intervals or ())))

    dispatcher = DummyDispatcher()
    coordinator = IngestionCoordinator(
        ingestion_service=service,
        acquisition=AcquisitionService(),
        artifact_store=InMemoryArtifactStore(),
        job_store=job_store,
        metrics=MetricsRecorder(),
        audit_logger=AuditLogger(),
        policy_resolver=resolve_policy,
        task_dispatcher=dispatcher,
        retry_intervals=(5, 10, 20),
        parser_registry=default_registry(),
        knowledge_base_store=kb_store,
    )

    tenant = Tenant(org_id="org", workspace_id="ws", user_id="user")
    submissions = [
        DocumentSubmission(
            document_id=f"doc-{idx}",
            document_title=f"Document {idx}",
            knowledge_base_id=f"kb-{idx}",
            schema=None,
            source_type="inline",
            parser_hint=None,
            metadata={},
            content="text",
            content_uri=None,
        )
        for idx in range(2)
    ]

    job = coordinator.ingest_bulk(tenant, submissions)

    assert job.status == IngestionJobStatus.QUEUED
    assert all(doc.status == DocumentStatus.QUEUED for doc in job.documents)
    assert len(dispatcher.calls) == 2
    assert all(call[2] == (5, 10, 20) for call in dispatcher.calls)


def test_ingestion_persists_pipeline_artifacts(tmp_path):
    service = IngestionService(
        parser_registry=default_registry(),
        chunking_engine=ChunkingEngine(namespace="test"),
        embedding_provider=HashEmbeddingProvider(dimension=64),
        vector_store=InMemoryVectorStore(),
        term_index=InMemoryTermIndex(),
        audit_logger=AuditLogger(),
        metrics=MetricsRecorder(),
    )
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    job_store = RedisPostgresIngestionJobStore(connection=connection)
    kb_store = KnowledgeBaseStore(connection=connection)
    artifact_store = LocalArtifactStore(str(tmp_path / "artifacts"))
    coordinator = IngestionCoordinator(
        ingestion_service=service,
        acquisition=AcquisitionService(),
        artifact_store=artifact_store,
        job_store=job_store,
        metrics=MetricsRecorder(),
        audit_logger=AuditLogger(),
        policy_resolver=resolve_policy,
        parser_registry=default_registry(),
        knowledge_base_store=kb_store,
    )
    tenant = Tenant(org_id="org", workspace_id="ws", user_id="user")
    submission = DocumentSubmission(
        document_id="doc-artifacts",
        document_title="DSM Criteria",
        knowledge_base_id="kb-artifacts",
        schema="dsm5",
        source_type="inline",
        parser_hint="plain_text",
        metadata={},
        content="Criterion A. Exposure to trauma",
        content_uri=None,
    )

    job = coordinator.ingest_document(tenant, submission)

    assert artifact_store.json_artifact_exists(
        tenant,
        job_id=job.job_id,
        document_id=submission.document_id,
        name="chunks",
    )
    policy_payload = artifact_store.load_json_artifact(
        tenant,
        job_id=job.job_id,
        document_id=submission.document_id,
        name="policy",
    )
    assert isinstance(policy_payload, dict)
    assert policy_payload["policy_name"] == job.documents[0].policy
    assert job.documents[0].progress["embedded"]["status"] == "succeeded"


def test_content_type_infers_pdf_parser(tmp_path):
    class DummyPDFParser(ParserAdapter):
        def __init__(self) -> None:
            self.calls = 0

        def parse(self, content: bytes | str, *, document_title: str, content_type: str | None = None):
            self.calls += 1
            text = content.decode("utf-8") if isinstance(content, bytes) else str(content)
            yield SectionText(
                section_slug="body",
                title=document_title,
                text=text,
                metadata={"parser": "dummy"},
            )

    registry = default_registry()
    dummy_parser = DummyPDFParser()
    registry.register(
        "pdf",
        dummy_parser,
        metadata=ParserMetadata(
            name="pdf",
            content_types=("application/pdf",),
            description="Dummy PDF parser for tests",
        ),
    )

    service = IngestionService(
        parser_registry=registry,
        chunking_engine=ChunkingEngine(namespace="test"),
        embedding_provider=HashEmbeddingProvider(dimension=64),
        vector_store=InMemoryVectorStore(),
        term_index=InMemoryTermIndex(),
        audit_logger=AuditLogger(),
        metrics=MetricsRecorder(),
    )
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    job_store = RedisPostgresIngestionJobStore(connection=connection)
    kb_store = KnowledgeBaseStore(connection=connection)
    coordinator = IngestionCoordinator(
        ingestion_service=service,
        acquisition=AcquisitionService(),
        artifact_store=LocalArtifactStore(str(tmp_path / "artifacts")),
        job_store=job_store,
        metrics=MetricsRecorder(),
        audit_logger=AuditLogger(),
        policy_resolver=resolve_policy,
        parser_registry=registry,
        knowledge_base_store=kb_store,
    )
    tenant = Tenant(org_id="org", workspace_id="ws", user_id="user")
    submission = DocumentSubmission(
        document_id="doc-pdf",
        document_title="PDF Upload",
        knowledge_base_id="kb-pdf",
        schema=None,
        source_type="inline",
        parser_hint=None,
        metadata={"content_type": "application/pdf"},
        content="Sample PDF text",
        content_uri=None,
    )

    job = coordinator.ingest_document(tenant, submission)
    document = job.documents[0]
    assert document.parser == "pdf"
    assert dummy_parser.calls == 1
    assert document.metadata["parser_source"] == "acquisition"
