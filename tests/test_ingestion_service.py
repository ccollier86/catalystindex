import pytest

from catalystindex.acquisition.service import AcquisitionService
from catalystindex.artifacts.store import InMemoryArtifactStore
from catalystindex.chunking.engine import ChunkingEngine
from catalystindex.embeddings.hash import HashEmbeddingProvider
from catalystindex.models.common import Tenant
from catalystindex.parsers.registry import default_registry
from catalystindex.policies.resolver import resolve_policy
from catalystindex.services.ingestion import IngestionService
from catalystindex.services.ingestion_jobs import (
    DocumentSubmission,
    IngestionCoordinator,
    IngestionJobStatus,
    InMemoryIngestionJobStore,
)
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
    coordinator = IngestionCoordinator(
        ingestion_service=service,
        acquisition=AcquisitionService(),
        artifact_store=InMemoryArtifactStore(),
        job_store=InMemoryIngestionJobStore(),
        metrics=MetricsRecorder(),
        audit_logger=AuditLogger(),
        policy_resolver=resolve_policy,
    )
    tenant = Tenant(org_id="org", workspace_id="ws", user_id="user")
    submission = DocumentSubmission(
        document_id="doc-2",
        document_title="Treatment Plan",
        schema="treatment_planner",
        source_type="inline",
        parser_hint="plain_text",
        metadata={"source_label": "unit-test"},
        content="Patient treatment objectives and plans",
        content_uri=None,
    )
    job = coordinator.ingest_document(tenant, submission)

    assert job.status == IngestionJobStatus.COMPLETED
    assert len(job.documents) == 1
    document = job.documents[0]
    assert document.chunk_count > 0
    assert document.metadata["source_type"] == "inline"
    assert document.metadata["source_label"] == "unit-test"
    assert document.artifact_uri is not None
