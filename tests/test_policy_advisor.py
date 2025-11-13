from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from catalystindex.models.common import Tenant
from catalystindex.services.ingestion import IngestionService
from catalystindex.services.ingestion_jobs import DocumentSubmission, IngestionCoordinator, RedisPostgresIngestionJobStore
from catalystindex.services.policy_advisor import PolicyAdvice
from catalystindex.chunking.engine import ChunkingEngine
from catalystindex.embeddings.hash import HashEmbeddingProvider
from catalystindex.parsers.registry import default_registry
from catalystindex.policies.resolver import resolve_policy
from catalystindex.artifacts.store import InMemoryArtifactStore
from catalystindex.storage.term_index import InMemoryTermIndex
from catalystindex.storage.vector_store import InMemoryVectorStore
from catalystindex.telemetry.logger import AuditLogger, MetricsRecorder
from catalystindex.acquisition.service import AcquisitionService, AcquisitionResult


@dataclass
class StubAdvisor:
    advice: PolicyAdvice

    def advise(self, **_: object) -> PolicyAdvice:  # pragma: no cover - trivial stub
        return self.advice

    @property
    def enabled(self) -> bool:
        return True


def test_policy_advisor_influences_policy_selection():
    ingestion = IngestionService(
        parser_registry=default_registry(),
        chunking_engine=ChunkingEngine(namespace="advisor-test"),
        embedding_provider=HashEmbeddingProvider(dimension=32),
        vector_store=InMemoryVectorStore(),
        term_index=InMemoryTermIndex(),
        audit_logger=AuditLogger(),
        metrics=MetricsRecorder(),
    )
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    job_store = RedisPostgresIngestionJobStore(connection=connection)
    coordinator = IngestionCoordinator(
        ingestion_service=ingestion,
        acquisition=AcquisitionService(),
        artifact_store=InMemoryArtifactStore(),
        job_store=job_store,
        metrics=MetricsRecorder(),
        audit_logger=AuditLogger(),
        policy_resolver=resolve_policy,
        policy_advisor=StubAdvisor(PolicyAdvice(policy_name="treatment_planner", confidence=0.9, tags={"doc_type": "treatment"}, notes=None)),
    )
    tenant = Tenant(org_id="org", workspace_id="ws", user_id="u1")
    submission = DocumentSubmission(
        document_id="doc-policy",
        document_title="Behavioral Activation Plan",
        schema=None,
        source_type="inline",
        parser_hint="plain_text",
        metadata={},
        content="Patient treatment plan",
        content_uri=None,
    )
    result = coordinator.ingest_document(tenant, submission)
    document = result.documents[0]
    assert document.policy == "treatment_planner"
    assert document.metadata.get("advisor_policy") == "treatment_planner"
    assert document.metadata.get("advisor_tags") == {"doc_type": "treatment"}
