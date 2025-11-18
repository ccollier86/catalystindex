from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from catalystindex.models.common import SectionText, Tenant
from catalystindex.services.ingestion import IngestionService
from catalystindex.services.ingestion_jobs import DocumentSubmission, IngestionCoordinator, RedisPostgresIngestionJobStore
from catalystindex.services.knowledge_base import KnowledgeBaseStore
from catalystindex.services.policy_advisor import PolicyAdvice
from catalystindex.services.policy_synthesizer import PolicySynthesisResult
from catalystindex.chunking.engine import ChunkingEngine
from catalystindex.embeddings.hash import HashEmbeddingProvider
from catalystindex.parsers.base import ParserAdapter, ParserMetadata
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
    kb_store = KnowledgeBaseStore(connection=connection)
    coordinator = IngestionCoordinator(
        ingestion_service=ingestion,
        acquisition=AcquisitionService(),
        artifact_store=InMemoryArtifactStore(),
        job_store=job_store,
        metrics=MetricsRecorder(),
        audit_logger=AuditLogger(),
        policy_resolver=resolve_policy,
        policy_advisor=StubAdvisor(PolicyAdvice(policy_name="treatment_planner", confidence=0.9, tags={"doc_type": "treatment"}, notes=None)),
        parser_registry=default_registry(),
        knowledge_base_store=kb_store,
    )
    tenant = Tenant(org_id="org", workspace_id="ws", user_id="u1")
    submission = DocumentSubmission(
        document_id="doc-policy",
        document_title="Behavioral Activation Plan",
        knowledge_base_id="kb-policy",
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


def test_policy_advisor_sets_parser_and_overrides():
    registry = default_registry()

    class DummyPDFParser(ParserAdapter):
        def parse(self, content, *, document_title: str, content_type: str | None = None):  # pragma: no cover - simple stub
            text = content.decode("utf-8") if isinstance(content, bytes) else str(content)
            yield SectionText(section_slug="body", title=document_title, text=text)

    registry.register(
        "pdf",
        DummyPDFParser(),
        metadata=ParserMetadata(name="pdf", content_types=("application/pdf",), description="dummy pdf"),
    )

    ingestion = IngestionService(
        parser_registry=registry,
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
    kb_store = KnowledgeBaseStore(connection=connection)
    advisor = StubAdvisor(
        PolicyAdvice(
            policy_name="dsm5",
            confidence=0.7,
            tags={},
            notes=None,
            parser_hint="pdf",
            chunk_overrides={"chunk_modes": ["window"], "window_overlap": 40},
        )
    )
    coordinator = IngestionCoordinator(
        ingestion_service=ingestion,
        acquisition=AcquisitionService(),
        artifact_store=InMemoryArtifactStore(),
        job_store=job_store,
        metrics=MetricsRecorder(),
        audit_logger=AuditLogger(),
        policy_resolver=resolve_policy,
        policy_advisor=advisor,
        parser_registry=registry,
        knowledge_base_store=kb_store,
    )
    tenant = Tenant(org_id="org", workspace_id="ws", user_id="u1")
    submission = DocumentSubmission(
        document_id="doc-advisor-parser",
        document_title="Unknown PDF",
        knowledge_base_id="kb-advisor",
        schema=None,
        source_type="inline",
        parser_hint=None,
        metadata={"content_type": "application/pdf"},
        content="Criterion A text",
        content_uri=None,
    )
    job = coordinator.ingest_document(tenant, submission)
    document = job.documents[0]
    assert document.parser == "pdf"
    assert document.metadata.get("advisor_parser") == "pdf"
    assert document.metadata.get("advisor_policy_overrides") == {"chunk_modes": ["window"], "window_overlap": 40}


def test_resolve_ccbhc_policy_template():
    policy = resolve_policy("ccbhc")
    assert policy.policy_name == "ccbhc"
    assert "section" in policy.chunk_modes
    assert policy.llm_metadata.enabled is True
    assert policy.window_size == 480


def test_policy_synthesizer_applies_overrides():
    class StubSynth:
        def synthesize(self, **_: object) -> PolicySynthesisResult:
            return PolicySynthesisResult(
                chunk_modes=["window"],
                window_size=256,
                window_overlap=64,
                max_chunk_tokens=512,
                notes="custom",
            )

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
    kb_store = KnowledgeBaseStore(connection=connection)
    coordinator = IngestionCoordinator(
        ingestion_service=ingestion,
        acquisition=AcquisitionService(),
        artifact_store=InMemoryArtifactStore(),
        job_store=job_store,
        metrics=MetricsRecorder(),
        audit_logger=AuditLogger(),
        policy_resolver=resolve_policy,
        policy_advisor=None,
        policy_synthesizer=StubSynth(),
        parser_registry=default_registry(),
        knowledge_base_store=kb_store,
    )
    tenant = Tenant(org_id="org", workspace_id="ws", user_id="u1")
    submission = DocumentSubmission(
        document_id="doc-synth",
        document_title="Ops Report",
        knowledge_base_id="kb-synth",
        schema=None,
        source_type="inline",
        parser_hint="plain_text",
        metadata={"content_type": "text/plain"},
        content="Operational metrics and staffing tables",
        content_uri=None,
    )
    job = coordinator.ingest_document(tenant, submission)
    document = job.documents[0]
    assert document.metadata.get("synthesized_policy")
