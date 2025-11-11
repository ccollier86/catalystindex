import pytest

from catalystindex.chunking.engine import ChunkingEngine
from catalystindex.embeddings.hash import HashEmbeddingProvider
from catalystindex.models.common import Tenant
from catalystindex.parsers.base import PlainTextParser
from catalystindex.policies.resolver import resolve_policy
from catalystindex.services.ingestion import IngestionService
from catalystindex.storage.term_index import TermIndex
from catalystindex.storage.vector_store import InMemoryVectorStore
from catalystindex.telemetry.logger import AuditLogger, MetricsRecorder


def test_ingestion_generates_chunks():
    service = IngestionService(
        parser=PlainTextParser(),
        chunking_engine=ChunkingEngine(namespace="test"),
        embedding_provider=HashEmbeddingProvider(dimension=64),
        vector_store=InMemoryVectorStore(),
        term_index=TermIndex(),
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
