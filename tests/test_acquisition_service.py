from __future__ import annotations

from dataclasses import dataclass

from catalystindex.acquisition.service import AcquisitionResult, AcquisitionService, URLFetcher
from catalystindex.models.common import Tenant
from catalystindex.services.ingestion import IngestionService
from catalystindex.chunking.engine import ChunkingEngine
from catalystindex.embeddings.hash import HashEmbeddingProvider
from catalystindex.parsers.registry import default_registry
from catalystindex.policies.resolver import resolve_policy
from catalystindex.storage.term_index import InMemoryTermIndex
from catalystindex.storage.vector_store import InMemoryVectorStore
from catalystindex.telemetry.logger import AuditLogger, MetricsRecorder


@dataclass(slots=True)
class FakeFetcher(URLFetcher):
    result: AcquisitionResult

    def fetch(self, url: str, *, metadata=None, parser_hint=None) -> AcquisitionResult:  # pragma: no cover - simple stub
        return self.result


def test_acquisition_service_uses_url_fetcher(tmp_path):
    result = AcquisitionResult(
        content=b"Example Firecrawl content",
        content_type="text/html",
        source_uri="https://example.com/page",
        parser_hint="html",
        metadata={"firecrawl": True},
    )
    service = AcquisitionService(url_fetcher=FakeFetcher(result=result))
    acquired = service.acquire(source_type="url", content=None, content_uri="https://example.com/page")
    assert acquired.parser_hint == "html"
    assert acquired.metadata["firecrawl"] is True


def test_ingestion_pipeline_with_url_fetch(tmp_path):
    result = AcquisitionResult(
        content=b"Criterion A. Exposure to trauma",
        content_type="text/plain",
        source_uri="https://example.com/dsm",
        parser_hint="plain_text",
        metadata={"firecrawl": True, "payload_size": 30},
    )
    acquisition = AcquisitionService(url_fetcher=FakeFetcher(result=result))
    ingestion = IngestionService(
        parser_registry=default_registry(),
        chunking_engine=ChunkingEngine(namespace="test"),
        embedding_provider=HashEmbeddingProvider(dimension=32),
        vector_store=InMemoryVectorStore(),
        term_index=InMemoryTermIndex(),
        audit_logger=AuditLogger(),
        metrics=MetricsRecorder(),
    )
    tenant = Tenant(org_id="org", workspace_id="ws", user_id="tester")
    policy = resolve_policy("DSM Criteria", "dsm5")
    record = ingestion.ingest(
        tenant=tenant,
        document_id="doc-url",
        document_title="DSM Criteria",
        content=result.content,
        policy=policy,
        parser_name=result.parser_hint,
        document_metadata=result.metadata,
    )
    assert record.chunks, "URL ingestion should produce chunks"
    assert record.chunks[0].metadata.get("firecrawl") is True
