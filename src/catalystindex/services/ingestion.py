from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ..chunking.engine import ChunkingEngine
from ..embeddings.base import EmbeddingProvider
from ..models.common import ChunkRecord, SectionText, Tenant
from ..parsers.base import ParserAdapter
from ..policies.resolver import ChunkingPolicy
from ..storage.vector_store import VectorDocument, VectorStoreClient
from ..telemetry.logger import AuditLogger, MetricsRecorder


@dataclass(slots=True)
class IngestionResult:
    document_id: str
    policy: ChunkingPolicy
    chunks: Sequence[ChunkRecord]


class IngestionService:
    """Ingestion orchestrator applying parsing, chunking, and upsert."""

    def __init__(
        self,
        *,
        parser: ParserAdapter,
        chunking_engine: ChunkingEngine,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStoreClient,
        audit_logger: AuditLogger,
        metrics: MetricsRecorder,
    ) -> None:
        self._parser = parser
        self._chunking_engine = chunking_engine
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._audit_logger = audit_logger
        self._metrics = metrics

    def ingest(
        self,
        *,
        tenant: Tenant,
        document_id: str,
        document_title: str,
        content: bytes | str,
        policy: ChunkingPolicy,
    ) -> IngestionResult:
        sections: Iterable[SectionText] = self._parser.parse(content, document_title=document_title)
        chunks = self._chunking_engine.generate_chunks(sections, policy, document_id)
        embeddings = list(self._embedding_provider.embed([chunk.text for chunk in chunks]))
        documents = [
            VectorDocument(chunk=chunk, vector=embedding, track="text")
            for chunk, embedding in zip(chunks, embeddings)
        ]
        self._vector_store.upsert(tenant, documents)
        self._metrics.record_ingestion(len(chunks))
        self._audit_logger.ingest_completed(tenant, document_id=document_id, chunk_count=len(chunks), policy=policy.policy_name)
        return IngestionResult(document_id=document_id, policy=policy, chunks=tuple(chunks))


__all__ = ["IngestionService", "IngestionResult"]
