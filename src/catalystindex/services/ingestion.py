from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Sequence

from ..storage.term_index import TermIndex

from ..chunking.engine import ChunkingEngine
from ..embeddings.base import EmbeddingProvider
from ..models.common import ChunkRecord, SectionText, Tenant
from ..parsers.registry import ParserRegistry
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
        parser_registry: ParserRegistry,
        chunking_engine: ChunkingEngine,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStoreClient,
        term_index: TermIndex,
        audit_logger: AuditLogger,
        metrics: MetricsRecorder,
    ) -> None:
        self._parser_registry = parser_registry
        self._chunking_engine = chunking_engine
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._term_index = term_index
        self._audit_logger = audit_logger
        self._metrics = metrics
        self._embedding_cache: Dict[str, Sequence[float]] = {}
        self._default_parser = "plain_text"

    def ingest(
        self,
        *,
        tenant: Tenant,
        document_id: str,
        document_title: str,
        content: bytes | str,
        policy: ChunkingPolicy,
        parser_name: str | None = None,
    ) -> IngestionResult:
        parser = self._parser_registry.resolve(parser_name or self._default_parser)
        sections: Iterable[SectionText] = parser.parse(content, document_title=document_title)
        chunks = [
            self._enrich_chunk(tenant, document_id, chunk, policy)
            for chunk in self._chunking_engine.generate_chunks(sections, policy, document_id)
        ]
        embeddings = self._embed_chunks(chunks)
        documents = [
            VectorDocument(chunk=chunk, vector=embedding, track="text")
            for chunk, embedding in zip(chunks, embeddings)
        ]
        self._vector_store.upsert(tenant, documents)
        self._metrics.record_ingestion(len(chunks))
        self._audit_logger.ingest_completed(tenant, document_id=document_id, chunk_count=len(chunks), policy=policy.policy_name)
        return IngestionResult(document_id=document_id, policy=policy, chunks=tuple(chunks))

    def _embed_chunks(self, chunks: Sequence[ChunkRecord]) -> Sequence[Sequence[float]]:
        missing_texts = [chunk.text for chunk in chunks if chunk.text not in self._embedding_cache]
        if missing_texts:
            for text, vector in zip(missing_texts, self._embedding_provider.embed(missing_texts)):
                self._embedding_cache[text] = vector
        return [self._embedding_cache[chunk.text] for chunk in chunks]

    def _enrich_chunk(self, tenant: Tenant, document_id: str, chunk: ChunkRecord, policy: ChunkingPolicy) -> ChunkRecord:
        summary = _summarize(chunk.text, policy.llm_metadata.summary_length if policy.llm_metadata.enabled else 160)
        key_terms = _extract_key_terms(chunk.text, limit=policy.llm_metadata.max_terms)
        confidence_note = None
        if len(chunk.text.split()) > policy.max_chunk_tokens:
            confidence_note = "chunk exceeds policy token limit"
        metadata = {
            **chunk.metadata,
            "summary_model": policy.llm_metadata.model if policy.llm_metadata.enabled else "heuristic",
        }
        enriched = replace(
            chunk,
            summary=summary,
            key_terms=key_terms,
            confidence_note=confidence_note,
            metadata=metadata,
        )
        if key_terms:
            self._term_index.update(tenant, document_id, chunk.chunk_id, key_terms)
        return enriched


def _summarize(text: str, limit: int) -> str | None:
    if not text:
        return None
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if not sentences:
        return text[:limit]
    summary = sentences[0][:limit]
    return summary


def _extract_key_terms(text: str, *, limit: int) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text.lower())
    stopwords = {
        "with",
        "that",
        "this",
        "from",
        "into",
        "about",
        "should",
        "could",
        "would",
    }
    scored: Dict[str, int] = {}
    for token in tokens:
        if token in stopwords:
            continue
        scored[token] = scored.get(token, 0) + 1
    sorted_terms = sorted(scored.items(), key=lambda item: item[1], reverse=True)
    return [term for term, _ in sorted_terms[:limit]]


__all__ = ["IngestionService", "IngestionResult"]
