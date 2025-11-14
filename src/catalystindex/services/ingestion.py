from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, List, Sequence

from time import perf_counter

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
    embeddings: Sequence[Sequence[float]]


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
        document_metadata: Dict[str, object] | None = None,
        progress_callback: Callable[[str, str, Dict[str, object]], None] | None = None,
    ) -> IngestionResult:
        def emit(stage: str, status: str, details: Dict[str, object] | None = None) -> None:
            if progress_callback:
                progress_callback(stage, status, details or {})

        start = perf_counter()
        parser = self._parser_registry.resolve(parser_name or self._default_parser)
        content_type = None
        if document_metadata and isinstance(document_metadata.get("content_type"), str):
            content_type = document_metadata.get("content_type")
        emit("parsed", "running", {"parser": parser.__class__.__name__})
        try:
            sections_iter: Iterable[SectionText] = parser.parse(
                content,
                document_title=document_title,
                content_type=content_type,
            )
            sections = list(sections_iter)
        except Exception as exc:
            emit("parsed", "failed", {"error": str(exc)})
            raise
        emit("parsed", "succeeded", {"sections": len(sections)})
        emit("chunked", "running", {"policy": policy.policy_name})
        try:
            chunks = [
            self._enrich_chunk(
                tenant,
                document_id,
                chunk,
                policy,
                document_metadata=document_metadata or {},
            )
            for chunk in self._chunking_engine.generate_chunks(sections, policy, document_id)
        ]
        except Exception as exc:
            emit("chunked", "failed", {"error": str(exc)})
            raise
        emit("chunked", "succeeded", {"chunks": len(chunks)})
        emit(
            "embedded",
            "running",
            {"provider": type(self._embedding_provider).__name__},
        )
        try:
            embeddings = tuple(self._embed_chunks(chunks))
        except Exception as exc:
            emit("embedded", "failed", {"error": str(exc)})
            raise
        emit("embedded", "succeeded", {"count": len(embeddings)})
        documents = [
            VectorDocument(chunk=chunk, vector=embedding, track="text")
            for chunk, embedding in zip(chunks, embeddings)
        ]
        emit("uploaded", "running", {"documents": len(documents)})
        try:
            self._vector_store.upsert(tenant, documents)
        except Exception as exc:
            emit("uploaded", "failed", {"error": str(exc)})
            raise
        emit("uploaded", "succeeded", {"documents": len(documents)})
        duration_ms = (perf_counter() - start) * 1000.0
        self._metrics.record_ingestion(len(chunks), latency_ms=duration_ms)
        return IngestionResult(
            document_id=document_id,
            policy=policy,
            chunks=tuple(chunks),
            embeddings=embeddings,
        )

    def _embed_chunks(self, chunks: Sequence[ChunkRecord]) -> Sequence[Sequence[float]]:
        missing_texts = [chunk.text for chunk in chunks if chunk.text not in self._embedding_cache]
        if missing_texts:
            for text, vector in zip(missing_texts, self._embedding_provider.embed(missing_texts)):
                self._embedding_cache[text] = vector
        return [self._embedding_cache[chunk.text] for chunk in chunks]

    def _enrich_chunk(
        self,
        tenant: Tenant,
        document_id: str,
        chunk: ChunkRecord,
        policy: ChunkingPolicy,
        *,
        document_metadata: Dict[str, object],
    ) -> ChunkRecord:
        summary = _summarize(chunk.text, policy.llm_metadata.summary_length if policy.llm_metadata.enabled else 160)
        key_terms = _extract_key_terms(chunk.text, limit=policy.llm_metadata.max_terms)
        confidence_note = None
        if len(chunk.text.split()) > policy.max_chunk_tokens:
            confidence_note = "chunk exceeds policy token limit"
        metadata = dict(chunk.metadata)
        metadata.update(document_metadata)
        metadata.update(
            {
                "summary_model": policy.llm_metadata.model if policy.llm_metadata.enabled else "heuristic",
            }
        )
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
