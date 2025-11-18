from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, List, Sequence

from time import perf_counter

from ..storage.term_index import TermIndex

from ..chunking.engine import ChunkingEngine
from ..embeddings.base import EmbeddingProvider
from ..models.common import ChunkRecord, SectionText, Tenant
from ..parsers.registry import ParserRegistry
from ..policies.resolver import ChunkingPolicy
from ..storage.vector_store import VectorDocument, VectorStoreClient
from ..telemetry.logger import AuditLogger, MetricsRecorder
from ..config.settings import get_settings
import concurrent.futures


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
        enable_sparse_vectors: bool = False,
    ) -> None:
        self._parser_registry = parser_registry
        self._chunking_engine = chunking_engine
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._term_index = term_index
        self._audit_logger = audit_logger
        self._metrics = metrics
        self._enable_sparse = enable_sparse_vectors
        self._embedding_cache: Dict[str, Sequence[float]] = {}
        self._default_parser = "plain_text"
        self._llm_client = None
        try:
            from openai import OpenAI  # type: ignore

            settings = get_settings()
            api_key = settings.policy_advisor.api_key or settings.policy_synthesis.api_key or settings.embeddings.api_key or os.getenv("OPENAI_API_KEY")
            model = settings.policy_advisor.model if settings.policy_advisor and settings.policy_advisor.model else "gpt-4o-mini"
            if api_key:
                self._llm_client = OpenAI(api_key=api_key, base_url=settings.policy_advisor.base_url)
                self._llm_model = model
            else:
                self._llm_client = None
                self._llm_model = None
        except Exception:
            self._llm_client = None
            self._llm_model = None

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
        document_metadata = dict(document_metadata or {})
        parser = self._parser_registry.resolve(parser_name or self._default_parser)
        content_type = None
        if isinstance(document_metadata.get("content_type"), str):
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
            raw_chunks = list(self._chunking_engine.generate_chunks(sections, policy, document_id))
            chunks = self._enrich_chunks_batched(
                tenant,
                document_id,
                raw_chunks,
                policy,
                document_metadata=document_metadata,
                document_title=document_title,
            )
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
        kb_value = document_metadata.get("knowledge_base_id")
        knowledge_base_id = str(kb_value) if kb_value else None
        if not knowledge_base_id:
            raise ValueError("knowledge_base_id metadata is required for vector storage")
        documents = [
            VectorDocument(
                chunk=chunk,
                vector=embedding,
                track="text",
                knowledge_base_id=knowledge_base_id,
                sparse_vector=self._build_sparse_vector(chunk.text) if self._enable_sparse else None,
            )
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

    def ingest_from_cache(
        self,
        *,
        tenant: Tenant,
        document_id: str,
        document_title: str,
        policy: ChunkingPolicy,
        cached_chunks: List[Dict[str, object]],
        cached_embeddings: List[Dict[str, object]],
        document_metadata: Dict[str, object],
        progress_callback: Callable[[str, str, Dict[str, object]], None] | None = None,
    ) -> IngestionResult:
        def emit(stage: str, status: str, details: Dict[str, object] | None = None) -> None:
            if progress_callback:
                progress_callback(stage, status, details or {})

        knowledge_base_id = str(document_metadata.get("knowledge_base_id") or "")
        if not knowledge_base_id:
            raise ValueError("knowledge_base_id metadata is required for vector storage")

        chunks = _deserialize_chunk_records(cached_chunks)
        embeddings_map = _deserialize_embeddings(cached_embeddings)
        updated_chunks: List[ChunkRecord] = []
        embeddings: List[Sequence[float]] = []
        for chunk in chunks:
            metadata = dict(chunk.metadata)
            metadata.update(document_metadata)
            updated_chunk = replace(chunk, metadata=metadata)
            updated_chunks.append(updated_chunk)
            vector = embeddings_map.get(updated_chunk.chunk_id)
            if vector is None:
                raise ValueError(f"missing cached embedding for chunk {updated_chunk.chunk_id}")
            embeddings.append(vector)
            if updated_chunk.key_terms:
                self._term_index.update(
                    tenant,
                    knowledge_base_id,
                    document_id,
                    updated_chunk.chunk_id,
                    updated_chunk.key_terms,
                )

        documents = [
            VectorDocument(
                chunk=chunk,
                vector=vector,
                track="text",
                knowledge_base_id=knowledge_base_id,
                sparse_vector=self._build_sparse_vector(chunk.text) if self._enable_sparse else None,
            )
            for chunk, vector in zip(updated_chunks, embeddings)
        ]

        emit("parsed", "succeeded", {"cached": True})
        emit("chunked", "succeeded", {"cached": True, "chunks": len(updated_chunks)})
        emit("embedded", "succeeded", {"cached": True, "count": len(embeddings)})
        emit("uploaded", "running", {"cached": True, "documents": len(documents)})
        self._vector_store.upsert(tenant, documents)
        emit("uploaded", "succeeded", {"cached": True, "documents": len(documents)})
        self._metrics.record_ingestion(len(updated_chunks), latency_ms=0.0)
        return IngestionResult(
            document_id=document_id,
            policy=policy,
            chunks=tuple(updated_chunks),
            embeddings=tuple(embeddings),
        )

    def _embed_chunks(self, chunks: Sequence[ChunkRecord]) -> Sequence[Sequence[float]]:
        missing_texts = [chunk.text for chunk in chunks if chunk.text not in self._embedding_cache]
        if missing_texts:
            for text, vector in zip(missing_texts, self._embedding_provider.embed(missing_texts)):
                self._embedding_cache[text] = vector
        return [self._embedding_cache[chunk.text] for chunk in chunks]

    def _build_sparse_vector(self, text: str) -> Dict[int, float] | None:
        tokens = [token for token in text.lower().split() if token]
        if not tokens:
            return None
        counts = Counter(tokens)
        if not counts:
            return None
        max_count = max(counts.values())
        if max_count <= 0:
            return None
        scale = 1.0 / float(max_count)
        sparse_vector: Dict[int, float] = {}
        for token, count in counts.items():
            digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % 131071 or 1
            sparse_vector[index] = float(count) * scale
        return sparse_vector

    def _enrich_chunks_batched(
        self,
        tenant: Tenant,
        document_id: str,
        chunks: Sequence[ChunkRecord],
        policy: ChunkingPolicy,
        *,
        document_metadata: Dict[str, object],
        document_title: str | None = None,
        batch_size: int = 8,
        max_workers: int | None = None,
    ) -> List[ChunkRecord]:
        """Batch LLM enrichment for speed/cost; preserves mandatory LLM requirement."""
        if max_workers is None:
            max_workers = min(6, max(1, (len(chunks) // batch_size) + 1))
        payloads = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = []
            titles = []
            skip_mask = []
            for c in batch:
                txt = c.text or ""
                texts.append(txt)
                titles.append(document_title or "")
                skip_mask.append(len(txt.strip()) == 0)
            payloads.append((batch, texts, titles, skip_mask))

        enriched: List[ChunkRecord] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = [
                exe.submit(
                    _llm_enrich_batch,
                    self._llm_client,
                    self._llm_model,
                    texts,
                    titles=titles,
                    summary_len=policy.llm_metadata.summary_length or 200,
                    max_terms=policy.llm_metadata.max_terms or 6,
                    skip_mask=skip_mask,
                )
                for (_, texts, titles, skip_mask) in payloads
            ]
            for (batch, _, _, _), fut in zip(payloads, futures):
                results = fut.result()
                for chunk, (summary, key_terms, req_prev, prev_id, confidence_note, extras) in zip(batch, results):
                    enriched_chunk = self._enrich_chunk_from_llm(
                        tenant,
                        document_id,
                        chunk,
                        policy,
                        document_metadata=document_metadata,
                        document_title=document_title,
                        summary=summary,
                        key_terms=key_terms,
                        req_prev=req_prev,
                        prev_id=prev_id,
                        confidence_note=confidence_note,
                        extras=extras,
                    )
                    enriched.append(enriched_chunk)
        return enriched

    def _enrich_chunk_from_llm(
        self,
        tenant: Tenant,
        document_id: str,
        chunk: ChunkRecord,
        policy: ChunkingPolicy,
        *,
        document_metadata: Dict[str, object],
        document_title: str | None = None,
        summary: str | None,
        key_terms: List[str],
        req_prev: bool | None,
        prev_id: str | None,
        confidence_note: str | None,
        extras: Dict[str, Any],
    ) -> ChunkRecord:
        if summary is None:
            summary = ""
        if req_prev is not None:
            chunk = replace(chunk, requires_previous=bool(req_prev))
        if prev_id:
            chunk = replace(chunk, prev_chunk_id=str(prev_id))
        if confidence_note:
            chunk = replace(chunk, confidence_note=str(confidence_note))
        if len(chunk.text.split()) > policy.max_chunk_tokens:
            confidence_note = "chunk exceeds policy token limit"
        metadata = dict(chunk.metadata)
        metadata.update(document_metadata)
        metadata.update({"summary_model": self._llm_model})
        if extras:
            # Persist any additional LLM-enriched fields without losing existing metadata.
            for k, v in extras.items():
                if v is not None:
                    metadata[k] = v
        enriched = replace(
            chunk,
            summary=summary,
            key_terms=key_terms,
            confidence_note=confidence_note,
            metadata=metadata,
        )
        knowledge_base_id = str(document_metadata.get("knowledge_base_id") or "")
        if key_terms and knowledge_base_id:
            self._term_index.update(tenant, knowledge_base_id, document_id, chunk.chunk_id, key_terms)
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


def _deserialize_chunk_records(payload: List[Dict[str, object]]) -> List[ChunkRecord]:
    records: List[ChunkRecord] = []
    for entry in payload or []:
        records.append(
            ChunkRecord(
                chunk_id=entry.get("chunk_id", ""),
                section_slug=entry.get("section_slug", ""),
                text=entry.get("text", ""),
                chunk_tier=entry.get("chunk_tier", "semantic"),
                start_page=int(entry.get("start_page", 1) or 1),
                end_page=int(entry.get("end_page", 1) or 1),
                bbox_pointer=entry.get("bbox_pointer"),
                summary=entry.get("summary"),
                key_terms=list(entry.get("key_terms") or []),
                requires_previous=bool(entry.get("requires_previous", False)),
                prev_chunk_id=entry.get("prev_chunk_id"),
                confidence_note=entry.get("confidence_note"),
                metadata=dict(entry.get("metadata") or {}),
            )
        )
    return records


def _deserialize_embeddings(payload: List[Dict[str, object]]) -> Dict[str, Sequence[float]]:
    mapping: Dict[str, Sequence[float]] = {}
    for entry in payload or []:
        chunk_id = entry.get("chunk_id")
        vector = entry.get("vector")
        if chunk_id and isinstance(vector, list):
            mapping[str(chunk_id)] = [float(value) for value in vector]
    return mapping


def _llm_enrich(
    client,
    model: str,
    text: str,
    *,
    summary_len: int,
    max_terms: int,
    title: str = "",
) -> tuple[str | None, List[str], bool | None, str | None, str | None, Dict[str, Any]]:
    """Use OpenAI to produce summary, key terms, and dependency hints."""
    prompt = (
        "You are enriching document chunks for a clinical retrieval system. "
        "Use the provided title for context if present.\n"
        f"Return JSON with fields: summary (<= {summary_len} chars), key_terms (array, up to {max_terms}, lowercased, no stopwords), "
        "requires_previous (boolean), prev_chunk_id (string or null), confidence_note (short string label), "
        "extras (object with any additional signals like icd_mentions or insights). "
        "Respond with JSON only."
    )
    clipped_title = title[:200].strip()
    title_line = f"Title: {clipped_title}\n" if clipped_title else ""
    clipped = f"{title_line}{text[:3000]}"
    body = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": clipped},
    ]
    try:
        resp = client.responses.create(model=model, input=body)  # type: ignore[attr-defined]
        raw = resp.output[0].content[0].text  # type: ignore[attr-defined]
        # Be lenient in parsing JSON from the model output.
        data = None
        try:
            data = json.loads(raw)
        except Exception:
            import re

            match = re.search(r"\\{.*\\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
        if not isinstance(data, dict):
            return None, [], None, None, None, {}
        summary = data.get("summary")
        terms = data.get("key_terms") or []
        requires_previous = data.get("requires_previous")
        prev_chunk_id = data.get("prev_chunk_id")
        confidence_note = data.get("confidence_note")
        extras = data.get("extras") if isinstance(data.get("extras"), dict) else {}
        if isinstance(terms, list):
            terms = [str(t).strip().lower() for t in terms if t]
        else:
            terms = []
        if isinstance(summary, str) and summary_len > 0:
            summary = summary[:summary_len]
        return (
            summary if summary else None,
            terms[:max_terms] if max_terms else terms,
            bool(requires_previous) if requires_previous is not None else None,
            str(prev_chunk_id) if prev_chunk_id else None,
            str(confidence_note) if confidence_note else None,
            extras,
        )
    except Exception:
        return None, [], None, None, None, {}


def _llm_enrich_batch(
    client,
    model: str,
    texts: List[str],
    *,
    titles: List[str],
    summary_len: int,
    max_terms: int,
    skip_mask: List[bool] | None = None,
    ) -> List[tuple[str | None, List[str], bool | None, str | None, str | None, Dict[str, Any]]]:
    """Batch LLM enrichment for multiple chunks to reduce latency/cost."""
    if not texts:
        return []
    results = []
    for idx, (text, title) in enumerate(zip(texts, titles)):
        if skip_mask and skip_mask[idx]:
            results.append(("", [], None, None, None, {}))
            continue
        results.append(
            _llm_enrich(
                client,
                model,
                text,
                title=title,
                summary_len=summary_len,
                max_terms=max_terms,
            )
        )
    return results


__all__ = ["IngestionService", "IngestionResult"]
