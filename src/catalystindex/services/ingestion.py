from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, List, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from ..sparse.qodex_generator import QodexSparseGenerator

from time import perf_counter

from ..storage.term_index import TermIndex
from ..storage.visual_store import VisualStore

from ..chunking.engine import ChunkingEngine
from ..embeddings.base import EmbeddingProvider
from ..models.common import ChunkRecord, PageRecord, SectionText, Tenant, VisualElement
from ..parsers.registry import ParserRegistry
from ..parsers.visual_linker import VisualLinker
from ..policies.resolver import ChunkingPolicy
from ..storage.vector_store import PageDocument, VectorDocument, VectorStoreClient
from ..telemetry.logger import AuditLogger, MetricsRecorder
from ..config.settings import get_settings
import concurrent.futures

# Import tiktoken for accurate token counting (used for validation only)
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    # Initialize tokenizer for cl100k_base (used by text-embedding-3-* models)
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
except ImportError:
    TIKTOKEN_AVAILABLE = False
    _TOKENIZER = None


def _count_tokens(text: str) -> int:
    """
    Count actual tokens using tiktoken (accurate) or estimate from words (fallback).

    OpenAI embedding models (text-embedding-3-*) use cl100k_base encoding.
    Max tokens: 8191 for text-embedding-3-small/large.

    NOTE: This is used for VALIDATION only. Chunking engine should prevent
    chunks from ever exceeding the limit.
    """
    if TIKTOKEN_AVAILABLE and _TOKENIZER:
        try:
            return len(_TOKENIZER.encode(text))
        except Exception:
            pass

    # Fallback: Estimate tokens from words (1.3x multiplier for safety)
    word_count = len(text.split())
    return int(word_count * 1.3)


@dataclass(slots=True)
class IngestionResult:
    document_id: str
    policy: ChunkingPolicy
    chunks: Sequence[ChunkRecord]
    embeddings: Sequence[Sequence[float]]
    visual_elements: Sequence[VisualElement] = ()  # Images/tables extracted from document


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
        visual_store: VisualStore | None = None,
        enable_sparse_vectors: bool = False,
        sparse_generator: "QodexSparseGenerator | None" = None,
        llm_batch_size: int = 8,
        llm_max_workers: int | None = None,
        llm_max_retries: int = 2,
        llm_retry_delay_base: float = 0.5,
        llm_max_concurrent_per_batch: int | None = None,
    ) -> None:
        self._parser_registry = parser_registry
        self._chunking_engine = chunking_engine
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._term_index = term_index
        self._audit_logger = audit_logger
        self._metrics = metrics
        self._visual_store = visual_store  # Optional visual element storage
        self._enable_sparse = enable_sparse_vectors
        self._sparse_generator = sparse_generator  # Qodex metadata-aware sparse vector generator
        self._llm_batch_size = llm_batch_size
        self._llm_max_workers = llm_max_workers
        self._llm_max_retries = llm_max_retries
        self._llm_retry_delay_base = llm_retry_delay_base
        self._llm_max_concurrent_per_batch = llm_max_concurrent_per_batch
        self._embedding_cache: Dict[str, Sequence[float]] = {}
        self._default_parser = "plain_text"
        self._llm_client = None
        self._visual_linker = VisualLinker()  # Service for linking images/tables to chunks
        self._logger = metrics.logger if hasattr(metrics, "logger") else logging.getLogger(__name__)
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
        content: bytes | str | None = None,
        sections: Sequence[SectionText] | None = None,
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

        # Use pre-parsed sections if provided, otherwise parse content
        raw_elements: List[object] = []  # Will hold raw elements for visual extraction

        if sections is not None:
            # Sections already parsed (from policy orchestrator flow)
            sections_list = list(sections)
            emit("parsed", "succeeded", {"sections": len(sections_list), "source": "pre_parsed"})
        else:
            # Parse content (legacy flow or when sections not available)
            if content is None:
                raise ValueError("Either 'content' or 'sections' must be provided")
            parser = self._parser_registry.resolve(parser_name or self._default_parser)
            content_type = None
            if isinstance(document_metadata.get("content_type"), str):
                content_type = document_metadata.get("content_type")
            emit("parsed", "running", {"parser": parser.__class__.__name__})
            try:
                # Try parse_with_elements() for visual extraction support
                if hasattr(parser, "parse_with_elements"):
                    sections_list, raw_elements = parser.parse_with_elements(
                        content,
                        document_title=document_title,
                        content_type=content_type,
                    )
                    self._logger.info(
                        f"Parser '{parser.__class__.__name__}' returned {len(raw_elements)} raw elements "
                        f"for visual extraction"
                    )
                else:
                    # Fallback to standard parse() for parsers without visual support
                    sections_iter: Iterable[SectionText] = parser.parse(
                        content,
                        document_title=document_title,
                        content_type=content_type,
                    )
                    sections_list = list(sections_iter)
            except Exception as exc:
                emit("parsed", "failed", {"error": str(exc)})
                raise
            emit("parsed", "succeeded", {"sections": len(sections_list)})
        emit("chunked", "running", {"policy": policy.policy_name})
        try:
            raw_chunks = list(self._chunking_engine.generate_chunks(sections_list, policy, document_id))
        except Exception as exc:
            emit("chunked", "failed", {"error": str(exc)})
            raise
        emit("chunked", "succeeded", {"chunks": len(raw_chunks)})

        # Visual linking: Extract images/tables and link to chunks
        visual_elements: List[VisualElement] = []
        if raw_elements:
            emit("visual_linking", "running", {"raw_elements": len(raw_elements)})
            try:
                visual_result = self._visual_linker.extract_and_link(
                    elements=raw_elements,
                    chunks=raw_chunks,
                    document_id=document_id,
                )
                visual_elements = visual_result.visual_elements
                raw_chunks = visual_result.chunks_with_visuals

                emit(
                    "visual_linking",
                    "succeeded",
                    {
                        "visual_elements": len(visual_elements),
                        "linked_chunks": sum(
                            1 for c in raw_chunks if c.metadata.get("visual_element_ids")
                        ),
                    },
                )
            except Exception as exc:
                # Don't fail ingestion if visual linking fails - log and continue
                self._logger.warning(
                    f"Visual linking failed for document '{document_id}': {exc}",
                    exc_info=True,
                )
                emit("visual_linking", "failed", {"error": str(exc)})

        # Check if qodex-parse already provided enhanced metadata
        has_qodex_metadata = (
            raw_chunks
            and len(raw_chunks) > 0
            and raw_chunks[0].metadata.get("parser") == "qodex"
            and ("keywords" in raw_chunks[0].metadata or "search_terms" in raw_chunks[0].metadata)
        )

        if has_qodex_metadata:
            # Use qodex-provided metadata instead of LLM enrichment
            emit("enriched", "running", {"source": "qodex", "enabled": False})
            chunks = self._extract_qodex_metadata(tenant, document_id, raw_chunks, document_metadata)
            emit("enriched", "succeeded", {"source": "qodex", "chunks": len(chunks)})
        elif policy.llm_metadata and policy.llm_metadata.enabled:
            # Fall back to LLM enrichment for non-qodex parsers
            emit("enriched", "running", {"source": "llm", "enabled": True})
            try:
                chunks = self._enrich_chunks_batched(
                    tenant,
                    document_id,
                    raw_chunks,
                    policy,
                    document_metadata=document_metadata,
                    document_title=document_title,
                    batch_size=self._llm_batch_size,
                    max_workers=self._llm_max_workers,
                    max_retries=self._llm_max_retries,
                    retry_delay_base=self._llm_retry_delay_base,
                    max_concurrent_per_batch=self._llm_max_concurrent_per_batch,
                )
            except Exception as exc:
                emit("enriched", "failed", {"error": str(exc)})
                raise
            emit("enriched", "succeeded", {"source": "llm", "chunks": len(chunks)})
        else:
            # No enrichment - just pass through with document metadata
            emit("enriched", "running", {"source": "none", "enabled": False})
            chunks = [
                replace(chunk, metadata={**chunk.metadata, **document_metadata})
                for chunk in raw_chunks
            ]
            emit("enriched", "succeeded", {"source": "none", "chunks": len(chunks)})

        # NVIDIA-style page extraction: Extract pages and link to chunks
        pages: List[PageRecord] = []
        emit("page_extraction", "running", {})
        try:
            pages, chunks = self._extract_pages(
                document_id=document_id,
                sections=sections_list,
                chunks=chunks,
            )
            emit("page_extraction", "succeeded", {"pages": len(pages)})
        except Exception as exc:
            # Don't fail ingestion if page extraction fails - log and continue
            self._logger.warning(
                f"Page extraction failed for document '{document_id}': {exc}",
                exc_info=True,
            )
            emit("page_extraction", "failed", {"error": str(exc)})

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

        # Embed pages for NVIDIA-style context expansion
        page_embeddings: List[Sequence[float]] = []
        if pages:
            emit("page_embedded", "running", {"pages": len(pages)})
            try:
                page_texts = [page.text for page in pages]
                page_embeddings = list(self._embedding_provider.embed(page_texts))
                emit("page_embedded", "succeeded", {"count": len(page_embeddings)})
            except Exception as exc:
                # Don't fail ingestion if page embedding fails - log and continue
                self._logger.warning(
                    f"Page embedding failed for document '{document_id}': {exc}",
                    exc_info=True,
                )
                emit("page_embedded", "failed", {"error": str(exc)})
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
                sparse_vector=self._build_sparse_vector(chunk) if self._enable_sparse else None,
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

        # Upsert page-level vectors for NVIDIA-style context expansion
        if pages and page_embeddings:
            page_documents = [
                PageDocument(
                    page=page,
                    vector=embedding,
                    track="text",
                    knowledge_base_id=knowledge_base_id,
                    sparse_vector=self._build_sparse_vector(page.text) if self._enable_sparse else None,
                )
                for page, embedding in zip(pages, page_embeddings)
            ]
            emit("page_uploaded", "running", {"pages": len(page_documents)})
            try:
                self._vector_store.upsert_pages(tenant, page_documents)
                emit("page_uploaded", "succeeded", {"pages": len(page_documents)})
                self._logger.info(
                    f"Uploaded {len(page_documents)} page vectors for document '{document_id}'"
                )
            except Exception as exc:
                # Don't fail ingestion if page upsert fails - log and continue
                self._logger.warning(
                    f"Failed to upsert page vectors for document '{document_id}': {exc}",
                    exc_info=True,
                )
                emit("page_uploaded", "failed", {"error": str(exc)})

        # Store visual elements if any were extracted
        if visual_elements and self._visual_store:
            try:
                self._visual_store.store_visuals(
                    tenant=tenant,
                    document_id=document_id,
                    knowledge_base_id=knowledge_base_id,
                    visual_elements=visual_elements,
                )
                self._logger.info(
                    f"Stored {len(visual_elements)} visual elements for document '{document_id}'"
                )
            except Exception as exc:
                # Don't fail ingestion if visual storage fails - log and continue
                self._logger.warning(
                    f"Failed to store visual elements for document '{document_id}': {exc}",
                    exc_info=True,
                )

        duration_ms = (perf_counter() - start) * 1000.0
        self._metrics.record_ingestion(len(chunks), latency_ms=duration_ms)
        return IngestionResult(
            document_id=document_id,
            policy=policy,
            chunks=tuple(chunks),
            embeddings=embeddings,
            visual_elements=tuple(visual_elements),
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
        """
        Embed chunks after validating they don't exceed token limits.

        CRITICAL: Chunks should NEVER exceed limits - this is enforced by the
        chunking engine. If a chunk exceeds the limit, it indicates a bug in
        the chunking logic and will raise an error.
        """
        EMBEDDING_TOKEN_LIMIT = 8191

        # Validate all chunks before embedding
        for chunk in chunks:
            token_count = _count_tokens(chunk.text)
            if token_count > EMBEDDING_TOKEN_LIMIT:
                raise RuntimeError(
                    f"CRITICAL: Chunk '{chunk.chunk_id}' has {token_count} tokens, "
                    f"which exceeds the embedding model limit of {EMBEDDING_TOKEN_LIMIT}. "
                    f"This violates the chunking policy contract and indicates a bug in "
                    f"the chunking engine. Chunks should NEVER exceed this limit."
                )

        # Embed chunks that aren't in cache
        missing_texts = []
        for chunk in chunks:
            if chunk.text not in self._embedding_cache:
                missing_texts.append(chunk.text)

        if missing_texts:
            for text, vector in zip(missing_texts, self._embedding_provider.embed(missing_texts)):
                self._embedding_cache[text] = vector

        # Return embeddings
        return [self._embedding_cache[chunk.text] for chunk in chunks]

    def _build_sparse_vector(self, chunk_or_text: ChunkRecord | str) -> Dict[int, float] | None:
        """Build sparse vector from chunk (with qodex metadata) or raw text.

        Args:
            chunk_or_text: ChunkRecord with metadata, or plain text string

        Returns:
            Sparse vector as {index: weight} dict, or None if no content
        """
        # Route to qodex generator if available and input is ChunkRecord
        if isinstance(chunk_or_text, ChunkRecord) and self._sparse_generator:
            return self._sparse_generator.generate(chunk_or_text)

        # Fallback to simple TF-based sparse vector (for pages or when generator unavailable)
        text = chunk_or_text.text if isinstance(chunk_or_text, ChunkRecord) else chunk_or_text
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
        max_retries: int = 2,
        retry_delay_base: float = 0.5,
        max_concurrent_per_batch: int | None = None,
    ) -> List[ChunkRecord]:
        """
        Batch LLM enrichment with parallel execution, retries, and observability.

        Args:
            batch_size: Chunks per batch (recommended: 8-16)
            max_workers: Parallel batches (recommended: 3-6)
            max_retries: Retry attempts per chunk (default: 2)
            retry_delay_base: Base delay for exponential backoff (default: 0.5s)
            max_concurrent_per_batch: Max parallel LLM calls within batch (default: batch_size)
        """
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

        # Aggregate metrics across all batches
        total_metrics = {
            "total_batches": len(payloads),
            "total_chunks": len(chunks),
            "total_time": 0.0,
            "total_skipped": 0,
            "total_failed": 0,
            "total_empty": 0,
        }

        enriched: List[ChunkRecord] = []
        enrich_start = perf_counter()

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
                    max_retries=max_retries,
                    retry_delay_base=retry_delay_base,
                    max_concurrent=max_concurrent_per_batch,
                    logger=self._logger,
                )
                for (_, texts, titles, skip_mask) in payloads
            ]

            for (batch, _, _, _), fut in zip(payloads, futures):
                results, batch_metrics = fut.result()

                # Aggregate batch metrics
                total_metrics["total_time"] += batch_metrics["batch_time"]
                total_metrics["total_skipped"] += batch_metrics["chunks_skipped"]
                total_metrics["total_failed"] += batch_metrics["chunks_failed"]
                total_metrics["total_empty"] += batch_metrics["empty_outputs"]

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

        total_elapsed = perf_counter() - enrich_start

        # Log aggregate metrics
        self._logger.info(
            f"LLM enrichment complete: {total_metrics['total_chunks']} chunks, "
            f"{total_metrics['total_batches']} batches, "
            f"{total_elapsed:.2f}s wall-time, "
            f"{total_metrics['total_time']:.2f}s batch-time, "
            f"throughput={total_metrics['total_chunks']/total_elapsed:.1f} chunks/sec, "
            f"skipped={total_metrics['total_skipped']}, "
            f"failed={total_metrics['total_failed']}, "
            f"empty={total_metrics['total_empty']}"
        )

        # Fail only if error rate exceeds 5% (tolerating individual chunk failures)
        if total_metrics["total_chunks"] > 0:
            error_rate = total_metrics["total_failed"] / total_metrics["total_chunks"]
            if error_rate > 0.05:
                raise RuntimeError(
                    f"LLM enrichment failed: {total_metrics['total_failed']}/{total_metrics['total_chunks']} chunks failed "
                    f"({error_rate*100:.1f}% error rate exceeds 5% threshold)"
                )
            elif total_metrics["total_failed"] > 0 or total_metrics["total_empty"] > 0:
                self._logger.warning(
                    f"LLM enrichment completed with errors: {total_metrics['total_failed']} failed, "
                    f"{total_metrics['total_empty']} empty (error rate: {error_rate*100:.1f}%, below 5% threshold)"
                )

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

        # CRITICAL: Use actual token count, not word count!
        actual_tokens = _count_tokens(chunk.text)
        if actual_tokens > policy.max_chunk_tokens:
            confidence_note = f"chunk exceeds policy token limit ({actual_tokens} > {policy.max_chunk_tokens})"
            self._logger.warning(
                f"Chunk '{chunk.chunk_id}' exceeds token limit: {actual_tokens} tokens > {policy.max_chunk_tokens} max"
            )
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

    def _extract_qodex_metadata(
        self,
        tenant: Tenant,
        document_id: str,
        chunks: Sequence[ChunkRecord],
        document_metadata: Dict[str, object],
    ) -> List[ChunkRecord]:
        """
        Extract qodex-provided metadata into ChunkRecord fields.

        Qodex-parse (mode="full") provides rich metadata:
        - keywords: 3-5 searchable terms
        - search_terms: Additional retrieval terms
        - semantic_neighbors: Similar chunks with scores
        - topic_label: Topic classification
        - spatial: Bounding boxes and relationships

        This method extracts keywords/search_terms into chunk.key_terms
        and preserves all other qodex metadata in the metadata dict.
        """
        enriched_chunks: List[ChunkRecord] = []
        knowledge_base_id = str(document_metadata.get("knowledge_base_id") or "")

        for chunk in chunks:
            metadata = dict(chunk.metadata)
            metadata.update(document_metadata)

            # Extract keywords from qodex metadata
            qodex_keywords = metadata.get("keywords") or []
            qodex_search_terms = metadata.get("search_terms") or []

            # Merge and deduplicate
            key_terms_set = set()
            if isinstance(qodex_keywords, list):
                key_terms_set.update(str(k).lower().strip() for k in qodex_keywords if k)
            if isinstance(qodex_search_terms, list):
                key_terms_set.update(str(k).lower().strip() for k in qodex_search_terms if k)

            key_terms = sorted(key_terms_set)[:10]  # Top 10 terms

            # Mark enrichment source
            metadata["enrichment_source"] = "qodex"

            enriched = replace(
                chunk,
                key_terms=key_terms,
                metadata=metadata,
            )

            # Update term index
            if key_terms and knowledge_base_id:
                self._term_index.update(tenant, knowledge_base_id, document_id, chunk.chunk_id, key_terms)

            enriched_chunks.append(enriched)

        self._logger.info(
            f"Extracted qodex metadata for {len(enriched_chunks)} chunks "
            f"(avg {sum(len(c.key_terms) for c in enriched_chunks) / max(len(enriched_chunks), 1):.1f} terms/chunk)"
        )

        return enriched_chunks

    def _extract_pages(
        self,
        *,
        document_id: str,
        sections: Sequence[SectionText],
        chunks: List[ChunkRecord],
    ) -> tuple[List[PageRecord], List[ChunkRecord]]:
        """
        Extract page-level records from sections and link to chunks.

        Creates bidirectional chunk↔page pointers for NVIDIA-style context expansion.

        Returns:
            (pages, updated_chunks) - pages with chunk_ids, chunks with page_id in metadata
        """
        # Build page text from sections
        pages_by_number: Dict[int, Dict[str, Any]] = {}

        for section in sections:
            # Get all pages covered by this section
            page_numbers = section.page_numbers if section.page_numbers else list(range(section.start_page, section.end_page + 1))

            for page_num in page_numbers:
                if page_num not in pages_by_number:
                    pages_by_number[page_num] = {
                        "page_number": page_num,
                        "texts": [],
                        "chunk_ids": set(),
                    }
                pages_by_number[page_num]["texts"].append(section.text)

        # Link chunks to pages
        updated_chunks: List[ChunkRecord] = []
        for chunk in chunks:
            # Determine which page(s) this chunk belongs to
            # Use start_page as primary page for single-page assignment
            primary_page = chunk.start_page

            page_id = f"{document_id}_page{primary_page}"

            # Add page_id to chunk metadata
            metadata = dict(chunk.metadata)
            metadata["page_id"] = page_id

            updated_chunk = replace(chunk, metadata=metadata)
            updated_chunks.append(updated_chunk)

            # Add chunk_id to page's chunk_ids
            if primary_page in pages_by_number:
                pages_by_number[primary_page]["chunk_ids"].add(chunk.chunk_id)

        # Create PageRecord instances
        pages: List[PageRecord] = []
        for page_num in sorted(pages_by_number.keys()):
            page_data = pages_by_number[page_num]
            page_id = f"{document_id}_page{page_num}"

            # Combine all section texts for this page
            page_text = "\n\n".join(page_data["texts"])

            page = PageRecord(
                page_id=page_id,
                document_id=document_id,
                page_number=page_num,
                text=page_text,
                chunk_ids=sorted(page_data["chunk_ids"]),
                metadata={},
            )
            pages.append(page)

        self._logger.info(
            f"Extracted {len(pages)} pages with {len(updated_chunks)} chunks "
            f"for document '{document_id}'"
        )

        return pages, updated_chunks


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
    """
    Enterprise-grade chunk enrichment with simplified output: keywords + likely_question.

    Uses gpt-4o-mini for fast, cost-effective enrichment with high-quality enterprise prompts.
    Output focuses on practical retrieval signals rather than complex metadata.
    """
    # Enterprise-level prompt with clear instructions and examples
    system_prompt = """You are an expert metadata enrichment system for a RAG (Retrieval-Augmented Generation) platform.

Your task is to analyze document chunks and extract TWO key pieces of retrieval metadata:

1. **keywords**: 3-5 highly specific, searchable terms that capture the essence of this chunk
   - Focus on domain-specific terminology, concepts, and entities
   - Prefer precise terms over generic ones (e.g., "CCBHC certification criteria" > "criteria")
   - Use lowercase, no stopwords
   - Include acronyms if relevant (e.g., "hra", "ccbhc", "aco")

2. **likely_question**: A single natural-language question a user would ask to find this chunk
   - Write in conversational language as a user would actually ask
   - Be specific enough to retrieve this chunk (not too broad)
   - Examples: "What are the CCBHC staff requirements?", "How do I calculate Medicare reimbursement rates?"

**Output Format:**
Return JSON only, no additional text:
{
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "likely_question": "What is..."
}

**Examples:**

Example 1:
Chunk: "CCBHC certification requires at least one psychiatrist on staff or under contract. The psychiatrist must be available for consultation within 24 hours..."
Output: {
  "keywords": ["ccbhc", "psychiatrist", "certification requirements", "24-hour consultation"],
  "likely_question": "What are the CCBHC psychiatrist staffing requirements?"
}

Example 2:
Chunk: "Medicare Part B reimburses telehealth services at 95% of the in-person rate for rural beneficiaries. Urban telehealth remains at 80% unless..."
Output: {
  "keywords": ["medicare part b", "telehealth reimbursement", "rural beneficiaries", "95% rate"],
  "likely_question": "How much does Medicare reimburse for telehealth in rural areas?"
}

Example 3:
Chunk: "Risk adjustment methodology uses HCC codes to calculate capitation payments. Each HCC has a coefficient that modifies the base rate..."
Output: {
  "keywords": ["hcc codes", "risk adjustment", "capitation payments", "coefficient"],
  "likely_question": "How do HCC codes affect risk adjustment payments?"
}

**Guidelines:**
- Focus on what makes THIS chunk unique and retrievable
- Keywords should help semantic search find relevant content
- The question should match user intent patterns
- If chunk is too vague or fragmented, make your best professional judgment
"""

    # Prepare the chunk with context
    clipped_title = title[:200].strip()
    title_context = f"Document Title: {clipped_title}\n\n" if clipped_title else ""
    chunk_text = text[:3000]  # Limit to 3K chars to fit within context window

    user_message = f"{title_context}Chunk Text:\n{chunk_text}"

    # Call LLM with structured output
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},  # Ensure JSON output
            temperature=0.3,  # Low temperature for consistent extraction
            max_tokens=300,  # Small output (just keywords + question)
        )

        raw_output = response.choices[0].message.content or "{}"

        # Parse JSON response
        data = json.loads(raw_output)

        # Extract keywords and question
        keywords = data.get("keywords") or []
        likely_question = data.get("likely_question") or ""

        # Normalize keywords (lowercase, no stopwords, limit to max_terms)
        if isinstance(keywords, list):
            keywords = [str(k).strip().lower() for k in keywords if k and str(k).strip()]
        else:
            keywords = []

        # Limit to max_terms (default 3-5)
        keywords = keywords[:max_terms] if max_terms else keywords[:5]

        # Store question in extras for future use
        extras = {"likely_question": str(likely_question).strip()} if likely_question else {}

        # Return simplified output (summary=None, key_terms=keywords, extras=question)
        return (
            None,  # No summary needed per user requirements
            keywords,
            None,  # No requires_previous
            None,  # No prev_chunk_id
            None,  # No confidence_note
            extras,  # Contains likely_question
        )

    except Exception as exc:
        # Graceful degradation on failure
        logging.getLogger(__name__).warning(f"LLM enrichment failed: {exc}", exc_info=True)
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
    max_retries: int = 2,
    retry_delay_base: float = 0.5,
    max_concurrent: int | None = None,
    logger=None,
) -> tuple[List[tuple[str | None, List[str], bool | None, str | None, str | None, Dict[str, Any]]], Dict[str, Any]]:
    """
    Batch LLM enrichment with true parallelism, retry logic, and observability.

    Returns:
        (results, metrics) where metrics contains timing, skip counts, failures, etc.
    """
    import logging
    import random
    import time

    if logger is None:
        logger = logging.getLogger(__name__)

    if not texts:
        return [], {"batch_time": 0.0, "chunks_processed": 0, "chunks_skipped": 0, "chunks_failed": 0, "empty_outputs": 0}

    batch_start = perf_counter()

    # Determine concurrency limit (default to batch size for within-batch parallelism)
    if max_concurrent is None:
        max_concurrent = len(texts)

    def _enrich_with_retry(idx: int, text: str, title: str) -> tuple[int, tuple]:
        """Enrich single chunk with retry logic."""
        # Fast-path: skip empty chunks
        if skip_mask and skip_mask[idx]:
            return idx, ("", [], None, None, None, {})

        last_error = None
        for attempt in range(max_retries):
            try:
                result = _llm_enrich(
                    client,
                    model,
                    text,
                    title=title,
                    summary_len=summary_len,
                    max_terms=max_terms,
                )
                return idx, result
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = retry_delay_base * (2 ** attempt) + random.uniform(0, 0.1)
                    logger.warning(
                        f"LLM enrichment attempt {attempt + 1}/{max_retries} failed for chunk {idx}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)

        # All retries exhausted
        logger.error(f"LLM enrichment failed for chunk {idx} after {max_retries} attempts: {last_error}")
        return idx, (None, [], None, None, None, {})

    # Parallel execution within batch
    results_dict: Dict[int, tuple] = {}
    chunks_skipped = sum(1 for skip in (skip_mask or []) if skip)
    chunks_failed = 0
    empty_outputs = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = [
            executor.submit(_enrich_with_retry, idx, text, title)
            for idx, (text, title) in enumerate(zip(texts, titles))
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                idx, result = future.result()
                results_dict[idx] = result

                # Track metrics
                summary, key_terms, _, _, _, _ = result
                if not summary or (not summary.strip() and not key_terms):
                    empty_outputs += 1

            except Exception as e:
                logger.error(f"Unexpected error in batch enrichment: {e}")
                chunks_failed += 1

    # Reconstruct results in original order
    results = [results_dict.get(i, (None, [], None, None, None, {})) for i in range(len(texts))]

    batch_time = perf_counter() - batch_start

    metrics = {
        "batch_time": batch_time,
        "chunks_processed": len(texts),
        "chunks_skipped": chunks_skipped,
        "chunks_failed": chunks_failed,
        "empty_outputs": empty_outputs,
    }

    logger.info(
        f"Batch enrichment complete: {len(texts)} chunks in {batch_time:.2f}s "
        f"({len(texts)/batch_time:.1f} chunks/sec), "
        f"skipped={chunks_skipped}, failed={chunks_failed}, empty={empty_outputs}"
    )

    return results, metrics


__all__ = ["IngestionService", "IngestionResult"]
