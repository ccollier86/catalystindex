from __future__ import annotations

import hashlib
from collections import Counter
from importlib import import_module
from importlib.util import find_spec
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from ..embeddings.base import EmbeddingProvider
from ..models.common import RetrievalResult, Tenant

from ..storage.vector_store import VectorStoreClient
from ..storage.term_index import TermIndex
from ..telemetry.logger import AuditLogger, MetricsRecorder


@dataclass(slots=True)
class SearchOptions:
    mode: str = "economy"
    tracks: Tuple["TrackOptions", ...] | None = None
    filters: Dict[str, object] | None = None
    limit: int | None = None
    alias_limit: int = 5
    alias_enabled: bool = True
    debug: bool = False

    @property
    def economy_mode(self) -> bool:
        return self.mode.lower() != "premium"


@dataclass(slots=True)
class TrackOptions:
    name: str
    limit: int | None = None
    filters: Dict[str, object] | None = None
    use_sparse: bool = False


@dataclass(slots=True)
class QueryAnalysis:
    intent: str = "general"
    requires_vision: bool = False


@dataclass(slots=True)
class SearchDebugDetails:
    raw_query: str
    expanded_query: str
    alias_terms: Tuple[str, ...] = ()
    intent: str | None = None
    mode: str = "economy"
    tracks: Tuple[str, ...] = ()


@dataclass(slots=True)
class SearchExecution:
    results: List[RetrievalResult]
    debug: SearchDebugDetails | None = None
    explanations: Dict[str, str] = field(default_factory=dict)


class SearchService:
    """Hybrid retrieval service with optional economy mode."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStoreClient,
        term_index: TermIndex | None,
        audit_logger: AuditLogger,
        metrics: MetricsRecorder,
        reranker: "Reranker" | None = None,
        economy_k: int = 10,
        premium_k: int = 24,
        enable_sparse_queries: bool = False,
        premium_rerank_enabled: bool = True,
        feedback_weight: float = 0.15,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._term_index = term_index
        self._audit_logger = audit_logger
        self._metrics = metrics

        self._reranker = reranker
        self._economy_k = max(1, economy_k)
        self._premium_k = max(self._economy_k, premium_k)
        self._sparse_queries_enabled = enable_sparse_queries
        self._premium_rerank_enabled = premium_rerank_enabled
        self._feedback_weight = max(0.0, feedback_weight)

    def retrieve(self, tenant: Tenant, *, query: str, options: SearchOptions | None = None) -> SearchExecution:
        options = options or SearchOptions()
        normalized_mode = options.mode.lower()
        start = perf_counter()
        analysis = self._analyze_query(query)
        expanded_query, alias_terms = self._expand_query(tenant, query, options)
        embedding = next(iter(self._embedding_provider.embed([expanded_query])))
        limit = options.limit or (self._economy_k if normalized_mode != "premium" else self._premium_k)
        track_options = self._resolve_tracks(options, analysis, default_limit=limit, mode=normalized_mode)
        results_by_track: Dict[str, List[RetrievalResult]] = {}
        sparse_query = None
        if self._sparse_queries_enabled and normalized_mode == "premium":
            sparse_query = self._build_sparse_query(expanded_query)
        for track in track_options:
            track_limit = track.limit or limit
            filters = self._merge_filters(options.filters, track.filters)
            try:
                results = self._vector_store.query(
                    tenant,
                    embedding,
                    track=track.name,
                    limit=track_limit,
                    filters=filters,
                    sparse_vector=sparse_query if getattr(track, "use_sparse", False) else None,
                )
            except Exception:
                self._metrics.record_dependency_failure("qdrant")
                raise
            if results:
                results_by_track[track.name] = results

        fused_results = self._fuse_results(results_by_track, limit)
        if (
            normalized_mode == "premium"
            and self._reranker
            and self._premium_rerank_enabled
            and fused_results
        ):
            fused_results = list(self._reranker.rerank(expanded_query, fused_results, limit=limit))
        final_results = self._apply_feedback_boost(fused_results[:limit])

        explanations = {
            result.chunk.chunk_id: self._build_explanation(result, index)
            for index, result in enumerate(final_results, start=1)
        }

        debug: SearchDebugDetails | None = None
        if options.debug:
            debug = SearchDebugDetails(
                raw_query=query,
                expanded_query=expanded_query,
                alias_terms=tuple(alias_terms),
                intent=analysis.intent,
                mode=normalized_mode,
                tracks=tuple(results_by_track.keys()) or tuple(track.name for track in track_options),
            )

        duration_ms = (perf_counter() - start) * 1000.0
        self._metrics.record_search(
            len(final_results),
            economy=(normalized_mode != "premium"),
            latency_ms=duration_ms,
        )
        self._audit_logger.search_executed(tenant, query=expanded_query, result_count=len(final_results))
        return SearchExecution(results=final_results, debug=debug, explanations=explanations)

    # Internal helpers -------------------------------------------------
    def _expand_query(self, tenant: Tenant, query: str, options: SearchOptions) -> Tuple[str, Tuple[str, ...]]:
        if not self._term_index or not options.alias_enabled:
            return query, ()
        aliases = self._term_index.expand_query(tenant, query, limit=options.alias_limit)
        unique_aliases = tuple(dict.fromkeys(aliases)) if aliases else ()
        if not unique_aliases:
            return query, ()
        alias_text = " ".join(unique_aliases)
        return f"{query} {alias_text}".strip(), unique_aliases

    def _resolve_tracks(
        self,
        options: SearchOptions,
        analysis: QueryAnalysis,
        *,
        default_limit: int,
        mode: str,
    ) -> Tuple[TrackOptions, ...]:
        if options.tracks:
            return options.tracks
        text_filters = self._filters_for_intent(analysis, track="text")
        tracks: List[TrackOptions] = [
            TrackOptions(
                name="text",
                limit=default_limit,
                filters=text_filters,
                use_sparse=self._sparse_queries_enabled and mode == "premium",
            )
        ]
        if analysis.requires_vision:
            vision_filters = self._filters_for_intent(analysis, track="vision")
            tracks.append(
                TrackOptions(
                    name="vision",
                    limit=max(default_limit // 2, 1),
                    filters=vision_filters,
                    use_sparse=False,
                )
            )
        return tuple(tracks)

    def _filters_for_intent(self, analysis: QueryAnalysis, *, track: str) -> Dict[str, object] | None:
        tiers = self._chunk_tiers_for_intent(analysis.intent, track=track)
        filters: Dict[str, object] = {}
        if tiers:
            filters["chunk_tier"] = tiers
        if track == "text":
            policy = self._policy_for_intent(analysis.intent)
            if policy:
                filters["policy"] = policy
        return filters or None

    def _chunk_tiers_for_intent(self, intent: str, *, track: str) -> Tuple[str, ...] | None:
        if track == "vision":
            return ("vision",)
        mapping: Dict[str, Tuple[str, ...]] = {
            "diagnosis_lookup": ("criteria", "semantic"),
            "treatment_planning": ("semantic", "window"),
            "vision_required": ("semantic", "window", "criteria"),
            "general": ("semantic", "window", "criteria"),
        }
        return mapping.get(intent, mapping["general"])

    def _policy_for_intent(self, intent: str) -> Tuple[str, ...] | None:
        mapping: Dict[str, Tuple[str, ...]] = {
            "diagnosis_lookup": ("dsm5",),
            "treatment_planning": ("treatment_planner",),
        }
        return mapping.get(intent)

    def _analyze_query(self, query: str) -> QueryAnalysis:
        lowered = query.lower()
        vision_markers = ("see figure", "see table", "diagram", "image", "figure", "table")
        requires_vision = any(marker in lowered for marker in vision_markers)
        if "diagnosis" in lowered or "criteria" in lowered:
            intent = "diagnosis_lookup"
        elif "treatment" in lowered or "plan" in lowered:
            intent = "treatment_planning"
        elif requires_vision:
            intent = "vision_required"
        else:
            intent = "general"
        return QueryAnalysis(intent=intent, requires_vision=requires_vision)

    def _merge_filters(
        self,
        base: Dict[str, object] | None,
        specific: Dict[str, object] | None,
    ) -> Dict[str, object] | None:
        if not base and not specific:
            return None
        merged: Dict[str, object] = {}
        if base:
            merged.update(base)
        if specific:
            merged.update(specific)
        sanitized = {key: value for key, value in merged.items() if value is not None}
        return sanitized or None

    def _build_sparse_query(self, query: str) -> Dict[int, float] | None:
        tokens = [token for token in query.lower().split() if token]
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

    def _fuse_results(self, results: Mapping[str, Sequence[RetrievalResult]], limit: int) -> List[RetrievalResult]:
        if not results:
            return []
        if len(results) == 1:
            return list(next(iter(results.values())))
        rrf_constant = 60
        fused_scores: Dict[str, MutableMapping[str, object]] = {}
        for track_name, track_results in results.items():
            for rank, result in enumerate(track_results, start=1):
                chunk_id = result.chunk.chunk_id
                entry = fused_scores.setdefault(
                    chunk_id,
                    {"result": result, "best_score": result.score, "rrf": 0.0, "track": track_name},
                )
                entry["rrf"] += 1.0 / (rrf_constant + rank)
                if result.score > entry["best_score"]:
                    entry["result"] = result
                    entry["best_score"] = result.score
                    entry["track"] = track_name
        scored: List[Tuple[float, RetrievalResult]] = []
        for entry in fused_scores.values():
            result = entry["result"]
            fused_score = float(result.score) + float(entry["rrf"])
            scored.append((fused_score, RetrievalResult(chunk=result.chunk, score=fused_score, track=result.track, vision_context=result.vision_context)))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored][:limit]

    def _apply_feedback_boost(self, results: Sequence[RetrievalResult]) -> List[RetrievalResult]:
        if not self._feedback_weight:
            return list(results)
        boosted: List[RetrievalResult] = []
        for result in results:
            feedback_score = float(result.chunk.metadata.get("feedback_score", 0.0) or 0.0)
            adjusted = float(result.score) + self._feedback_weight * feedback_score
            boosted.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=adjusted,
                    track=result.track,
                    vision_context=result.vision_context,
                )
            )
        return boosted

    def _build_explanation(self, result: RetrievalResult, index: int) -> str:
        chunk = result.chunk
        section = chunk.section_slug or "section"
        return (
            f"Rank {index} from {result.track} track, tier={chunk.chunk_tier}, score={result.score:.3f}."
        )


class Reranker:
    """Protocol for reranker implementations."""

    def rerank(self, query: str, results: Sequence[RetrievalResult], *, limit: int) -> Iterable[RetrievalResult]:
        raise NotImplementedError


class EmbeddingReranker(Reranker):
    """Lightweight reranker using embedding similarity as a heuristic."""

    def __init__(self, embedding_provider: EmbeddingProvider, weight: float = 0.3) -> None:
        self._embedding_provider = embedding_provider
        self._weight = weight

    def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        *,
        limit: int,
    ) -> Iterable[RetrievalResult]:
        if not results:
            return []
        query_vector = next(iter(self._embedding_provider.embed([query])))
        chunk_texts = [result.chunk.summary or result.chunk.text for result in results]
        chunk_vectors = list(self._embedding_provider.embed(chunk_texts))
        rescored: List[Tuple[float, RetrievalResult]] = []
        for result, chunk_vector in zip(results, chunk_vectors):
            rerank_score = _dot(query_vector, chunk_vector)
            combined = (1 - self._weight) * float(result.score) + self._weight * rerank_score
            rescored.append((combined, RetrievalResult(chunk=result.chunk, score=combined, track=result.track, vision_context=result.vision_context)))
        rescored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in rescored][:limit]


class CohereReranker(Reranker):
    """External reranker backed by Cohere's ReRank API."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str | None = None,
        top_n: int = 20,
        client: Any | None = None,
    ) -> None:
        if client is None:
            if not api_key:
                raise ValueError("Cohere reranker requires an API key when client is not provided")
            if find_spec("cohere") is None:
                raise RuntimeError(
                    "Cohere reranker requires the 'cohere' package. Install optional dependencies to enable it."
                )
            cohere_module = import_module("cohere")
            client = cohere_module.Client(api_key)  # type: ignore[attr-defined]
        self._client = client
        self._model = model or "rerank-english-v3.0"
        self._top_n = max(1, top_n)

    def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        *,
        limit: int,
    ) -> Iterable[RetrievalResult]:
        if not results:
            return []
        documents = [result.chunk.summary or result.chunk.text for result in results]
        response = self._client.rerank(  # type: ignore[attr-defined]
            query=query,
            documents=documents,
            top_n=min(self._top_n, max(1, limit), len(documents)),
            model=self._model,
        )
        reranked: List[RetrievalResult] = []
        seen: set[int] = set()
        for item in getattr(response, "results", []):
            index = getattr(item, "index", None)
            if index is None or index >= len(results):
                continue
            seen.add(index)
            base = results[index]
            score = float(getattr(item, "relevance_score", base.score))
            reranked.append(
                RetrievalResult(chunk=base.chunk, score=score, track=base.track, vision_context=base.vision_context)
            )
            if len(reranked) >= limit:
                break
        if len(reranked) < limit:
            for idx, base in enumerate(results):
                if idx in seen:
                    continue
                reranked.append(base)
                if len(reranked) >= limit:
                    break
        return reranked[:limit]


class OpenAIReranker(Reranker):
    """External reranker that uses OpenAI embeddings for similarity scoring."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str | None = None,
        base_url: str | None = None,
        weight: float = 0.5,
        client: Any | None = None,
    ) -> None:
        if client is None:
            if not api_key:
                raise ValueError("OpenAI reranker requires an API key when client is not provided")
            if find_spec("openai") is None:
                raise RuntimeError(
                    "OpenAI reranker requires the 'openai' package. Install optional dependencies to enable it."
                )
            openai_module = import_module("openai")
            client = openai_module.OpenAI(api_key=api_key, base_url=base_url)  # type: ignore[attr-defined,call-arg]
        self._client = client
        self._model = model or "text-embedding-3-large"
        self._weight = weight

    def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        *,
        limit: int,
    ) -> Iterable[RetrievalResult]:
        if not results:
            return []
        documents = [result.chunk.summary or result.chunk.text for result in results]
        query_embedding_response = self._client.embeddings.create(  # type: ignore[attr-defined]
            model=self._model,
            input=[query],
        )
        query_vector = query_embedding_response.data[0].embedding  # type: ignore[index]
        document_embeddings_response = self._client.embeddings.create(  # type: ignore[attr-defined]
            model=self._model,
            input=documents,
        )
        document_vectors = [record.embedding for record in document_embeddings_response.data]  # type: ignore[attr-defined]
        rescored: List[Tuple[float, RetrievalResult]] = []
        for index, base in enumerate(results):
            if index < len(document_vectors):
                doc_vector = document_vectors[index]
                similarity = _dot(query_vector, doc_vector)
                combined = (1 - self._weight) * float(base.score) + self._weight * similarity
            else:
                combined = float(base.score)
            rescored.append(
                (
                    combined,
                    RetrievalResult(
                        chunk=base.chunk,
                        score=combined,
                        track=base.track,
                        vision_context=base.vision_context,
                    ),
                )
            )
        rescored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in rescored][:limit]

def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(l * r for l, r in zip(left, right)))


__all__ = [
    "CohereReranker",
    "EmbeddingReranker",
    "OpenAIReranker",
    "SearchDebugDetails",
    "SearchExecution",
    "SearchOptions",
    "SearchService",
    "TrackOptions",
]
