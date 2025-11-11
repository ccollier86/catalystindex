from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

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
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._term_index = term_index
        self._audit_logger = audit_logger
        self._metrics = metrics

        self._reranker = reranker

    def retrieve(self, tenant: Tenant, *, query: str, options: SearchOptions | None = None) -> SearchExecution:
        options = options or SearchOptions()
        normalized_mode = options.mode.lower()
        start = perf_counter()
        analysis = self._analyze_query(query)
        expanded_query, alias_terms = self._expand_query(tenant, query, options)
        embedding = next(iter(self._embedding_provider.embed([expanded_query])))
        limit = options.limit or (8 if normalized_mode != "premium" else 24)
        track_options = self._resolve_tracks(options, analysis, default_limit=limit)
        results_by_track: Dict[str, List[RetrievalResult]] = {}
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
                )
            except Exception:
                self._metrics.record_dependency_failure("qdrant")
                raise
            if results:
                results_by_track[track.name] = results

        fused_results = self._fuse_results(results_by_track, limit)
        if normalized_mode == "premium" and self._reranker and fused_results:
            fused_results = list(self._reranker.rerank(expanded_query, fused_results, limit=limit))
        final_results = fused_results[:limit]

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
    ) -> Tuple[TrackOptions, ...]:
        if options.tracks:
            return options.tracks
        tracks: List[TrackOptions] = [TrackOptions(name="text", limit=default_limit)]
        if analysis.requires_vision:
            tracks.append(TrackOptions(name="vision", limit=max(default_limit // 2, 1)))
        return tuple(tracks)

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
        return merged

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


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(l * r for l, r in zip(left, right)))


__all__ = [
    "EmbeddingReranker",
    "SearchDebugDetails",
    "SearchExecution",
    "SearchOptions",
    "SearchService",
    "TrackOptions",
]
