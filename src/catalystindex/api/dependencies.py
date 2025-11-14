from __future__ import annotations

import logging
import os
import sqlite3
from functools import lru_cache
from typing import Any

from fastapi import Depends, Header, HTTPException, status

from ..acquisition.firecrawl import FirecrawlFetcher, HttpURLFetcher
from ..acquisition.service import AcquisitionService, URLFetcher
from ..artifacts.store import ArtifactStore, InMemoryArtifactStore, LocalArtifactStore, S3ArtifactStore
from ..auth.jwt import decode_jwt, ensure_scopes, extract_tenant
from ..config.settings import get_settings
from ..embeddings.base import EmbeddingProvider
from ..embeddings.hash import HashEmbeddingProvider
from ..parsers.registry import default_registry
from ..policies.resolver import resolve_policy
from ..services.feedback import FeedbackService
from ..services.generation import GenerationService
from ..services.ingestion import IngestionService
from ..services.ingestion_jobs import (
    IngestionCoordinator,
    IngestionJobStore,
    IngestionTaskDispatcher,
    RedisPostgresIngestionJobStore,
)
from ..services.knowledge_base import KnowledgeBaseStore
from ..services.policy_advisor import PolicyAdvisor, PolicyAdvice
from ..workers.dispatcher import RQIngestionTaskDispatcher
from ..services.search import CohereReranker, EmbeddingReranker, OpenAIReranker, SearchService
from ..storage.term_index import InMemoryTermIndex, RedisTermIndex, TermIndex
from ..storage.vector_store import InMemoryVectorStore, QdrantVectorStore, VectorStoreClient
from ..telemetry.logger import AuditLogger, MetricsRecorder
from ..chunking.engine import ChunkingEngine

LOGGER = logging.getLogger("catalystindex.dependencies")


@lru_cache
def get_metrics() -> MetricsRecorder:
    settings = get_settings()
    enable_prometheus = settings.features.enable_metrics
    exporter_port = settings.metrics_exporter_port if enable_prometheus else None
    return MetricsRecorder(
        namespace=settings.telemetry_namespace,
        enable_prometheus=enable_prometheus,
        exporter_port=exporter_port,
        exporter_address=settings.metrics_exporter_address,
    )


@lru_cache
def get_audit_logger() -> AuditLogger:
    return AuditLogger()


@lru_cache
def get_vector_store() -> VectorStoreClient:
    settings = get_settings()
    backend = settings.storage.vector_backend.lower()
    qdrant_enabled = backend == "qdrant" or settings.storage.qdrant.enabled
    if qdrant_enabled:
        try:
            from qdrant_client import QdrantClient  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "Qdrant backend requested but 'qdrant-client' is not installed. Install optional dependencies."
            ) from exc
        qdrant_settings = settings.storage.qdrant
        client = QdrantClient(
            host=qdrant_settings.host,
            port=qdrant_settings.port,
            grpc_port=qdrant_settings.grpc_port,
            api_key=qdrant_settings.api_key,
            prefer_grpc=qdrant_settings.prefer_grpc,
        )
        return QdrantVectorStore(
            client,
            collection_prefix=qdrant_settings.collection_prefix,
            vector_size=settings.storage.vector_dimension,
            sparse_enabled=qdrant_settings.sparse_vectors,
            metadata_fields={"environment": settings.environment},
        )
    return InMemoryVectorStore()


@lru_cache
def get_term_index() -> TermIndex:
    settings = get_settings()
    backend = settings.storage.term_index_backend.lower()
    redis_enabled = backend == "redis" or settings.storage.redis.enabled
    if redis_enabled:
        try:
            import redis  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "Redis backend requested but 'redis' package is not installed. Install optional dependencies."
            ) from exc
        redis_settings = settings.storage.redis
        client = redis.Redis.from_url(redis_settings.url)
        return RedisTermIndex(client, ttl_seconds=redis_settings.ttl_seconds)
    if settings.environment.lower() != "dev":
        LOGGER.warning(
            "term_index.in_memory",
            extra={
                "message": "Term index falling back to in-memory implementation. Configure redis to enable persistence.",
            },
        )
    return InMemoryTermIndex()


@lru_cache
def get_embedding_provider() -> HashEmbeddingProvider:
    settings = get_settings()
    embeddings_settings = settings.embeddings
    provider = (embeddings_settings.provider or "openai").lower()
    if provider == "openai":
        builder = _build_openai_embedding_provider(embeddings_settings)
        if builder:
            return builder
        LOGGER.warning(
            "embeddings.fallback",
            extra={"provider": "openai", "detail": "Falling back to hash embeddings"},
        )
    elif provider not in {"hash", "dev_hash"}:
        LOGGER.warning(
            "embeddings.unknown_provider",
            extra={"provider": provider},
        )
    return HashEmbeddingProvider(dimension=embeddings_settings.dimension or settings.storage.vector_dimension)


def _build_openai_embedding_provider(embeddings_settings) -> EmbeddingProvider | None:
    try:
        from ..embeddings.openai import OpenAIEmbeddingProvider
    except ModuleNotFoundError:
        return None
    api_key = embeddings_settings.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    model = embeddings_settings.model or "text-embedding-3-large"
    return OpenAIEmbeddingProvider(api_key=api_key, model=model, base_url=embeddings_settings.base_url)


@lru_cache
def get_chunking_engine() -> ChunkingEngine:
    return ChunkingEngine(namespace="catalystindex")


@lru_cache
def get_parser_registry():
    return default_registry()


@lru_cache
def get_artifact_store() -> ArtifactStore:
    settings = get_settings()
    backend = settings.storage.artifacts.backend.lower()
    if backend == "memory":
        return InMemoryArtifactStore()
    if backend == "s3":
        s3_settings = settings.storage.artifacts.s3
        return S3ArtifactStore(
            bucket=s3_settings.bucket,
            prefix=s3_settings.prefix,
            region_name=s3_settings.region,
            endpoint_url=s3_settings.endpoint_url,
        )
    return LocalArtifactStore(settings.storage.artifacts.base_path)


@lru_cache
def get_url_fetcher() -> URLFetcher:
    settings = get_settings()
    firecrawl_settings = settings.acquisition.firecrawl
    if firecrawl_settings.enabled:
        if not firecrawl_settings.api_key:
            raise RuntimeError("Firecrawl acquisition enabled but no API key configured.")
        return FirecrawlFetcher(
            api_key=firecrawl_settings.api_key,
            base_url=firecrawl_settings.base_url,
            timeout=firecrawl_settings.timeout,
            scrape_format=firecrawl_settings.format,
        )
    if settings.acquisition.use_http_fallback:
        return HttpURLFetcher()
    raise RuntimeError("URL fetching disabled by configuration")


@lru_cache
def get_job_store_connection() -> Any:
    settings = get_settings()
    dsn = settings.jobs.store.postgres_dsn
    if dsn.startswith("sqlite://"):
        path = dsn[len("sqlite://") :]
        database = path.lstrip("/") or ":memory:"
        connection = sqlite3.connect(database, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection
    try:
        import psycopg  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Postgres DSN configured for ingestion jobs but 'psycopg' is not installed. Install optional dependencies.",
        ) from exc
    return psycopg.connect(dsn)


@lru_cache
def get_ingestion_job_store() -> IngestionJobStore:
    settings = get_settings()
    redis_client = None
    redis_url = settings.jobs.store.redis_url
    if redis_url:
        try:
            import redis  # type: ignore
        except ModuleNotFoundError:  # pragma: no cover - optional dependency
            redis_client = None
        else:
            redis_client = redis.Redis.from_url(redis_url)
    elif settings.environment.lower() != "dev":
        LOGGER.warning(
            "jobs.redis_missing",
            extra={"message": "Redis URL not configured for job store; job status caching is disabled."},
        )
    if not settings.jobs.store.postgres_dsn or settings.jobs.store.postgres_dsn.startswith("sqlite://"):
        if settings.environment.lower() != "dev":
            LOGGER.warning(
                "jobs.in_memory_store",
                extra={
                    "message": "Ingestion job store is using SQLite/in-memory; configure Postgres for production.",
                },
            )
    return RedisPostgresIngestionJobStore(
        connection=get_job_store_connection(),
        redis_client=redis_client,
        namespace=settings.jobs.store.redis_namespace,
    )


@lru_cache
def get_knowledge_base_store() -> KnowledgeBaseStore:
    return KnowledgeBaseStore(connection=get_job_store_connection())


@lru_cache
def get_ingestion_task_dispatcher() -> IngestionTaskDispatcher | None:
    settings = get_settings()
    worker_settings = settings.jobs.worker
    if not worker_settings.enabled:
        return None
    if not settings.jobs.store.redis_url:
        raise RuntimeError("Ingestion worker dispatch requires a Redis URL to be configured.")
    try:
        dispatcher: IngestionTaskDispatcher = RQIngestionTaskDispatcher(
            redis_url=settings.jobs.store.redis_url,
            queue_name=worker_settings.queue_name,
            default_timeout=worker_settings.default_timeout,
            max_retries=worker_settings.max_retries,
            retry_intervals=worker_settings.retry_intervals,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "Ingestion worker dispatch enabled but 'rq' is not installed. Install optional dependencies.",
        ) from exc
    return dispatcher


@lru_cache
def get_acquisition_service() -> AcquisitionService:
    return AcquisitionService(url_fetcher=get_url_fetcher())


def get_ingestion_service() -> IngestionService:
    return IngestionService(
        parser_registry=get_parser_registry(),
        chunking_engine=get_chunking_engine(),
        embedding_provider=get_embedding_provider(),
        vector_store=get_vector_store(),
        term_index=get_term_index(),
        audit_logger=get_audit_logger(),
        metrics=get_metrics(),
    )


@lru_cache
def get_ingestion_coordinator() -> IngestionCoordinator:
    settings = get_settings()
    return IngestionCoordinator(
        ingestion_service=get_ingestion_service(),
        acquisition=get_acquisition_service(),
        artifact_store=get_artifact_store(),
        job_store=get_ingestion_job_store(),
        metrics=get_metrics(),
        audit_logger=get_audit_logger(),
        policy_resolver=resolve_policy,
        task_dispatcher=get_ingestion_task_dispatcher(),
        retry_intervals=settings.jobs.worker.retry_intervals,
        policy_advisor=get_policy_advisor(),
        parser_registry=get_parser_registry(),
        knowledge_base_store=get_knowledge_base_store(),
    )


@lru_cache
def get_policy_advisor() -> PolicyAdvisor | None:
    settings = get_settings()
    advisor_settings = settings.policy_advisor
    if not advisor_settings.enabled:
        return None
    api_key = advisor_settings.api_key or os.getenv("CATALYST_POLICY_ADVISOR__api_key") or os.getenv("OPENAI_API_KEY")
    try:
        return PolicyAdvisor(
            enabled=advisor_settings.enabled,
            provider=advisor_settings.provider,
            model=advisor_settings.model or "gpt-4o-mini",
            api_key=api_key,
            base_url=advisor_settings.base_url,
            sample_chars=advisor_settings.sample_chars,
        )
    except RuntimeError as exc:
        LOGGER.warning("policy_advisor.disabled", extra={"error": str(exc)})
        return None


def _build_reranker(settings, embedding_provider):
    reranker_config = settings.reranker
    if not settings.features.enable_premium_rerank:
        return None
    provider = (reranker_config.provider or "").lower()
    if not reranker_config.enabled or provider in ("embedding", "baseline"):
        return EmbeddingReranker(embedding_provider, weight=reranker_config.weight)
    if provider in ("none", "disabled"):
        return None
    if provider == "cohere":
        api_key = reranker_config.api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            LOGGER.warning(
                "reranker.cohere.no_api_key",
                extra={"provider": "cohere", "detail": "Falling back to embedding reranker"},
            )
            return EmbeddingReranker(embedding_provider, weight=reranker_config.weight)
        return CohereReranker(
            api_key=api_key,
            model=reranker_config.model,
            top_n=reranker_config.top_n,
        )
    if provider == "openai":
        api_key = reranker_config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            LOGGER.warning(
                "reranker.openai.no_api_key",
                extra={"provider": "openai", "detail": "Falling back to embedding reranker"},
            )
            return EmbeddingReranker(embedding_provider, weight=reranker_config.weight)
        return OpenAIReranker(
            api_key=api_key,
            model=reranker_config.model,
            base_url=reranker_config.base_url,
            weight=reranker_config.weight,
        )
    raise ValueError(f"Unsupported reranker provider: {reranker_config.provider}")


def get_search_service() -> SearchService:
    settings = get_settings()
    embedding_provider = get_embedding_provider()
    reranker = _build_reranker(settings, embedding_provider)
    return SearchService(
        embedding_provider=embedding_provider,
        vector_store=get_vector_store(),
        term_index=get_term_index(),
        audit_logger=get_audit_logger(),
        metrics=get_metrics(),
        reranker=reranker,
        economy_k=settings.storage.economy_max_k,
        premium_k=settings.storage.premium_max_k,
        enable_sparse_queries=settings.storage.qdrant.sparse_vectors,
        premium_rerank_enabled=settings.features.enable_premium_rerank,
        feedback_weight=settings.features.search_feedback_weight,
    )


def get_generation_service() -> GenerationService:
    return GenerationService(
        search_service=get_search_service(),
        metrics=get_metrics(),
        audit_logger=get_audit_logger(),
    )


@lru_cache
def get_feedback_service() -> FeedbackService:
    return FeedbackService(
        term_index=get_term_index(),
        metrics=get_metrics(),
        audit_logger=get_audit_logger(),
        vector_store=get_vector_store(),
    )


def get_tenant(authorization: str = Header(alias="Authorization")) -> tuple[dict, object]:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header must be Bearer token")
    token = authorization.split(" ", 1)[1]
    claims = decode_jwt(token)
    return claims, extract_tenant(claims)


def require_scopes(*scopes: str):
    def dependency(
        claims_and_tenant = Depends(get_tenant),
    ) -> tuple[dict, object]:
        claims, tenant = claims_and_tenant
        ensure_scopes(claims, scopes)
        return claims, tenant

    return dependency


__all__ = [
    "get_ingestion_service",
    "get_ingestion_coordinator",
    "get_search_service",
    "get_generation_service",
    "get_feedback_service",
    "require_scopes",
]
