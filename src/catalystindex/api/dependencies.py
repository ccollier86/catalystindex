from __future__ import annotations

import sqlite3
from functools import lru_cache
from typing import Any

from fastapi import Depends, Header, HTTPException, status

from ..acquisition.service import AcquisitionService
from ..artifacts.store import ArtifactStore, InMemoryArtifactStore, LocalArtifactStore
from ..auth.jwt import decode_jwt, ensure_scopes, extract_tenant
from ..config.settings import get_settings
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
from ..workers.dispatcher import RQIngestionTaskDispatcher
from ..services.search import EmbeddingReranker, SearchService
from ..storage.term_index import InMemoryTermIndex, RedisTermIndex, TermIndex
from ..storage.vector_store import InMemoryVectorStore, QdrantVectorStore, VectorStoreClient
from ..telemetry.logger import AuditLogger, MetricsRecorder
from ..chunking.engine import ChunkingEngine


@lru_cache
def get_metrics() -> MetricsRecorder:
    return MetricsRecorder()


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
    return InMemoryTermIndex()


@lru_cache
def get_embedding_provider() -> HashEmbeddingProvider:
    settings = get_settings()
    return HashEmbeddingProvider(dimension=settings.storage.vector_dimension)


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
    return LocalArtifactStore(settings.storage.artifacts.base_path)


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
    return RedisPostgresIngestionJobStore(
        connection=get_job_store_connection(),
        redis_client=redis_client,
        namespace=settings.jobs.store.redis_namespace,
    )


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
    return AcquisitionService()


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
    )


def get_search_service() -> SearchService:
    embedding_provider = get_embedding_provider()
    return SearchService(
        embedding_provider=embedding_provider,
        vector_store=get_vector_store(),
        term_index=get_term_index(),
        audit_logger=get_audit_logger(),
        metrics=get_metrics(),
        reranker=EmbeddingReranker(embedding_provider),
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
