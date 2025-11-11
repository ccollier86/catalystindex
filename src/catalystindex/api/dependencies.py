from __future__ import annotations

from functools import lru_cache
from fastapi import Depends, Header, HTTPException, status

from ..auth.jwt import decode_jwt, ensure_scopes, extract_tenant
from ..config.settings import get_settings
from ..embeddings.hash import HashEmbeddingProvider
from ..parsers.registry import default_registry
from ..services.generation import GenerationService
from ..services.ingestion import IngestionService
from ..services.search import SearchService
from ..storage.vector_store import InMemoryVectorStore
from ..telemetry.logger import AuditLogger, MetricsRecorder
from ..chunking.engine import ChunkingEngine


@lru_cache
def get_metrics() -> MetricsRecorder:
    return MetricsRecorder()


@lru_cache
def get_audit_logger() -> AuditLogger:
    return AuditLogger()


@lru_cache
def get_vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


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


def get_ingestion_service() -> IngestionService:
    registry = get_parser_registry()
    parser = registry.resolve("plain_text")
    return IngestionService(
        parser=parser,
        chunking_engine=get_chunking_engine(),
        embedding_provider=get_embedding_provider(),
        vector_store=get_vector_store(),
        audit_logger=get_audit_logger(),
        metrics=get_metrics(),
    )


def get_search_service() -> SearchService:
    return SearchService(
        embedding_provider=get_embedding_provider(),
        vector_store=get_vector_store(),
        audit_logger=get_audit_logger(),
        metrics=get_metrics(),
    )


def get_generation_service() -> GenerationService:
    return GenerationService(
        search_service=get_search_service(),
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
    "get_search_service",
    "get_generation_service",
    "require_scopes",
]
