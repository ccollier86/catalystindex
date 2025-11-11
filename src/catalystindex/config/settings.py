from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic import BaseModel, Field


class SecuritySettings(BaseModel):
    jwt_secret: str = Field(default="dev-secret", description="HS256 secret for JWT validation")
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    required_scopes: List[str] = Field(default_factory=list, description="Scopes required for the active endpoint")


class QdrantSettings(BaseModel):
    enabled: bool = False
    host: str = "localhost"
    port: int = 6333
    grpc_port: int | None = None
    api_key: str | None = None
    collection_prefix: str = "catalystindex"
    prefer_grpc: bool = False
    sparse_vectors: bool = False


class RedisSettings(BaseModel):
    enabled: bool = False
    url: str = "redis://localhost:6379/0"
    ttl_seconds: int = 60 * 60 * 24 * 7


class StorageSettings(BaseModel):
    vector_dimension: int = 128
    premium_max_k: int = 24
    economy_max_k: int = 10
    vector_backend: str = Field(default="memory", description="Vector store backend (memory or qdrant)")
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    term_index_backend: str = Field(default="memory", description="Term index backend (memory or redis)")
    redis: RedisSettings = Field(default_factory=RedisSettings)


class FeatureFlags(BaseModel):
    enable_generation: bool = True
    enable_metrics: bool = True


class AppSettings(BaseModel):
    environment: str = "dev"
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    telemetry_namespace: str = "catalystindex"


@lru_cache
def get_settings() -> AppSettings:
    """Return cached application settings."""

    return AppSettings()


__all__ = ["AppSettings", "get_settings"]
