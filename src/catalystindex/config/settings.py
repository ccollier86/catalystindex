from __future__ import annotations

from functools import lru_cache
from typing import List

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


class S3ArtifactSettings(BaseModel):
    bucket: str = Field(default="", description="Bucket name for the artifact store")
    prefix: str = Field(default="", description="Key prefix for stored artifacts")
    region: str | None = Field(default=None, description="AWS region for the bucket")
    endpoint_url: str | None = Field(default=None, description="Custom endpoint for S3-compatible storage")


class ArtifactSettings(BaseModel):
    backend: str = Field(default="local", description="Artifact backend (local, s3, or memory)")
    base_path: str = Field(default="artifacts", description="Filesystem path for local artifact storage")
    s3: S3ArtifactSettings = Field(default_factory=S3ArtifactSettings)


class FirecrawlSettings(BaseModel):
    enabled: bool = Field(default=False, description="Enable Firecrawl-backed URL fetching")
    api_key: str = Field(default="", description="Firecrawl API key")
    base_url: str = Field(default="https://api.firecrawl.dev", description="Firecrawl API base URL")
    timeout: int = Field(default=60, description="Request timeout (seconds) for Firecrawl")
    format: str = Field(default="markdown", description="Preferred Firecrawl scrape format")


class AcquisitionSettings(BaseModel):
    use_http_fallback: bool = Field(default=True, description="Use HTTP fetcher when Firecrawl is disabled")
    firecrawl: FirecrawlSettings = Field(default_factory=FirecrawlSettings)


class StorageSettings(BaseModel):
    vector_dimension: int = 128
    premium_max_k: int = 24
    economy_max_k: int = 10
    vector_backend: str = Field(default="memory", description="Vector store backend (memory or qdrant)")
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    term_index_backend: str = Field(default="memory", description="Term index backend (memory or redis)")
    redis: RedisSettings = Field(default_factory=RedisSettings)
    artifacts: ArtifactSettings = Field(default_factory=ArtifactSettings)


class FeatureFlags(BaseModel):
    enable_generation: bool = True
    enable_metrics: bool = True
    enable_premium_rerank: bool = True


class RerankerSettings(BaseModel):
    enabled: bool = Field(default=False, description="Enable external reranker for premium mode")
    provider: str = Field(default="embedding", description="Reranker provider (embedding, cohere, openai)")
    model: str | None = Field(default=None, description="Model identifier for the reranker provider")
    api_key: str | None = Field(default=None, description="API key used by the reranker provider")
    base_url: str | None = Field(default=None, description="Optional base URL override for OpenAI-compatible rerankers")
    top_n: int = Field(default=20, description="Maximum number of documents sent to the reranker")
    weight: float = Field(default=0.3, description="Weight assigned when blending reranker scores")


class JobStoreSettings(BaseModel):
    postgres_dsn: str = Field(default="sqlite:///:memory:", description="DSN for ingestion job store")
    redis_url: str | None = Field(default=None, description="Redis URL for job state caching")
    redis_namespace: str = Field(default="catalystindex", description="Redis key namespace")


class JobWorkerSettings(BaseModel):
    enabled: bool = Field(default=False, description="Enable background worker dispatch")
    queue_name: str = Field(default="ingestion", description="RQ queue name for ingestion tasks")
    default_timeout: int = Field(default=900, description="Default timeout (seconds) for ingestion tasks")
    max_retries: int = Field(default=3, description="Maximum retry attempts for ingestion tasks")
    retry_intervals: List[int] = Field(default_factory=lambda: [15, 30, 60, 120])


class JobSettings(BaseModel):
    store: JobStoreSettings = Field(default_factory=JobStoreSettings)
    worker: JobWorkerSettings = Field(default_factory=JobWorkerSettings)


class AppSettings(BaseModel):
    environment: str = "dev"
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    telemetry_namespace: str = "catalystindex"
    metrics_exporter_port: int | None = 9464
    metrics_exporter_address: str = "0.0.0.0"
    jobs: JobSettings = Field(default_factory=JobSettings)
    acquisition: AcquisitionSettings = Field(default_factory=AcquisitionSettings)


@lru_cache
def get_settings() -> AppSettings:
    """Return cached application settings."""

    return AppSettings()


__all__ = ["AppSettings", "get_settings"]
