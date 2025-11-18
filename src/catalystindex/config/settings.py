from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import List

from pydantic import BaseModel, Field


class SecuritySettings(BaseModel):
    jwt_secret: str = Field(default="dev-secret", description="HS256 secret for JWT validation")
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    required_scopes: List[str] = Field(default_factory=list, description="Scopes required for the active endpoint")
    api_key: str | None = Field(default=None, description="Static API key for simple auth")
    default_org_id: str = Field(default="org1", description="Default org for API key auth")
    default_workspace_id: str = Field(default="ws1", description="Default workspace for API key auth")
    default_user_id: str = Field(default="user1", description="Default user for API key auth")
    allow_anonymous: bool = Field(default=False, description="Allow anonymous access without auth")


class QdrantSettings(BaseModel):
    enabled: bool = False
    host: str = "localhost"
    port: int = 6333
    grpc_port: int | None = None
    api_key: str | None = None
    collection_prefix: str = "catalystindex"
    prefer_grpc: bool = False
    sparse_vectors: bool = False
    # Bulk/index tuning
    hnsw_m: int | None = None
    hnsw_m_final: int | None = None
    indexing_threshold_kb: int | None = None
    shard_number: int | None = None
    on_disk_vectors: bool | None = None
    hnsw_on_disk: bool | None = None
    defer_indexing: bool = False


class RedisSettings(BaseModel):
    enabled: bool = False
    url: str = "redis://localhost:6379/0"
    ttl_seconds: int = 60 * 60 * 24 * 7


class S3ArtifactSettings(BaseModel):
    bucket: str = Field(default="", description="Bucket name for the artifact store")
    prefix: str | None = Field(default="", description="Key prefix for stored artifacts")
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
    vector_dimension: int = 3072
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
    search_feedback_weight: float = 0.15


class EmbeddingsSettings(BaseModel):
    provider: str = Field(default="openai", description="Embedding provider identifier (hash or openai)")
    model: str | None = Field(default="text-embedding-3-large", description="Model identifier for the provider")
    api_key: str | None = Field(default=None, description="API key for hosted embedding providers")
    base_url: str | None = Field(default=None, description="Override base URL for OpenAI-compatible endpoints")
    dimension: int = Field(default=3072, description="Expected embedding dimension")
    allow_hash_fallback: bool = Field(
        default=True,
        description="Enable dev-lite hash embeddings when a hosted provider is unavailable",
    )


class PolicyAdvisorSettings(BaseModel):
    enabled: bool = Field(default=False, description="Enable LLM-based policy advising")
    provider: str = Field(default="openai", description="LLM provider")
    model: str | None = Field(default="gpt-4o-mini", description="Model used for advising")
    api_key: str | None = Field(default=None, description="API key for the provider")
    base_url: str | None = Field(default=None, description="Optional base URL override")
    sample_chars: int = Field(default=4000, description="How many characters of the document to sample")


class PolicySynthesisSettings(BaseModel):
    enabled: bool = Field(default=False, description="Enable LLM-driven policy synthesis when templates are missing")
    provider: str = Field(default="openai", description="LLM provider")
    model: str | None = Field(default="gpt-4o-mini", description="Model used for synthesis")
    api_key: str | None = Field(default=None, description="API key for the provider")
    base_url: str | None = Field(default=None, description="Optional base URL override")
    sample_chars: int = Field(default=4000, description="How many characters to sample for synthesis")


class RerankerSettings(BaseModel):
    enabled: bool = Field(default=True, description="Enable premium reranking")
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
    max_active_docs: int = Field(default=4, description="Maximum concurrently running documents per worker")
    max_queue_length: int = Field(default=200, description="Maximum queued documents before backpressure")


class JobSettings(BaseModel):
    store: JobStoreSettings = Field(default_factory=JobStoreSettings)
    worker: JobWorkerSettings = Field(default_factory=JobWorkerSettings)


class AppSettings(BaseModel):
    environment: str = "dev"
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    embeddings: EmbeddingsSettings = Field(default_factory=EmbeddingsSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    policy_advisor: PolicyAdvisorSettings = Field(default_factory=PolicyAdvisorSettings)
    policy_synthesis: PolicySynthesisSettings = Field(default_factory=PolicySynthesisSettings)
    telemetry_namespace: str = "catalystindex"
    metrics_exporter_port: int | None = 9464
    metrics_exporter_address: str = "0.0.0.0"
    jobs: JobSettings = Field(default_factory=JobSettings)
    acquisition: AcquisitionSettings = Field(default_factory=AcquisitionSettings)


@lru_cache
def get_settings() -> AppSettings:
    """Return cached application settings."""
    base = AppSettings()
    overrides = _load_env_overrides()
    if not overrides:
        return base
    settings = overrides if isinstance(overrides, AppSettings) else AppSettings(**overrides)
    if isinstance(settings, dict):
        settings = AppSettings(**settings)
    updates = {}
    if isinstance(settings.security, dict):
        updates["security"] = SecuritySettings(**settings.security)
    if isinstance(settings.storage, dict):
        updates["storage"] = StorageSettings(**settings.storage)
    if isinstance(settings.features, dict):
        updates["features"] = FeatureFlags(**settings.features)
    if isinstance(settings.embeddings, dict):
        updates["embeddings"] = EmbeddingsSettings(**settings.embeddings)
    if isinstance(settings.reranker, dict):
        updates["reranker"] = RerankerSettings(**settings.reranker)
    if isinstance(settings.policy_advisor, dict):
        updates["policy_advisor"] = PolicyAdvisorSettings(**settings.policy_advisor)
    if isinstance(settings.policy_synthesis, dict):
        updates["policy_synthesis"] = PolicySynthesisSettings(**settings.policy_synthesis)
    if isinstance(settings.jobs, dict):
        updates["jobs"] = JobSettings(**settings.jobs)
    if isinstance(settings.acquisition, dict):
        updates["acquisition"] = AcquisitionSettings(**settings.acquisition)
    if updates:
        settings = settings.model_copy(update=updates)
    return settings


def _load_env_overrides(prefix: str = "CATALYST_") -> dict:
    overrides: dict = {}
    for key, raw_value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix) :].strip("_")
        if not path:
            continue
        segments = [segment.lower() for segment in path.split("__") if segment]
        if not segments:
            continue
        cursor = overrides
        for segment in segments[:-1]:
            cursor = cursor.setdefault(segment, {})
        cursor[segments[-1]] = _coerce_env_value(raw_value)
    return overrides


def _coerce_env_value(value: str):
    if value == "":
        return None
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if value.isdigit():
        try:
            return int(value)
        except ValueError:
            pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") or value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


__all__ = ["AppSettings", "get_settings"]
