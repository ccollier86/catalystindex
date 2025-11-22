"""Hydra structured configuration schemas.

This module defines dataclass-based configuration schemas for Hydra.
These schemas provide type-safe configuration with validation and IDE support.

Usage:
    from hydra import compose, initialize
    from catalystindex.config.schemas import Config

    with initialize(config_path="../../../configs"):
        cfg = compose(config_name="config", overrides=["env=prod"])
        # cfg is type Config with full IDE support
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SecurityConfig:
    """Security configuration."""

    jwt_secret: str = "dev-secret-change-me"


@dataclass
class APIConfig:
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8888
    reload: bool = False
    workers: int = 1


@dataclass
class QdrantConfig:
    """Qdrant vector database configuration."""

    host: str = "qdrant"
    port: int = 6333
    api_key: str = ""
    sparse_vectors: bool = True
    batch_size: int = 100
    timeout: int = 30


@dataclass
class RedisConfig:
    """Redis configuration."""

    url: str = "redis://redis:6379/0"
    max_connections: int = 50


@dataclass
class S3Config:
    """S3 artifact storage configuration."""

    bucket: str = "catalyst-index-uploads"
    prefix: str = "artifacts"
    region: str = "us-east-1"
    endpoint_url: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""


@dataclass
class ArtifactsConfig:
    """Artifact storage configuration."""

    backend: str = "local"
    base_path: str = "/data/artifacts"
    s3: S3Config = field(default_factory=S3Config)


@dataclass
class StorageConfig:
    """Storage configuration."""

    vector_backend: str = "qdrant"
    vector_dimension: int = 3072
    term_index_backend: str = "redis"
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)


@dataclass
class EmbeddingsConfig:
    """Embeddings provider configuration."""

    provider: str = "openai"
    model: str = "text-embedding-3-large"
    api_key: str = ""
    allow_hash_fallback: bool = False
    batch_size: int = 100
    timeout: int = 30
    input_type: Optional[str] = None  # Cohere-specific


@dataclass
class RerankerConfig:
    """Reranker configuration."""

    enabled: bool = True
    provider: str = "openai"
    model: str = "text-embedding-3-large"
    api_key: str = ""
    weight: float = 0.35


@dataclass
class SemanticCacheConfig:
    """Semantic cache configuration."""

    enabled: bool = True
    redis_url: str = "redis://redis:6379/2"
    distance_threshold: float = 0.15
    ttl_seconds: int = 86400


@dataclass
class JobStoreConfig:
    """Job store configuration."""

    postgres_dsn: str = "postgresql://catalyst:catalyst@postgres:5432/catalystjobs"
    redis_url: str = "redis://redis:6379/1"


@dataclass
class WorkerConfig:
    """RQ worker configuration."""

    enabled: bool = True
    queue_name: str = "ingestion"
    default_timeout: int = 900
    max_retries: int = 3
    retry_intervals: List[int] = field(default_factory=lambda: [15, 30, 60, 120])
    max_active_docs: int = 4
    max_queue_length: int = 200
    llm_batch_size: int = 8
    llm_max_workers: int = 6
    llm_max_retries: int = 2
    llm_retry_delay_base: float = 0.5
    llm_max_concurrent_per_batch: int = 8


@dataclass
class JobsConfig:
    """Jobs configuration (store + worker)."""

    store: JobStoreConfig = field(default_factory=JobStoreConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)


@dataclass
class FeaturesConfig:
    """Feature flags configuration."""

    enable_metrics: bool = True
    enable_firecrawl: bool = False


@dataclass
class MetricsConfig:
    """Metrics exporter configuration."""

    exporter_port: int = 9464
    exporter_address: str = "0.0.0.0"


@dataclass
class FirecrawlConfig:
    """Firecrawl web scraping configuration."""

    enabled: bool = False
    api_key: str = ""


@dataclass
class AcquisitionConfig:
    """Acquisition configuration."""

    firecrawl: FirecrawlConfig = field(default_factory=FirecrawlConfig)


@dataclass
class PolicyAdvisorConfig:
    """Policy advisor configuration."""

    enabled: bool = True
    model: str = "gpt-4o-mini"
    api_key: str = ""


@dataclass
class PolicySynthesisConfig:
    """Policy synthesis configuration."""

    enabled: bool = True
    model: str = "gpt-4o-mini"
    api_key: str = ""


@dataclass
class Config:
    """Root Hydra configuration schema.

    This is the top-level configuration that combines all subsystems.
    Hydra will populate this from the composed YAML configs.

    Attributes:
        environment: Current environment (dev/staging/prod)
        security: Security settings
        api: API server settings
        storage: Vector/artifact storage settings
        embeddings: Embeddings provider settings
        reranker: Reranker settings
        semantic_cache: Semantic cache settings
        jobs: Job queue and worker settings
        features: Feature flags
        metrics: Metrics exporter settings
        acquisition: Web acquisition settings
        policy_advisor: Policy advisor settings
        policy_synthesis: Policy synthesis settings
    """

    environment: str = "dev"
    security: SecurityConfig = field(default_factory=SecurityConfig)
    api: APIConfig = field(default_factory=APIConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    semantic_cache: SemanticCacheConfig = field(default_factory=SemanticCacheConfig)
    jobs: JobsConfig = field(default_factory=JobsConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    policy_advisor: PolicyAdvisorConfig = field(default_factory=PolicyAdvisorConfig)
    policy_synthesis: PolicySynthesisConfig = field(default_factory=PolicySynthesisConfig)


__all__ = [
    "Config",
    "SecurityConfig",
    "APIConfig",
    "StorageConfig",
    "QdrantConfig",
    "RedisConfig",
    "S3Config",
    "ArtifactsConfig",
    "EmbeddingsConfig",
    "RerankerConfig",
    "SemanticCacheConfig",
    "JobsConfig",
    "JobStoreConfig",
    "WorkerConfig",
    "FeaturesConfig",
    "MetricsConfig",
    "AcquisitionConfig",
    "FirecrawlConfig",
    "PolicyAdvisorConfig",
    "PolicySynthesisConfig",
]
