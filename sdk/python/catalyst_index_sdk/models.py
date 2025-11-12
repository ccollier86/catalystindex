from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    section_slug: str
    text: str
    chunk_tier: str
    start_page: int
    end_page: int
    metadata: Dict[str, object]


@dataclass(slots=True)
class ArtifactRef:
    uri: str
    content_type: Optional[str]
    metadata: Dict[str, object]


@dataclass(slots=True)
class IngestionDocument:
    document_id: str
    status: str
    policy: Optional[str]
    chunk_count: int
    parser: Optional[str]
    artifact: Optional[ArtifactRef]
    metadata: Dict[str, object]
    chunks: List[Chunk]
    error: Optional[str] = None


@dataclass(slots=True)
class IngestionJob:
    job_id: str
    status: str
    documents: List[IngestionDocument]
    created_at: str
    updated_at: str
    error: Optional[str] = None


@dataclass(slots=True)
class IngestionJobSummary:
    job_id: str
    status: str
    document_count: int
    created_at: str
    updated_at: str
    error: Optional[str] = None


@dataclass(slots=True)
class SearchResult:
    chunk_id: str
    score: float
    track: str
    chunk_tier: str
    section_slug: str
    start_page: int
    end_page: int
    summary: Optional[str]
    key_terms: List[str]
    requires_previous: bool
    prev_chunk_id: Optional[str]
    confidence_note: Optional[str]
    bbox_pointer: Optional[str]
    metadata: Dict[str, object]
    vision_context: Optional[str]
    explanation: Optional[str]


@dataclass(slots=True)
class SearchDebug:
    raw_query: str
    expanded_query: str
    alias_terms: List[str]
    intent: Optional[str]
    mode: str
    tracks: List[str]


@dataclass(slots=True)
class SearchResultsEnvelope:
    mode: str
    tracks: Dict[str, int]
    results: List[SearchResult]
    debug: Optional[SearchDebug] = None


@dataclass(slots=True)
class GenerationResult:
    summary: str
    citations: Dict[str, float]
    chunk_count: int


@dataclass(slots=True)
class FeedbackReceipt:
    status: str
    positive: bool
    chunk_ids: List[str]
    recorded_at: str
    comment: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)
    analytics: Optional["FeedbackAnalytics"] = None


@dataclass(slots=True)
class FeedbackItemAnalytics:
    identifier: str
    positive: int
    negative: int
    score: int
    last_feedback_at: Optional[str]
    last_comment: Optional[str] = None


@dataclass(slots=True)
class FeedbackAnalytics:
    total_positive: int
    total_negative: int
    feedback_ratio: float
    generated_at: str
    chunks: List[FeedbackItemAnalytics]
    queries: List[FeedbackItemAnalytics]


@dataclass(slots=True)
class LatencySummary:
    count: int
    avg: float
    p50: float
    p95: float
    max: float


@dataclass(slots=True)
class IngestionMetrics:
    chunks: int
    latency_ms: LatencySummary


@dataclass(slots=True)
class SearchMetrics:
    requests: int
    economy_requests: int
    premium_requests: int
    latency_ms: LatencySummary


@dataclass(slots=True)
class GenerationMetrics:
    requests: int
    latency_ms: LatencySummary


@dataclass(slots=True)
class FeedbackMetrics:
    positive: int
    negative: int
    ratio: float
    latency_ms: LatencySummary


@dataclass(slots=True)
class DependencyMetrics:
    failures: Dict[str, int]
    retries: Dict[str, int]


@dataclass(slots=True)
class JobMetrics:
    total: int
    by_status: Dict[str, int]
    failed_documents: int


@dataclass(slots=True)
class TelemetryExporter:
    enabled: bool
    running: bool
    address: Optional[str]
    port: Optional[int]


@dataclass(slots=True)
class TelemetryMetrics:
    ingestion: IngestionMetrics
    jobs: JobMetrics
    search: SearchMetrics
    generation: GenerationMetrics
    feedback: FeedbackMetrics
    dependencies: DependencyMetrics
    exporter: TelemetryExporter


@dataclass(slots=True)
class IngestionJobStatus:
    job_id: str
    status: str
    documents_total: int
    documents_completed: int
    documents_failed: int
    updated_at: str
    error: Optional[str]


__all__ = [
    "ArtifactRef",
    "Chunk",
    "IngestionDocument",
    "IngestionJob",
    "IngestionJobSummary",
    "SearchDebug",
    "SearchResult",
    "SearchResultsEnvelope",
    "GenerationResult",
    "FeedbackReceipt",
    "FeedbackAnalytics",
    "FeedbackItemAnalytics",
    "LatencySummary",
    "IngestionMetrics",
    "SearchMetrics",
    "GenerationMetrics",
    "FeedbackMetrics",
    "DependencyMetrics",
    "JobMetrics",
    "TelemetryExporter",
    "TelemetryMetrics",
    "IngestionJobStatus",
]
