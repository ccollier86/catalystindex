from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..services.feedback import FeedbackAnalyticsSnapshot, FeedbackItemSnapshot, FeedbackService
from ..services.generation import GenerationService
from ..services.ingestion_jobs import DocumentStatus, DocumentSubmission, IngestionCoordinator
from ..services.search import SearchOptions, SearchService, TrackOptions
from ..models.common import ChunkRecord, RetrievalResult
from ..config.settings import get_settings
from .dependencies import (
    get_feedback_service,
    get_generation_service,
    get_ingestion_coordinator,
    get_metrics,
    get_search_service,
    require_scopes,
)

router = APIRouter()


class IngestRequest(BaseModel):
    document_id: str
    document_title: str
    content: str | None = None
    schema: str | None = None
    parser_hint: str | None = Field(default=None)
    source_type: str = Field(default="inline")
    content_uri: str | None = None
    metadata: dict = Field(default_factory=dict)


class ChunkModel(BaseModel):
    chunk_id: str
    section_slug: str
    text: str
    chunk_tier: str
    start_page: int
    end_page: int
    metadata: dict

    @classmethod
    def from_record(cls, record: ChunkRecord) -> "ChunkModel":
        return cls(
            chunk_id=record.chunk_id,
            section_slug=record.section_slug,
            text=record.text,
            chunk_tier=record.chunk_tier,
            start_page=record.start_page,
            end_page=record.end_page,
            metadata=record.metadata,
        )


class ArtifactModel(BaseModel):
    uri: str
    content_type: str | None = None
    metadata: dict = Field(default_factory=dict)


class DocumentResultModel(BaseModel):
    document_id: str
    status: str
    policy: str | None = None
    chunk_count: int = 0
    parser: str | None = None
    artifact: ArtifactModel | None = None
    metadata: dict = Field(default_factory=dict)
    error: str | None = None
    chunks: List[ChunkModel] | None = None


class IngestResponse(BaseModel):
    job_id: str
    status: str
    document: DocumentResultModel
    created_at: datetime
    updated_at: datetime


class BulkIngestRequest(BaseModel):
    documents: List[IngestRequest]


class BulkIngestResponse(BaseModel):
    job_id: str
    status: str
    submitted: int
    created_at: datetime
    updated_at: datetime
    retry_intervals: List[int] | None = None


class JobSummaryModel(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    document_count: int
    error: str | None = None


class JobDetailModel(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    documents: List[DocumentResultModel]


class AliasOptionsModel(BaseModel):
    enabled: bool = Field(default=True)
    limit: int = Field(default=5)


class TrackConfigModel(BaseModel):
    name: str = Field(default="text")
    limit: int | None = Field(default=None)
    filters: dict | None = Field(default=None)


class SearchRequest(BaseModel):
    query: str
    mode: Literal["economy", "premium"] = "economy"
    limit: int | None = Field(default=None)
    tracks: List[TrackConfigModel] | None = None
    filters: dict | None = None
    alias: AliasOptionsModel | None = None
    debug: bool = Field(default=False)


class RetrievalModel(BaseModel):
    chunk_id: str
    score: float
    track: str
    chunk_tier: str
    section_slug: str
    start_page: int
    end_page: int
    summary: str | None = None
    key_terms: List[str] = Field(default_factory=list)
    requires_previous: bool = False
    prev_chunk_id: str | None = None
    confidence_note: str | None = None
    bbox_pointer: str | None = None
    metadata: dict = Field(default_factory=dict)
    vision_context: str | None = None
    explanation: str | None = None


class TrackSummaryModel(BaseModel):
    name: str
    requested_limit: int | None = None
    retrieved: int = 0


class SearchDebugModel(BaseModel):
    raw_query: str
    expanded_query: str
    alias_terms: List[str]
    intent: str | None = None
    mode: str
    tracks: List[str]


class SearchResponse(BaseModel):
    mode: str
    tracks: List[TrackSummaryModel]
    results: List[RetrievalModel]
    debug: Optional[SearchDebugModel] = None


class GenerationRequest(BaseModel):
    query: str
    limit: int = 6


class GenerationResponseModel(BaseModel):
    summary: str
    citations: dict
    chunk_count: int


class FeedbackRequest(BaseModel):
    query: str
    chunk_ids: List[str]
    positive: bool = Field(default=True)
    comment: str | None = None
    metadata: dict | None = None


class FeedbackItemAnalyticsModel(BaseModel):
    identifier: str
    positive: int
    negative: int
    score: int
    last_feedback_at: datetime | None = None
    last_comment: str | None = None


class FeedbackAnalyticsModel(BaseModel):
    total_positive: int
    total_negative: int
    feedback_ratio: float
    generated_at: datetime
    chunks: List[FeedbackItemAnalyticsModel] = Field(default_factory=list)
    queries: List[FeedbackItemAnalyticsModel] = Field(default_factory=list)


class FeedbackResponseModel(BaseModel):
    status: str
    positive: bool
    chunk_ids: List[str]
    recorded_at: datetime
    comment: str | None = None
    metadata: dict | None = None
    analytics: FeedbackAnalyticsModel


class LatencySummaryModel(BaseModel):
    count: int
    avg: float
    p50: float
    p95: float
    max: float


class IngestionMetricsModel(BaseModel):
    chunks: int
    latency_ms: LatencySummaryModel


class SearchMetricsModel(BaseModel):
    requests: int
    economy_requests: int
    premium_requests: int
    latency_ms: LatencySummaryModel


class GenerationMetricsModel(BaseModel):
    requests: int
    latency_ms: LatencySummaryModel


class FeedbackMetricsModel(BaseModel):
    positive: int
    negative: int
    ratio: float
    latency_ms: LatencySummaryModel


class DependencyMetricsModel(BaseModel):
    failures: Dict[str, int]
    retries: Dict[str, int]


class JobMetricsModel(BaseModel):
    total: int
    by_status: Dict[str, int]
    failed_documents: int


class TelemetryExporterModel(BaseModel):
    enabled: bool
    running: bool
    address: str | None = None
    port: int | None = None


class TelemetryMetricsModel(BaseModel):
    ingestion: IngestionMetricsModel
    jobs: JobMetricsModel
    search: SearchMetricsModel
    generation: GenerationMetricsModel
    feedback: FeedbackMetricsModel
    dependencies: DependencyMetricsModel
    exporter: TelemetryExporterModel


class JobStatusModel(BaseModel):
    job_id: str
    status: str
    documents_total: int
    documents_completed: int
    documents_failed: int
    updated_at: datetime
    error: str | None = None


@router.get("/health")
def health() -> dict:
    settings = get_settings()
    return {"status": "ok", "environment": settings.environment}


@router.post("/ingest/document", response_model=IngestResponse)
def ingest_document(
    request: IngestRequest,
    scopes = Depends(require_scopes("ingest:write")),
    coordinator: IngestionCoordinator = Depends(get_ingestion_coordinator),
):
    _claims, tenant = scopes
    submission = DocumentSubmission(
        document_id=request.document_id,
        document_title=request.document_title,
        schema=request.schema,
        source_type=request.source_type,
        parser_hint=request.parser_hint,
        metadata=request.metadata,
        content=request.content,
        content_uri=request.content_uri,
    )
    job = coordinator.ingest_document(tenant, submission)
    document = job.documents[0]
    return IngestResponse(
        job_id=job.job_id,
        status=job.status.value,
        document=_document_to_model(document),
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.post("/ingest/bulk", response_model=BulkIngestResponse)
def ingest_bulk(
    request: BulkIngestRequest,
    scopes = Depends(require_scopes("ingest:write")),
    coordinator: IngestionCoordinator = Depends(get_ingestion_coordinator),
):
    _claims, tenant = scopes
    submissions = [
        DocumentSubmission(
            document_id=item.document_id,
            document_title=item.document_title,
            schema=item.schema,
            source_type=item.source_type,
            parser_hint=item.parser_hint,
            metadata=item.metadata,
            content=item.content,
            content_uri=item.content_uri,
        )
        for item in request.documents
    ]
    job = coordinator.ingest_bulk(tenant, submissions)
    return BulkIngestResponse(
        job_id=job.job_id,
        status=job.status.value,
        submitted=len(submissions),
        created_at=job.created_at,
        updated_at=job.updated_at,
        retry_intervals=list(coordinator.retry_intervals),
    )


@router.get("/ingest/jobs", response_model=List[JobSummaryModel])
def list_ingestion_jobs(
    scopes = Depends(require_scopes("ingest:read")),
    coordinator: IngestionCoordinator = Depends(get_ingestion_coordinator),
):
    _claims, tenant = scopes
    jobs = coordinator.list_jobs(tenant)
    return [
        JobSummaryModel(
            job_id=job.job_id,
            status=job.status.value,
            created_at=job.created_at,
            updated_at=job.updated_at,
            document_count=len(job.documents),
            error=job.error,
        )
        for job in jobs
    ]


@router.get("/ingest/jobs/{job_id}", response_model=JobDetailModel)
def get_ingestion_job(
    job_id: str,
    scopes = Depends(require_scopes("ingest:read")),
    coordinator: IngestionCoordinator = Depends(get_ingestion_coordinator),
):
    _claims, tenant = scopes
    job = coordinator.get_job(tenant, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    return JobDetailModel(
        job_id=job.job_id,
        status=job.status.value,
        created_at=job.created_at,
        updated_at=job.updated_at,
        error=job.error,
        documents=[_document_to_model(document) for document in job.documents],
    )


@router.get("/ingest/jobs/{job_id}/status", response_model=JobStatusModel)
def get_ingestion_job_status(
    job_id: str,
    scopes = Depends(require_scopes("ingest:read")),
    coordinator: IngestionCoordinator = Depends(get_ingestion_coordinator),
):
    _claims, tenant = scopes
    job = coordinator.get_job(tenant, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    completed = sum(1 for doc in job.documents if doc.status == DocumentStatus.SUCCEEDED)
    failed = sum(1 for doc in job.documents if doc.status == DocumentStatus.FAILED)
    total = len(job.documents)
    status_value = job.status.value if hasattr(job.status, "value") else str(job.status)
    return JobStatusModel(
        job_id=job.job_id,
        status=status_value,
        documents_total=total,
        documents_completed=completed,
        documents_failed=failed,
        updated_at=job.updated_at,
        error=job.error,
    )


@router.post("/search/query", response_model=SearchResponse)
def search_query(
    request: SearchRequest,
    scopes = Depends(require_scopes("search:read")),
    service: SearchService = Depends(get_search_service),
):
    claims, tenant = scopes
    alias_options = request.alias or AliasOptionsModel()
    requested_tracks = tuple(
        TrackOptions(name=track.name, limit=track.limit, filters=track.filters)
        for track in (request.tracks or [])
    )
    options = SearchOptions(
        mode=request.mode.lower(),
        limit=request.limit,
        filters=request.filters,
        tracks=requested_tracks or None,
        alias_limit=alias_options.limit,
        alias_enabled=alias_options.enabled,
        debug=request.debug,
    )
    execution = service.retrieve(tenant, query=request.query, options=options)
    track_counts = Counter(result.track for result in execution.results)
    track_summaries: List[TrackSummaryModel] = []
    if track_counts:
        for track_name, count in sorted(track_counts.items()):
            track_limit = None
            for track in requested_tracks:
                if track.name == track_name:
                    track_limit = track.limit
                    break
            if track_limit is None and request.limit is not None:
                track_limit = request.limit
            track_summaries.append(
                TrackSummaryModel(name=track_name, requested_limit=track_limit, retrieved=count)
            )
    else:
        defaults = requested_tracks or (TrackOptions(name="text", limit=request.limit),)
        for track in defaults:
            track_summaries.append(
                TrackSummaryModel(name=track.name, requested_limit=track.limit, retrieved=0)
            )

    debug_payload = None
    if execution.debug:
        debug_payload = SearchDebugModel(
            raw_query=execution.debug.raw_query,
            expanded_query=execution.debug.expanded_query,
            alias_terms=list(execution.debug.alias_terms),
            intent=execution.debug.intent,
            mode=execution.debug.mode,
            tracks=list(execution.debug.tracks),
        )

    return SearchResponse(
        mode=execution.debug.mode if execution.debug else options.mode,
        tracks=track_summaries,
        results=[
            _retrieval_to_model(result, execution.explanations.get(result.chunk.chunk_id))
            for result in execution.results
        ],
        debug=debug_payload,
    )


@router.post("/generate/summary", response_model=GenerationResponseModel)
def generate_summary(
    request: GenerationRequest,
    scopes = Depends(require_scopes("generate:write")),
    service: GenerationService = Depends(get_generation_service),
):
    claims, tenant = scopes
    response = service.summarize(tenant, query=request.query, limit=request.limit)
    return GenerationResponseModel(
        summary=response.summary,
        citations=response.citations,
        chunk_count=len(response.results),
    )


@router.post("/feedback", response_model=FeedbackResponseModel)
def submit_feedback(
    request: FeedbackRequest,
    scopes = Depends(require_scopes("feedback:write")),
    service: FeedbackService = Depends(get_feedback_service),
):
    _claims, tenant = scopes
    record = service.submit(
        tenant,
        query=request.query,
        chunk_ids=request.chunk_ids,
        positive=request.positive,
        comment=request.comment,
        metadata=request.metadata,
    )
    analytics_snapshot = service.analytics(tenant)
    return FeedbackResponseModel(
        status="recorded",
        positive=record.positive,
        chunk_ids=list(record.chunk_ids),
        recorded_at=record.recorded_at,
        comment=record.comment,
        metadata=record.metadata or None,
        analytics=_analytics_to_model(analytics_snapshot),
    )


@router.get("/feedback/analytics", response_model=FeedbackAnalyticsModel)
def get_feedback_analytics(
    scopes = Depends(require_scopes("feedback:read")),
    service: FeedbackService = Depends(get_feedback_service),
):
    _claims, tenant = scopes
    snapshot = service.analytics(tenant)
    return _analytics_to_model(snapshot)


@router.get("/telemetry/metrics", response_model=TelemetryMetricsModel)
def telemetry_metrics(
    scopes = Depends(require_scopes("telemetry:read")),
    metrics = Depends(get_metrics),
):
    _claims, _tenant = scopes
    snapshot = metrics.snapshot()
    return _telemetry_to_model(snapshot)


__all__ = ["router"]


def _document_to_model(result: object) -> DocumentResultModel:
    if isinstance(result, DocumentResultModel):  # pragma: no cover - defensive
        return result
    status_value = getattr(result, "status", DocumentStatus.QUEUED)
    if isinstance(status_value, DocumentStatus):
        status_str = status_value.value
    else:
        status_str = str(status_value)
    metadata = dict(getattr(result, "metadata", {}) or {})
    artifact_uri = getattr(result, "artifact_uri", None)
    artifact_content_type = getattr(result, "artifact_content_type", None)
    artifact = None
    if artifact_uri:
        artifact_metadata = getattr(result, "artifact_metadata", {}) or {}
        artifact = ArtifactModel(
            uri=artifact_uri,
            content_type=artifact_content_type,
            metadata=dict(artifact_metadata),
        )
    doc_chunks = getattr(result, "chunks", None)
    chunk_models = [ChunkModel.from_record(chunk) for chunk in doc_chunks] if doc_chunks else None
    return DocumentResultModel(
        document_id=getattr(result, "document_id"),
        status=status_str,
        policy=getattr(result, "policy", None),
        chunk_count=getattr(result, "chunk_count", 0),
        parser=getattr(result, "parser", None),
        artifact=artifact,
        metadata=metadata,
        error=getattr(result, "error", None),
        chunks=chunk_models,
    )


def _retrieval_to_model(result: RetrievalResult, explanation: str | None) -> RetrievalModel:
    chunk = result.chunk
    metadata = dict(chunk.metadata)
    return RetrievalModel(
        chunk_id=chunk.chunk_id,
        score=result.score,
        track=result.track,
        chunk_tier=chunk.chunk_tier,
        section_slug=chunk.section_slug,
        start_page=chunk.start_page,
        end_page=chunk.end_page,
        summary=chunk.summary,
        key_terms=list(chunk.key_terms),
        requires_previous=chunk.requires_previous,
        prev_chunk_id=chunk.prev_chunk_id,
        confidence_note=chunk.confidence_note,
        bbox_pointer=chunk.bbox_pointer,
        metadata=metadata,
        vision_context=result.vision_context,
        explanation=explanation,
    )


def _analytics_to_model(snapshot: FeedbackAnalyticsSnapshot) -> FeedbackAnalyticsModel:
    return FeedbackAnalyticsModel(
        total_positive=snapshot.total_positive,
        total_negative=snapshot.total_negative,
        feedback_ratio=snapshot.feedback_ratio,
        generated_at=snapshot.generated_at,
        chunks=[_analytics_item_to_model(item) for item in snapshot.chunks],
        queries=[_analytics_item_to_model(item) for item in snapshot.queries],
    )


def _analytics_item_to_model(item: FeedbackItemSnapshot) -> FeedbackItemAnalyticsModel:
    return FeedbackItemAnalyticsModel(
        identifier=item.identifier,
        positive=item.positive,
        negative=item.negative,
        score=item.score,
        last_feedback_at=item.last_feedback_at,
        last_comment=item.last_comment,
    )


def _latency_summary_to_model(summary: Dict[str, object]) -> LatencySummaryModel:
    return LatencySummaryModel(
        count=int(summary.get("count", 0) or 0),
        avg=float(summary.get("avg", 0.0) or 0.0),
        p50=float(summary.get("p50", 0.0) or 0.0),
        p95=float(summary.get("p95", 0.0) or 0.0),
        max=float(summary.get("max", 0.0) or 0.0),
    )


def _telemetry_to_model(snapshot: Dict[str, object]) -> TelemetryMetricsModel:
    ingestion = snapshot.get("ingestion", {}) or {}
    jobs = snapshot.get("jobs", {}) or {}
    search = snapshot.get("search", {}) or {}
    generation = snapshot.get("generation", {}) or {}
    feedback = snapshot.get("feedback", {}) or {}
    dependencies = snapshot.get("dependencies", {}) or {}
    exporter = snapshot.get("exporter", {}) or {}
    return TelemetryMetricsModel(
        ingestion=IngestionMetricsModel(
            chunks=int(ingestion.get("chunks", 0) or 0),
            latency_ms=_latency_summary_to_model(ingestion.get("latency_ms", {}) or {}),
        ),
        jobs=JobMetricsModel(
            total=int(jobs.get("total", 0) or 0),
            by_status={str(k): int(v) for k, v in (jobs.get("by_status", {}) or {}).items()},
            failed_documents=int(jobs.get("failed_documents", 0) or 0),
        ),
        search=SearchMetricsModel(
            requests=int(search.get("requests", 0) or 0),
            economy_requests=int(search.get("economy_requests", 0) or 0),
            premium_requests=int(search.get("premium_requests", 0) or 0),
            latency_ms=_latency_summary_to_model(search.get("latency_ms", {}) or {}),
        ),
        generation=GenerationMetricsModel(
            requests=int(generation.get("requests", 0) or 0),
            latency_ms=_latency_summary_to_model(generation.get("latency_ms", {}) or {}),
        ),
        feedback=FeedbackMetricsModel(
            positive=int(feedback.get("positive", 0) or 0),
            negative=int(feedback.get("negative", 0) or 0),
            ratio=float(feedback.get("ratio", 0.0) or 0.0),
            latency_ms=_latency_summary_to_model(feedback.get("latency_ms", {}) or {}),
        ),
        dependencies=DependencyMetricsModel(
            failures={k: int(v) for k, v in (dependencies.get("failures", {}) or {}).items()},
            retries={k: int(v) for k, v in (dependencies.get("retries", {}) or {}).items()},
        ),
        exporter=TelemetryExporterModel(
            enabled=bool(exporter.get("enabled", False)),
            running=bool(exporter.get("running", False)),
            address=exporter.get("address"),
            port=exporter.get("port"),
        ),
    )
