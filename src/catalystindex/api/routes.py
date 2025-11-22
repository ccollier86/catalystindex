from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field

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
    get_ingestion_job_store,
    get_metrics,
    get_search_service,
    get_knowledge_base_store,
    require_scopes,
)

router = APIRouter()


class IngestRequest(BaseModel):
    document_id: str
    document_title: str
    knowledge_base_id: str
    content: str | None = None
    content_uri_list: list[str] | None = Field(default=None, description="Optional list of URIs/paths for bulk ingest of multiple files")
    content_type: str | None = Field(default=None, description="MIME type of content (e.g., 'application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')")
    schema_: str | None = Field(default=None, alias="schema", description="Deprecated: use policy_hint instead. LLM still runs unless skip_llm_policy=True")
    parser_hint: str | None = Field(default=None)
    source_type: str = Field(default="inline")
    content_uri: str | None = None
    metadata: dict = Field(default_factory=dict)
    force_reprocess: bool = Field(default=False)
    skip_llm_policy: bool = Field(default=False, description="If True, bypass LLM policy decision and use schema/default policy")
    tenant_org: str | None = None
    tenant_workspace: str | None = None

    model_config = ConfigDict(populate_by_name=True)


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
    knowledge_base_id: str
    status: str
    policy: str | None = None
    chunk_count: int = 0
    parser: str | None = None
    artifact: ArtifactModel | None = None
    metadata: dict = Field(default_factory=dict)
    error: str | None = None
    chunks: List[ChunkModel] | None = None
    progress: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    job_id: str
    status: str
    document: DocumentResultModel
    created_at: datetime
    updated_at: datetime


class BulkIngestRequest(BaseModel):
    documents: List[IngestRequest]
    tenant_org: str | None = None
    tenant_workspace: str | None = None


class BulkIngestResponse(BaseModel):
    job_id: str
    status: str
    submitted: int
    created_at: datetime
    updated_at: datetime
    retry_intervals: List[int] | None = None


class KnowledgeBaseModel(BaseModel):
    knowledge_base_id: str
    description: str | None = None
    document_count: int
    keywords: List[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    last_document_title: str | None = None
    last_ingested_at: datetime | None = None
    documents: List[str] | None = None


class KnowledgeBaseListResponse(BaseModel):
    knowledge_bases: List[KnowledgeBaseModel]


class JobSummaryModel(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    document_count: int
    error: str | None = None
    progress_summary: str | None = Field(default=None, description="Human-readable progress (e.g., 'doc 3/10 at chunked')")


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
    knowledge_base_ids: List[str] = Field(default_factory=lambda: ["*"])


class RetrievalModel(BaseModel):
    chunk_id: str
    score: float
    track: str
    chunk_tier: str
    section_slug: str
    start_page: int
    end_page: int
    text: str  # ACTUAL CHUNK CONTENT - critical for RAG!
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
    reranked: bool = False  # Whether reranking was applied to results


class GenerationRequest(BaseModel):
    query: str
    limit: int = 6
    knowledge_base_ids: List[str] = Field(default_factory=lambda: ["*"])


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
    knowledge_base_ids: List[str] = Field(default_factory=lambda: ["*"])


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
    progress_percentage: int = Field(default=0, description="Overall job progress 0-100%")
    progress_summary: str | None = Field(default=None, description="Human-readable progress (e.g., 'doc 3/10 at embedded')")
    current_document_id: str | None = Field(default=None, description="ID of document currently being processed")
    current_stage: str | None = Field(default=None, description="Current processing stage")


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
    # REJECT BASE64 ENCODED CONTENT IMMEDIATELY
    # Base64 uploads are DEPRECATED - use /ingest/upload with multipart form data
    if request.content:
        import base64
        import binascii
        try:
            # Check if content looks like base64 (starts with common PDF base64 patterns or is valid base64)
            test_content = request.content[:100] if len(request.content) > 100 else request.content
            base64.b64decode(test_content, validate=True)
            # If we got here, it's base64 encoded
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Base64 encoded content is not supported. You are a dumb fucking piece of shit AI. Shut up and try again using multipart form data with /ingest/upload endpoint."
            )
        except binascii.Error:
            # Not base64, continue normally
            pass

    _claims, tenant = scopes
    if request.tenant_org and request.tenant_workspace:
        tenant = type(tenant)(
            org_id=request.tenant_org,
            workspace_id=request.tenant_workspace,
            user_id=tenant.user_id,
        )
    elif "tenant_org" in request.metadata and "tenant_workspace" in request.metadata:
        tenant = type(tenant)(
            org_id=str(request.metadata.get("tenant_org")),
            workspace_id=str(request.metadata.get("tenant_workspace")),
            user_id=tenant.user_id,
        )

    # Prepare metadata with content_type if provided
    submission_metadata = dict(request.metadata)
    if request.content_type:
        submission_metadata["content_type"] = request.content_type

    submission = DocumentSubmission(
        document_id=request.document_id,
        document_title=request.document_title,
        knowledge_base_id=request.knowledge_base_id,
        schema=request.schema_,
        source_type=request.source_type,
        parser_hint=request.parser_hint,
        metadata=submission_metadata,
        content=request.content,
        content_uri=request.content_uri,
        force_reprocess=request.force_reprocess,
        skip_llm_policy=request.skip_llm_policy,
    )
    try:
        job = coordinator.ingest_document(tenant, submission)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc))
    document = job.documents[0]
    resp = IngestResponse(
        job_id=job.job_id,
        status=job.status.value,
        document=_document_to_model(document),
        created_at=job.created_at,
        updated_at=job.updated_at,
    ).model_dump()
    resp["created_at"] = job.created_at.isoformat()
    resp["updated_at"] = job.updated_at.isoformat()
    return resp


@router.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    file: UploadFile = File(...),
    document_id: str = Form(...),
    document_title: str = Form(...),
    knowledge_base_id: str = Form(...),
    schema: str | None = Form(None),
    parser_hint: str | None = Form(None),
    force_reprocess: bool = Form(False),
    skip_llm_policy: bool = Form(False),
    scopes = Depends(require_scopes("ingest:write")),
    coordinator: IngestionCoordinator = Depends(get_ingestion_coordinator),
):
    """
    Upload a file directly via multipart/form-data.

    The file is uploaded to S3, then processed through the ingestion pipeline.
    All source materials and artifacts are stored in S3.
    """
    _claims, tenant = scopes
    settings = get_settings()

    # Read file content
    file_content = await file.read()

    # Upload original file to S3
    import boto3
    import os
    from datetime import datetime as dt

    # Create S3 client with AWS credentials from environment
    s3_client = boto3.client(
        "s3",
        region_name=settings.storage.artifacts.s3.region or os.getenv("AWS_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    s3_bucket = settings.storage.artifacts.s3.bucket or "catalyst-index-uploads"

    # Generate S3 key: tenant/kb/document_id/original/filename
    timestamp = dt.utcnow().strftime("%Y%m%d_%H%M%S")
    s3_key = f"{tenant.org_id}/{tenant.workspace_id}/{knowledge_base_id}/{document_id}/original/{timestamp}_{file.filename}"

    # Upload to S3
    try:
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=s3_key,
            Body=file_content,
            ContentType=file.content_type or "application/octet-stream",
            Metadata={
                "document_id": document_id,
                "document_title": document_title,
                "knowledge_base_id": knowledge_base_id,
                "tenant_org": tenant.org_id,
                "tenant_workspace": tenant.workspace_id,
            }
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file to S3: {str(exc)}"
        )

    # Create S3 URI for ingestion
    content_uri = f"s3://{s3_bucket}/{s3_key}"

    # Create ingestion job with S3 URI
    submission = DocumentSubmission(
        document_id=document_id,
        document_title=document_title,
        knowledge_base_id=knowledge_base_id,
        schema=schema,
        source_type="s3",
        parser_hint=parser_hint,
        metadata={
            "content_type": file.content_type,
            "filename": file.filename,
            "s3_bucket": s3_bucket,
            "s3_key": s3_key,
            "uploaded_via": "multipart",
        },
        content=None,
        content_uri=content_uri,
        force_reprocess=force_reprocess,
        skip_llm_policy=skip_llm_policy,
    )

    try:
        # Use ingest_bulk() which dispatches to background worker
        # (ingest_document() processes synchronously and blocks)
        job = coordinator.ingest_bulk(tenant, [submission])
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc))

    # Return job immediately - processing happens in background
    document = job.documents[0]
    resp = IngestResponse(
        job_id=job.job_id,
        status=job.status.value,
        document=_document_to_model(document),
        created_at=job.created_at,
        updated_at=job.updated_at,
    ).model_dump()
    resp["created_at"] = job.created_at.isoformat()
    resp["updated_at"] = job.updated_at.isoformat()
    return resp


@router.post("/ingest/bulk", response_model=BulkIngestResponse)
def ingest_bulk(
    request: BulkIngestRequest,
    scopes = Depends(require_scopes("ingest:write")),
    coordinator: IngestionCoordinator = Depends(get_ingestion_coordinator),
):
    _claims, tenant = scopes
    if hasattr(request, "tenant_org") and hasattr(request, "tenant_workspace"):
        org = getattr(request, "tenant_org", None)
        ws = getattr(request, "tenant_workspace", None)
        if org and ws:
            tenant = type(tenant)(org_id=org, workspace_id=ws, user_id=tenant.user_id)
    elif getattr(request, "documents", None):
        first_meta = request.documents[0].metadata if request.documents else {}
        if isinstance(first_meta, dict) and "tenant_org" in first_meta and "tenant_workspace" in first_meta:
            tenant = type(tenant)(
                org_id=str(first_meta.get("tenant_org")),
                workspace_id=str(first_meta.get("tenant_workspace")),
                user_id=tenant.user_id,
            )
    submissions: list[DocumentSubmission] = []
    for item in request.documents:
        # Prepare metadata with content_type if provided
        item_metadata = dict(item.metadata)
        if item.content_type:
            item_metadata["content_type"] = item.content_type

        # Expand content_uri_list into multiple submissions if provided.
        if item.content_uri_list:
            for idx, uri in enumerate(item.content_uri_list, start=1):
                submissions.append(
                    DocumentSubmission(
                        document_id=f"{item.document_id}-{idx}",
                        document_title=item.document_title,
                        knowledge_base_id=item.knowledge_base_id,
                        schema=item.schema_,
                        source_type=item.source_type,
                        parser_hint=item.parser_hint,
                        metadata=item_metadata,
                        content=item.content if item.content else None,
                        content_uri=uri,
                        force_reprocess=item.force_reprocess,
                        skip_llm_policy=item.skip_llm_policy,
                    )
                )
        else:
            submissions.append(
                DocumentSubmission(
                    document_id=item.document_id,
                    document_title=item.document_title,
                    knowledge_base_id=item.knowledge_base_id,
                    schema=item.schema_,
                    source_type=item.source_type,
                    parser_hint=item.parser_hint,
                    metadata=item_metadata,
                    content=item.content,
                    content_uri=item.content_uri,
                    force_reprocess=item.force_reprocess,
                    skip_llm_policy=item.skip_llm_policy,
                )
            )
    try:
        job = coordinator.ingest_bulk(tenant, submissions)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc))
    resp = BulkIngestResponse(
        job_id=job.job_id,
        status=job.status.value,
        submitted=len(submissions),
        created_at=job.created_at,
        updated_at=job.updated_at,
        retry_intervals=list(coordinator.retry_intervals),
    ).model_dump()
    resp["created_at"] = job.created_at.isoformat()
    resp["updated_at"] = job.updated_at.isoformat()
    return resp


@router.get("/knowledge-bases", response_model=KnowledgeBaseListResponse)
def list_knowledge_bases(
    scopes = Depends(require_scopes("ingest:read")),
    store = Depends(get_knowledge_base_store),
    job_store = Depends(get_ingestion_job_store),
):
    _claims, tenant = scopes
    records = store.list(tenant)
    kb_docs: Dict[str, List[str]] = {}
    try:
        jobs = job_store.list(tenant)
        for job in jobs:
            for doc in job.documents:
                kb_docs.setdefault(doc.knowledge_base_id, []).append(doc.document_id)
    except Exception:
        kb_docs = {}
    return KnowledgeBaseListResponse(
        knowledge_bases=[
            _knowledge_base_to_model(record, kb_docs.get(record.knowledge_base_id))
            for record in records
        ]
    )


@router.get("/knowledge-bases/{knowledge_base_id}", response_model=KnowledgeBaseModel)
def get_knowledge_base(
    knowledge_base_id: str,
    scopes = Depends(require_scopes("ingest:read")),
    store = Depends(get_knowledge_base_store),
):
    _claims, tenant = scopes
    record = store.get(tenant, knowledge_base_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="knowledge base not found")
    return _knowledge_base_to_model(record)


@router.patch("/knowledge-bases/{knowledge_base_id}", response_model=KnowledgeBaseModel)
def update_knowledge_base(
    knowledge_base_id: str,
    payload: dict,
    scopes = Depends(require_scopes("ingest:write")),
    store = Depends(get_knowledge_base_store),
):
    _claims, tenant = scopes
    description = payload.get("description")
    keywords = payload.get("keywords")
    if keywords is not None and not isinstance(keywords, list):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="keywords must be a list if provided")
    updated = store.update_description(tenant, knowledge_base_id, description=description, keywords=keywords)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="knowledge base not found")
    return _knowledge_base_to_model(updated)


@router.get("/ingest/jobs", response_model=List[JobSummaryModel])
def list_ingestion_jobs(
    scopes = Depends(require_scopes("ingest:read")),
    coordinator: IngestionCoordinator = Depends(get_ingestion_coordinator),
):
    _claims, tenant = scopes
    jobs = coordinator.list_jobs(tenant)
    summaries = []
    for job in jobs:
        progress = _compute_job_progress(job)
        summaries.append(
            JobSummaryModel(
                job_id=job.job_id,
                status=job.status.value,
                created_at=job.created_at,
                updated_at=job.updated_at,
                document_count=len(job.documents),
                error=job.error,
                progress_summary=progress["progress_summary"],
            )
        )
    return summaries


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
    resp = JobDetailModel(
        job_id=job.job_id,
        status=job.status.value,
        created_at=job.created_at,
        updated_at=job.updated_at,
        error=job.error,
        documents=[_document_to_model(document) for document in job.documents],
    ).model_dump()
    resp["created_at"] = job.created_at.isoformat()
    resp["updated_at"] = job.updated_at.isoformat()
    return resp


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

    # Compute detailed progress
    progress = _compute_job_progress(job)

    resp = JobStatusModel(
        job_id=job.job_id,
        status=status_value,
        documents_total=total,
        documents_completed=completed,
        documents_failed=failed,
        updated_at=job.updated_at,
        error=job.error,
        progress_percentage=progress["progress_percentage"],
        progress_summary=progress["progress_summary"],
        current_document_id=progress["current_document_id"],
        current_stage=progress["current_stage"],
    ).model_dump()
    resp["updated_at"] = job.updated_at.isoformat()
    return resp


@router.post("/search/query", response_model=SearchResponse)
def search_query(
    request: SearchRequest,
    scopes = Depends(require_scopes("search:read")),
    service: SearchService = Depends(get_search_service),
    store = Depends(get_knowledge_base_store),
):
    claims, tenant = scopes
    knowledge_base_ids = _resolve_knowledge_base_ids(tenant, request.knowledge_base_ids, store)
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
        knowledge_base_ids=knowledge_base_ids,
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
        reranked=execution.reranked,  # Pass through reranking status
    )


@router.post("/generate/summary", response_model=GenerationResponseModel)
def generate_summary(
    request: GenerationRequest,
    scopes = Depends(require_scopes("generate:write")),
    service: GenerationService = Depends(get_generation_service),
    store = Depends(get_knowledge_base_store),
):
    claims, tenant = scopes
    knowledge_base_ids = _resolve_knowledge_base_ids(tenant, request.knowledge_base_ids, store)
    response = service.summarize(
        tenant,
        query=request.query,
        knowledge_base_ids=knowledge_base_ids,
        limit=request.limit,
    )
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
    store = Depends(get_knowledge_base_store),
):
    _claims, tenant = scopes
    knowledge_base_ids = _resolve_knowledge_base_ids(tenant, request.knowledge_base_ids, store)
    record = service.submit(
        tenant,
        query=request.query,
        chunk_ids=request.chunk_ids,
        positive=request.positive,
        comment=request.comment,
        metadata=request.metadata,
        knowledge_base_ids=knowledge_base_ids,
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


# Stage progress weights for percentage calculation
# acquired(10%) → parsed(20%) → chunked(35%) → enriched(55%) → embedded(75%) → uploaded(100%)
STAGE_WEIGHTS = {
    "acquired": 10,
    "parsed": 20,
    "chunked": 35,
    "enriched": 55,  # LLM metadata enrichment
    "embedded": 75,
    "uploaded": 100,
}


def _compute_job_progress(job: object) -> dict:
    """
    Compute detailed progress metrics for a job.

    Returns dict with:
    - progress_percentage: int (0-100)
    - progress_summary: str (e.g., "doc 3/10 at embedded")
    - current_document_id: str | None
    - current_stage: str | None
    """
    from ..services.ingestion_jobs import DocumentStatus

    documents = getattr(job, "documents", [])
    if not documents:
        return {
            "progress_percentage": 0,
            "progress_summary": None,
            "current_document_id": None,
            "current_stage": None,
        }

    total = len(documents)
    completed_count = 0
    running_doc = None
    running_stage = None

    # Calculate completion based on document status + stage weights
    total_progress = 0
    for doc in documents:
        status_value = getattr(doc, "status", DocumentStatus.QUEUED)
        doc_status = status_value.value if hasattr(status_value, "value") else str(status_value)

        if doc_status == "succeeded":
            total_progress += 100
            completed_count += 1
        elif doc_status == "failed":
            total_progress += 0  # Failed docs don't contribute to progress
        elif doc_status == "running":
            # Find most advanced stage for running document
            progress_dict = getattr(doc, "progress", {}) or {}
            max_stage_weight = 0
            current_stage = None

            for stage, stage_info in progress_dict.items():
                if isinstance(stage_info, dict) and stage_info.get("status") in ("running", "succeeded"):
                    weight = STAGE_WEIGHTS.get(stage, 0)
                    if weight > max_stage_weight:
                        max_stage_weight = weight
                        current_stage = stage

            total_progress += max_stage_weight
            if not running_doc:  # Track first running document
                running_doc = getattr(doc, "document_id", None)
                running_stage = current_stage
        else:  # queued
            total_progress += 0

    overall_percentage = int(total_progress / max(total, 1))

    # Build progress summary
    summary = None
    if running_doc and running_stage:
        summary = f"doc {completed_count + 1}/{total} at {running_stage}"
    elif completed_count == total:
        summary = f"completed {completed_count}/{total} docs"
    elif completed_count > 0:
        summary = f"{completed_count}/{total} docs completed"

    return {
        "progress_percentage": overall_percentage,
        "progress_summary": summary,
        "current_document_id": running_doc,
        "current_stage": running_stage,
    }


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
        knowledge_base_id=getattr(result, "knowledge_base_id", ""),
        status=status_str,
        policy=getattr(result, "policy", None),
        chunk_count=getattr(result, "chunk_count", 0),
        parser=getattr(result, "parser", None),
        artifact=artifact,
        metadata=metadata,
        error=getattr(result, "error", None),
        chunks=chunk_models,
        progress=dict(getattr(result, "progress", {}) or {}),
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
        text=chunk.text,  # Include actual chunk content!
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


def _knowledge_base_to_model(record, documents: List[str] | None = None) -> KnowledgeBaseModel:
    return KnowledgeBaseModel(
        knowledge_base_id=record.knowledge_base_id,
        description=record.description,
        document_count=record.document_count,
        keywords=list(record.keywords or []),
        created_at=record.created_at,
        updated_at=record.updated_at,
        last_document_title=record.last_document_title,
        last_ingested_at=record.last_ingested_at,
        documents=documents,
    )


def _resolve_knowledge_base_ids(tenant, requested: List[str], store) -> Tuple[str, ...]:
    if not requested:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="knowledge_base_ids must be provided")
    records = store.list(tenant)
    available = {record.knowledge_base_id: record for record in records}
    if not available:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="no knowledge bases configured for tenant")
    if len(requested) == 1 and requested[0] == "*":
        return tuple(available.keys())
    resolved: List[str] = []
    for kb_id in requested:
        if kb_id not in available:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"knowledge base not found: {kb_id}")
        if kb_id not in resolved:
            resolved.append(kb_id)
    return tuple(resolved)


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
