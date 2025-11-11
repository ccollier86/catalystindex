from __future__ import annotations

from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..services.generation import GenerationService
from ..services.ingestion_jobs import DocumentStatus, DocumentSubmission, IngestionCoordinator
from ..services.search import SearchOptions, SearchService
from ..models.common import ChunkRecord
from ..config.settings import get_settings
from .dependencies import (
    get_generation_service,
    get_ingestion_coordinator,
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


class SearchRequest(BaseModel):
    query: str
    economy_mode: bool = False
    limit: int | None = None
    filters: dict | None = None


class RetrievalModel(BaseModel):
    chunk_id: str
    score: float
    chunk_tier: str
    section_slug: str
    track: str


class SearchResponse(BaseModel):
    results: List[RetrievalModel]


class GenerationRequest(BaseModel):
    query: str
    limit: int = 6


class GenerationResponseModel(BaseModel):
    summary: str
    citations: dict
    chunk_count: int


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


@router.post("/search/query", response_model=SearchResponse)
def search_query(
    request: SearchRequest,
    scopes = Depends(require_scopes("search:read")),
    service: SearchService = Depends(get_search_service),
):
    claims, tenant = scopes
    options = SearchOptions(
        economy_mode=request.economy_mode,
        limit=request.limit or (10 if request.economy_mode else 24),
        filters=request.filters,
    )
    results = service.retrieve(tenant, query=request.query, options=options)
    return SearchResponse(
        results=[
            RetrievalModel(
                chunk_id=result.chunk.chunk_id,
                score=result.score,
                chunk_tier=result.chunk.chunk_tier,
                section_slug=result.chunk.section_slug,
                track=result.track,
            )
            for result in results
        ]
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


__all__ = ["router"]


def _document_to_model(result: object) -> DocumentResultModel:
    if isinstance(result, DocumentResultModel):  # pragma: no cover - defensive
        return result
    status_value = getattr(result, "status", DocumentStatus.PENDING)
    if isinstance(status_value, DocumentStatus):
        status_str = status_value.value
    else:
        status_str = str(status_value)
    metadata = dict(getattr(result, "metadata", {}) or {})
    artifact_uri = getattr(result, "artifact_uri", None)
    artifact_content_type = getattr(result, "artifact_content_type", None)
    artifact = None
    if artifact_uri:
        artifact = ArtifactModel(uri=artifact_uri, content_type=artifact_content_type)
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
