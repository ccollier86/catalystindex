from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..policies.resolver import resolve_policy
from ..services.generation import GenerationService
from ..services.ingestion import IngestionService
from ..services.search import SearchOptions, SearchService
from ..models.common import ChunkRecord
from ..config.settings import get_settings
from .dependencies import get_generation_service, get_ingestion_service, get_search_service, require_scopes

router = APIRouter()


class IngestRequest(BaseModel):
    document_id: str
    document_title: str
    content: str
    schema: str | None = None
    parser: str = Field(default="plain_text")


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


class IngestResponse(BaseModel):
    document_id: str
    policy: str
    chunk_count: int
    chunks: List[ChunkModel]


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
    service: IngestionService = Depends(get_ingestion_service),
):
    claims, tenant = scopes
    policy = resolve_policy(request.document_title, request.schema)
    result = service.ingest(
        tenant=tenant,
        document_id=request.document_id,
        document_title=request.document_title,
        content=request.content,
        policy=policy,
        parser_name=request.parser,
    )
    return IngestResponse(
        document_id=result.document_id,
        policy=result.policy.policy_name,
        chunk_count=len(result.chunks),
        chunks=[ChunkModel.from_record(chunk) for chunk in result.chunks],
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
