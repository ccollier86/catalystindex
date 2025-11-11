from __future__ import annotations

from dataclasses import dataclass
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
    chunk_tier: str
    section_slug: str
    track: str


@dataclass(slots=True)
class GenerationResult:
    summary: str
    citations: Dict[str, float]
    chunk_count: int


__all__ = [
    "ArtifactRef",
    "Chunk",
    "IngestionDocument",
    "IngestionJob",
    "IngestionJobSummary",
    "SearchResult",
    "GenerationResult",
]
