from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


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
class IngestionJob:
    document_id: str
    policy: str
    chunks: List[Chunk]


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


__all__ = ["Chunk", "IngestionJob", "SearchResult", "GenerationResult"]
