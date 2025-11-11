from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass(slots=True)
class Tenant:
    org_id: str
    workspace_id: str
    user_id: str


@dataclass(slots=True)
class SectionText:
    section_slug: str
    title: str
    text: str
    start_page: int = 1
    end_page: int = 1
    bbox_pointer: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    section_slug: str
    text: str
    chunk_tier: str
    start_page: int
    end_page: int
    bbox_pointer: Optional[str]
    summary: Optional[str]
    key_terms: List[str]
    requires_previous: bool
    prev_chunk_id: Optional[str]
    confidence_note: Optional[str]
    metadata: Dict[str, str]


@dataclass(slots=True)
class RetrievalResult:
    chunk: ChunkRecord
    score: float
    track: str
    vision_context: Optional[str]


@dataclass(slots=True)
class AuditEvent:
    event: str
    timestamp: datetime
    tenant: Tenant
    actor: str
    details: Dict[str, object] = field(default_factory=dict)


__all__ = [
    "Tenant",
    "SectionText",
    "ChunkRecord",
    "RetrievalResult",
    "AuditEvent",
]
