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
    metadata: Dict[str, object] = field(default_factory=dict)
    # Enhanced metadata for intelligent PDF splitting
    page_numbers: List[int] = field(default_factory=list)  # All pages in this section
    toc_title: Optional[str] = None  # Original TOC heading if from TOC
    toc_level: Optional[int] = None  # TOC depth (1=chapter, 2=subsection, etc.)
    section_type: str = "unknown"  # "toc_based" | "header_based" | "page_based" | "unknown"
    # OpenParse-specific fields
    bbox: Optional[Dict[str, float]] = None  # Normalized bbox: {x, y, width, height, page_height, page_width}
    node_type: Optional[str] = None  # OpenParse node type: "text", "heading", "table", "list", "image"
    image_refs: List[str] = field(default_factory=list)  # VisualElement IDs for related images
    text_height: Optional[float] = None  # Text height in points (for heading detection)


@dataclass(slots=True)
class VisualElement:
    """Represents an extracted image or table from a document."""

    element_id: str  # e.g., "doc123_table_page5_1"
    element_type: str  # "image" | "table"
    page_number: int  # Page where this visual appears
    document_id: str  # Parent document ID
    artifact_uri: Optional[str] = None  # S3/local URI for stored artifact
    image_base64: Optional[str] = None  # Base64-encoded image data (if not using artifact_uri)
    coordinates: Optional[Dict[str, float]] = None  # Normalized bbox: {x, y, width, height}
    caption: Optional[str] = None  # Figure/table caption if detected
    related_chunk_ids: List[str] = field(default_factory=list)  # Chunks that reference this visual
    spatial_position: Optional[str] = None  # "above" | "below" | "inline" relative to chunk


@dataclass(slots=True)
class PageRecord:
    """Represents a full page from a document for NVIDIA-style context expansion."""

    page_id: str  # e.g., "doc123_page5"
    document_id: str  # Parent document ID
    page_number: int  # Page number in document
    text: str  # Full page text content
    chunk_ids: List[str] = field(default_factory=list)  # Chunks extracted from this page
    metadata: Dict[str, object] = field(default_factory=dict)  # Additional page metadata


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
    metadata: Dict[str, object]


@dataclass(slots=True)
class RetrievalResult:
    chunk: ChunkRecord
    score: float
    track: str
    vision_context: Optional[str]
    visual_elements: List["VisualElement"] = field(default_factory=list)  # Images/tables linked to this chunk


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
    "VisualElement",
    "PageRecord",
    "ChunkRecord",
    "RetrievalResult",
    "AuditEvent",
]
