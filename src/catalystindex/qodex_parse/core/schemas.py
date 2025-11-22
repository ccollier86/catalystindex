"""
Core data schemas for Qodex Parse

Clean, organized data structures for document chunks with spatial,
semantic, and enhanced metadata layers.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum
import numpy as np


class ChunkType(str, Enum):
    """Type of content chunk"""
    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    IMAGE = "image"
    LIST = "list"
    CODE = "code"
    METADATA = "metadata"


@dataclass(frozen=True)
class Bbox:
    """
    Bounding box with page coordinates.
    Origin: bottom-left corner (PDF standard)
    """
    x0: float
    y0: float
    x1: float
    y1: float
    page: int
    page_width: float
    page_height: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2

    def overlaps_vertical(self, other: 'Bbox', threshold: float = 0.5) -> bool:
        """Check if this bbox overlaps vertically with another"""
        if self.page != other.page:
            return False

        overlap_height = min(self.y1, other.y1) - max(self.y0, other.y0)
        if overlap_height <= 0:
            return False

        min_height = min(self.height, other.height)
        return overlap_height / min_height >= threshold

    def overlaps_horizontal(self, other: 'Bbox', threshold: float = 0.5) -> bool:
        """Check if this bbox overlaps horizontally with another"""
        if self.page != other.page:
            return False

        overlap_width = min(self.x1, other.x1) - max(self.x0, other.x0)
        if overlap_width <= 0:
            return False

        min_width = min(self.width, other.width)
        return overlap_width / min_width >= threshold

    def is_above(self, other: 'Bbox') -> bool:
        """Check if this bbox is above another (higher y coordinate)"""
        return self.page == other.page and self.y0 > other.y1

    def is_below(self, other: 'Bbox') -> bool:
        """Check if this bbox is below another (lower y coordinate)"""
        return self.page == other.page and self.y1 < other.y0

    def is_left_of(self, other: 'Bbox') -> bool:
        """Check if this bbox is to the left of another"""
        return self.page == other.page and self.x1 < other.x0

    def is_right_of(self, other: 'Bbox') -> bool:
        """Check if this bbox is to the right of another"""
        return self.page == other.page and self.x0 > other.x1


@dataclass
class SpatialMetadata:
    """
    Spatial/structural metadata layer
    Preserves document structure and page layout
    """
    # Page location
    page: int
    bbox: Bbox

    # Sequential navigation (document order)
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None

    # Spatial navigation (page layout)
    above_chunk_id: Optional[str] = None
    below_chunk_id: Optional[str] = None
    left_chunk_id: Optional[str] = None
    right_chunk_id: Optional[str] = None

    # Hierarchical navigation
    heading_parent_id: Optional[str] = None
    heading_level: Optional[int] = None

    # Page grouping
    same_page_chunks: List[str] = field(default_factory=list)


@dataclass
class SemanticMetadata:
    """
    Semantic metadata layer
    Captures meaning and topic relationships
    """
    # Semantic grouping
    group_id: Optional[str] = None

    # Similarity to adjacent chunks
    similarity_to_prev: Optional[float] = None
    similarity_to_next: Optional[float] = None

    # Topic boundaries
    is_topic_boundary: bool = False
    topic_shift_score: Optional[float] = None

    # Semantic neighbors (similar content anywhere in doc)
    semantic_neighbors: List[Dict[str, Any]] = field(default_factory=list)
    # Format: [{"chunk_id": "...", "similarity": 0.85, "excerpt": "..."}]


@dataclass
class EnhancedMetadata:
    """
    Enhanced/intelligent metadata layer
    LLM-generated or user-refined metadata
    """
    # Keywords and search terms
    keywords: List[str] = field(default_factory=list)
    search_terms: List[str] = field(default_factory=list)

    # Topic labeling
    topic_label: Optional[str] = None

    # User feedback and refinement
    user_annotations: Dict[str, Any] = field(default_factory=dict)
    quality_score: Optional[float] = None


@dataclass
class TableMetadata:
    """
    Table-specific metadata
    Used when chunk_type == TABLE
    """
    # Table structure
    num_rows: int
    num_cols: int
    has_header: bool

    # Table extraction method
    extraction_method: Literal["docling", "pymupdf", "unitable"]
    confidence: Optional[float] = None

    # Parsed data
    data: List[List[str]] = field(default_factory=list)  # Raw cell data
    df_json: Optional[str] = None  # Pandas DataFrame as JSON
    markdown: Optional[str] = None  # Markdown representation


@dataclass
class ImageMetadata:
    """
    Image-specific metadata
    Used when chunk_type == IMAGE
    """
    # Image properties
    width: int
    height: int
    format: str  # PNG, JPEG, etc.

    # Image data
    base64_data: Optional[str] = None
    file_path: Optional[str] = None

    # Image description (optional, from OCR or LLM)
    caption: Optional[str] = None
    alt_text: Optional[str] = None
    ocr_text: Optional[str] = None


@dataclass
class QodexChunk:
    """
    Main chunk data structure with layered metadata

    Design philosophy:
    - Single parse, multiple lenses
    - Preserve structure AND capture meaning
    - Clean, organized, queryable
    """

    # ===== IDENTITY =====
    chunk_id: str
    doc_id: str

    # ===== CONTENT =====
    text: str
    type: ChunkType
    tokens: int

    # ===== LAYERED METADATA =====
    spatial: SpatialMetadata
    semantic: Optional[SemanticMetadata] = None
    enhanced: Optional[EnhancedMetadata] = None

    # ===== TYPE-SPECIFIC METADATA =====
    table_meta: Optional[TableMetadata] = None
    image_meta: Optional[ImageMetadata] = None

    # ===== RAW DATA =====
    elements: List[Any] = field(default_factory=list)  # Raw OpenParse elements
    embedding: Optional[np.ndarray] = None

    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for JSON export"""
        result = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "type": self.type.value,
            "tokens": self.tokens,
            "spatial": {
                "page": self.spatial.page,
                "bbox": {
                    "x0": self.spatial.bbox.x0,
                    "y0": self.spatial.bbox.y0,
                    "x1": self.spatial.bbox.x1,
                    "y1": self.spatial.bbox.y1,
                    "page": self.spatial.bbox.page,
                    "page_width": self.spatial.bbox.page_width,
                    "page_height": self.spatial.bbox.page_height,
                },
                "prev_chunk_id": self.spatial.prev_chunk_id,
                "next_chunk_id": self.spatial.next_chunk_id,
                "above_chunk_id": self.spatial.above_chunk_id,
                "below_chunk_id": self.spatial.below_chunk_id,
                "left_chunk_id": self.spatial.left_chunk_id,
                "right_chunk_id": self.spatial.right_chunk_id,
                "heading_parent_id": self.spatial.heading_parent_id,
                "heading_level": self.spatial.heading_level,
                "same_page_chunks": self.spatial.same_page_chunks,
            }
        }

        if self.semantic:
            result["semantic"] = {
                "group_id": self.semantic.group_id,
                "similarity_to_prev": self.semantic.similarity_to_prev,
                "similarity_to_next": self.semantic.similarity_to_next,
                "is_topic_boundary": self.semantic.is_topic_boundary,
                "topic_shift_score": self.semantic.topic_shift_score,
                "semantic_neighbors": self.semantic.semantic_neighbors,
            }

        if self.enhanced:
            result["enhanced"] = {
                "keywords": self.enhanced.keywords,
                "search_terms": self.enhanced.search_terms,
                "topic_label": self.enhanced.topic_label,
                "user_annotations": self.enhanced.user_annotations,
                "quality_score": self.enhanced.quality_score,
            }

        if self.table_meta:
            result["table_meta"] = {
                "num_rows": self.table_meta.num_rows,
                "num_cols": self.table_meta.num_cols,
                "has_header": self.table_meta.has_header,
                "extraction_method": self.table_meta.extraction_method,
                "confidence": self.table_meta.confidence,
                "data": self.table_meta.data,
                "df_json": self.table_meta.df_json,
                "markdown": self.table_meta.markdown,
            }

        if self.image_meta:
            result["image_meta"] = {
                "width": self.image_meta.width,
                "height": self.image_meta.height,
                "format": self.image_meta.format,
                "base64_data": self.image_meta.base64_data,
                "file_path": self.image_meta.file_path,
                "caption": self.image_meta.caption,
                "alt_text": self.image_meta.alt_text,
                "ocr_text": self.image_meta.ocr_text,
            }

        if include_embedding and self.embedding is not None:
            result["embedding"] = self.embedding.tolist()

        return result


@dataclass
class QodexDocument:
    """
    Complete parsed document with all chunks and metadata
    """
    doc_id: str
    filename: str
    num_pages: int
    chunks: List[QodexChunk]

    # Document-level metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_chunks_by_page(self, page: int) -> List[QodexChunk]:
        """Get all chunks on a specific page"""
        return [c for c in self.chunks if c.spatial.page == page]

    def get_chunks_by_type(self, chunk_type: ChunkType) -> List[QodexChunk]:
        """Get all chunks of a specific type"""
        return [c for c in self.chunks if c.type == chunk_type]

    def get_chunk_by_id(self, chunk_id: str) -> Optional[QodexChunk]:
        """Get a specific chunk by ID"""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    def to_dict(self, include_embeddings: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for JSON export"""
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "num_pages": self.num_pages,
            "chunks": [c.to_dict(include_embedding=include_embeddings) for c in self.chunks],
            "metadata": self.metadata,
        }
