"""
Qodex-Parse adapter for production-grade universal document parsing.

Features:
- Universal format support (PDF, DOCX, HTML, Markdown)
- Fast PyMuPDF text extraction with semantic chunking
- Modern table extraction (PyMuPDF 1.26.x find_tables API)
- Accurate bounding box coordinates for all elements
- Spatial metadata (prev/next/above/below relationships)
- Optional semantic layer (embeddings, neighbors)
- Optional enhanced metadata (LLM keywords, questions)
- Clean, modern codebase (2025)
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Any

from .base import ParserAdapter, ParserMetadata
from ..models.common import SectionText, VisualElement
from ..qodex_parse import parse
from ..qodex_parse.core.schemas import QodexDocument, QodexChunk

logger = logging.getLogger(__name__)


# Parser metadata for registry
QODEX_PARSER_METADATA = ParserMetadata(
    name="qodex",
    content_types=(
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
        "text/html",
        "text/markdown",
    ),
    requires=("pymupdf",),
    description="Universal document parser with spatial metadata, table extraction, and optional semantic layer (PyMuPDF 1.26.x)",
)


class QodexParserAdapter(ParserAdapter):
    """
    Production-ready Qodex-Parse adapter for universal document parsing.

    Advantages over OpenParse:
    - Universal format support (PDF, DOCX, HTML, MD)
    - Modern PyMuPDF 1.26.x API (2025)
    - Cleaner metadata structure
    - Better spatial relationships
    - Optional semantic/enhanced layers
    - Simpler codebase, easier to maintain
    """

    def __init__(
        self,
        *,
        use_tables: bool = True,
        use_images: bool = True,
        use_headings: bool = True,
        mode: str = "basic",  # "basic" | "spatial" | "semantic" | "full"
        openai_key: Optional[str] = None,
    ) -> None:
        """
        Initialize Qodex-Parse adapter.

        Args:
            use_tables: Enable table extraction (default: True)
            use_images: Enable image extraction (default: True)
            use_headings: Enable heading detection (default: True)
            mode: Parsing mode - "basic" (free), "spatial", "semantic", or "full"
            openai_key: OpenAI API key (required for semantic/full modes)
        """
        self.use_tables = use_tables
        self.use_images = use_images
        self.use_headings = use_headings
        self.mode = mode
        self.openai_key = openai_key

        logger.info(
            f"Initialized QodexParserAdapter with mode={mode}, "
            f"tables={use_tables}, images={use_images}, headings={use_headings}"
        )

    def parse(
        self,
        content: bytes | str,
        *,
        document_title: str,
        content_type: str | None = None,
    ) -> Iterable[SectionText]:
        """
        Parse document content using Qodex-Parse.

        Args:
            content: Document bytes or string
            document_title: Document title for metadata
            content_type: Content type (e.g., "application/pdf")

        Returns:
            Iterable of SectionText objects with rich metadata

        Raises:
            ValueError: If content is invalid or parsing fails
        """
        # Handle string content (artifact pointers)
        if isinstance(content, str):
            # Assume it's a file path
            file_path = content
            temp_file = None
        else:
            # Save bytes to temp file (qodex needs file path)
            temp_file = tempfile.NamedTemporaryFile(
                suffix=self._get_suffix(content_type),
                delete=False
            )
            temp_file.write(content)
            temp_file.flush()
            file_path = temp_file.name

        try:
            # Parse with qodex_parse
            doc = parse(
                file_path,
                mode=self.mode,
                tables=self.use_tables,
                images=self.use_images,
                headings=self.use_headings,
                openai_key=self.openai_key,
            )

            logger.info(
                f"Parsed {doc.filename}: {len(doc.chunks)} chunks from {doc.num_pages} pages"
            )

            # Convert QodexChunks to SectionText
            yield from self._convert_chunks(doc, document_title)

        finally:
            # Clean up temp file
            if temp_file:
                Path(temp_file.name).unlink(missing_ok=True)

    def _get_suffix(self, content_type: Optional[str]) -> str:
        """Get file suffix from content type."""
        if not content_type:
            return ".pdf"  # Default to PDF

        type_map = {
            "application/pdf": ".pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "text/html": ".html",
            "text/markdown": ".md",
            "text/plain": ".txt",
        }
        return type_map.get(content_type, ".pdf")

    def _convert_chunks(
        self,
        doc: QodexDocument,
        document_title: str,
    ) -> Iterable[SectionText]:
        """
        Convert QodexChunks to SectionText format.

        Args:
            doc: Parsed QodexDocument
            document_title: Document title

        Yields:
            SectionText objects
        """
        for chunk in doc.chunks:
            # Build metadata dict
            metadata: Dict[str, Any] = {
                "parser": "qodex",
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "chunk_type": chunk.type.value,
                "tokens": chunk.tokens,
            }

            # Add spatial metadata if available
            if chunk.spatial:
                metadata["bbox"] = {
                    "x0": chunk.spatial.bbox.x0,
                    "y0": chunk.spatial.bbox.y0,
                    "x1": chunk.spatial.bbox.x1,
                    "y1": chunk.spatial.bbox.y1,
                    "page": chunk.spatial.bbox.page,
                    "page_width": chunk.spatial.bbox.page_width,
                    "page_height": chunk.spatial.bbox.page_height,
                }

                metadata["spatial"] = {
                    "prev_chunk_id": chunk.spatial.prev_chunk_id,
                    "next_chunk_id": chunk.spatial.next_chunk_id,
                    "above_chunk_id": chunk.spatial.above_chunk_id,
                    "below_chunk_id": chunk.spatial.below_chunk_id,
                    "heading_parent_id": chunk.spatial.heading_parent_id,
                    "heading_level": chunk.spatial.heading_level,
                }

            # Add semantic metadata if available
            if chunk.semantic:
                metadata["semantic"] = {
                    "group_id": chunk.semantic.group_id,
                    "similarity_to_prev": chunk.semantic.similarity_to_prev,
                    "similarity_to_next": chunk.semantic.similarity_to_next,
                    "is_topic_boundary": chunk.semantic.is_topic_boundary,
                }

                # Add semantic neighbors (top 3 for conciseness)
                if chunk.semantic.semantic_neighbors:
                    metadata["semantic_neighbors"] = [
                        {
                            "chunk_id": neighbor["chunk_id"],
                            "similarity": neighbor["similarity"],
                            "excerpt": neighbor["excerpt"],
                        }
                        for neighbor in chunk.semantic.semantic_neighbors[:3]
                    ]

            # Add enhanced metadata if available
            if chunk.enhanced:
                if chunk.enhanced.keywords:
                    metadata["keywords"] = chunk.enhanced.keywords
                if chunk.enhanced.search_terms:
                    metadata["search_terms"] = chunk.enhanced.search_terms
                if chunk.enhanced.topic_label:
                    metadata["topic_label"] = chunk.enhanced.topic_label

            # Add table metadata if available
            if chunk.table_meta:
                metadata["table"] = {
                    "num_rows": chunk.table_meta.num_rows,
                    "num_cols": chunk.table_meta.num_cols,
                    "has_header": chunk.table_meta.has_header,
                    "markdown": chunk.table_meta.markdown,
                }

            # Add image metadata if available
            if chunk.image_meta:
                metadata["image"] = {
                    "width": chunk.image_meta.width,
                    "height": chunk.image_meta.height,
                    "format": chunk.image_meta.format,
                }
                # Include base64 data if present (for visual elements)
                if chunk.image_meta.base64_data:
                    metadata["image"]["base64"] = chunk.image_meta.base64_data

            # Determine page numbers
            start_page = chunk.spatial.page if chunk.spatial else 1
            end_page = start_page
            page_numbers = [start_page]  # CRITICAL: required by visual_linker.py:260

            # Create section slug (unique identifier)
            section_slug = chunk.chunk_id

            # Create title (use chunk text preview or type)
            title = (
                chunk.text[:50] + "..."
                if len(chunk.text) > 50
                else chunk.text
            ) or f"{chunk.type.value}_chunk"

            # Build bbox_pointer AND normalized bbox dict
            bbox_pointer = None
            bbox_dict = None
            text_height = None

            if chunk.spatial and chunk.spatial.bbox:
                bbox = chunk.spatial.bbox

                # bbox_pointer for backwards compatibility
                bbox_pointer = f"page={bbox.page},x0={bbox.x0:.1f},y0={bbox.y0:.1f},x1={bbox.x1:.1f},y1={bbox.y1:.1f}"

                # Normalized bbox dict (CRITICAL: required by visual_linker)
                # Matches OpenParse format: normalized coordinates (0-1 range)
                bbox_dict = {
                    "x": bbox.x0 / bbox.page_width if bbox.page_width > 0 else 0,
                    "y": bbox.y0 / bbox.page_height if bbox.page_height > 0 else 0,
                    "width": (bbox.x1 - bbox.x0) / bbox.page_width if bbox.page_width > 0 else 0,
                    "height": (bbox.y1 - bbox.y0) / bbox.page_height if bbox.page_height > 0 else 0,
                    "page": bbox.page,
                    "page_height": bbox.page_height,
                    "page_width": bbox.page_width,
                }

                # Calculate text height for heading detection
                text_height = bbox.y1 - bbox.y0

            # Map QodexChunk type to node_type (matches OpenParse types)
            node_type_map = {
                "text": "text",
                "heading": "heading",
                "table": "table",
                "image": "image",
                "list": "list",
            }
            node_type = node_type_map.get(chunk.type.value.lower(), "text")

            # Create SectionText with all required fields
            section = SectionText(
                section_slug=section_slug,
                title=title,
                text=chunk.text,
                start_page=start_page,
                end_page=end_page,
                page_numbers=page_numbers,  # CRITICAL: required
                bbox=bbox_dict,  # CRITICAL: required for visual linking
                bbox_pointer=bbox_pointer,  # Legacy field
                node_type=node_type,  # CRITICAL: required
                image_refs=[],  # Empty for now, populated by visual_linker
                text_height=text_height,  # For heading detection
                metadata=metadata,
            )

            yield section


__all__ = ["QodexParserAdapter", "QODEX_PARSER_METADATA"]
