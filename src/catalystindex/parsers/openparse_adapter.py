"""
OpenParse adapter for production-grade PDF parsing.

Features:
- Fast, reliable text extraction with semantic chunking
- Accurate bounding box coordinates for all elements
- Table extraction with PyMuPDF (fast) or Unitable (SOTA)
- Image extraction with base64 encoding
- Spatial relationship tracking (images above/below text)
- Node type detection (text, heading, table, list, image)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
import uuid
from typing import Iterable, List, Dict, Optional, Any

try:
    from openparse import DocumentParser, processing
    from openparse.schemas import Bbox, Node, ImageElement, TableElement
except ImportError:
    raise ImportError("OpenParse is required. Install with: pip install openparse")

from .base import ParserAdapter, ParserMetadata
from ..models.common import SectionText, VisualElement

logger = logging.getLogger(__name__)


class OpenParseAdapter(ParserAdapter):
    """
    Production-ready OpenParse adapter for PDFs.

    Uses BasicIngestionPipeline to preserve document structure (headings, node types)
    while providing fast, accurate text extraction and rich metadata.

    Benefits over UnstructuredAdapter:
    - 10x faster (no PyMuPDF reconstruction issues)
    - Reliable bounding boxes for all elements
    - No 0-element failures
    - Built-in table extraction (PyMuPDF or Unitable)
    - Image extraction with base64 encoding
    - Spatial relationship tracking
    """

    def __init__(
        self,
        *,
        use_tables: bool = True,
        table_algorithm: str = "pymupdf",  # "pymupdf" | "unitable" | "table-transformers"
    ) -> None:
        """
        Initialize OpenParse adapter.

        Args:
            use_tables: Enable table extraction (default: True)
            table_algorithm: Table parsing algorithm (default: "pymupdf")
        """
        # Use BasicIngestionPipeline to preserve node_type and structure
        pipeline = processing.BasicIngestionPipeline()

        # Configure table extraction
        table_args = None
        if use_tables:
            table_args = {
                "parsing_algorithm": table_algorithm,
                "table_output_format": "markdown",  # Markdown is best for RAG
            }

        self.parser = DocumentParser(
            processing_pipeline=pipeline,
            table_args=table_args if use_tables else None,
        )

        logger.info(
            f"Initialized OpenParseAdapter with {table_algorithm if use_tables else 'no'} table extraction"
        )

    def parse(
        self,
        content: bytes | str,
        *,
        document_title: str,
        content_type: str | None = None,
    ) -> Iterable[SectionText]:
        """
        Parse PDF content using OpenParse.

        Args:
            content: PDF bytes or string
            document_title: Document title for metadata
            content_type: Content type (ignored, OpenParse is PDF-only)

        Returns:
            Iterable of SectionText objects with rich metadata
        """
        payload = content.encode("utf-8") if isinstance(content, str) else content

        # OpenParse requires a file path, write to temp file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(payload)
            tmp_path = tmp.name

        try:
            logger.info(f"Parsing PDF '{document_title}' with OpenParse")

            # Parse with OpenParse
            parsed_doc = self.parser.parse(tmp_path)

            # Extract images first (so we can link them to text chunks)
            images_by_page = self._extract_images(parsed_doc.nodes, document_title)

            # Convert OpenParse nodes to SectionText with spatial relationships
            sections = self._convert_nodes_to_sections(
                parsed_doc.nodes,
                document_title,
                images_by_page,
            )

            logger.info(
                f"Successfully parsed '{document_title}': "
                f"{len(sections)} sections, {sum(len(imgs) for imgs in images_by_page.values())} images"
            )

            return sections

        except Exception as exc:
            logger.error(f"OpenParse failed to parse '{document_title}': {exc}", exc_info=True)
            raise RuntimeError(
                f"OpenParse parsing failed for '{document_title}'. " f"Error: {str(exc)}"
            ) from exc
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _extract_images(
        self, nodes: List[Node], document_title: str
    ) -> Dict[int, List[VisualElement]]:
        """
        Extract all images from nodes and organize by page.

        Args:
            nodes: List of OpenParse nodes
            document_title: Document title for element IDs

        Returns:
            Dict mapping page number to list of VisualElements
        """
        images_by_page: Dict[int, List[VisualElement]] = {}

        for node in nodes:
            # Only process image nodes
            if "image" not in node.variant:
                continue

            if not node.elements:
                continue

            page = node.start_page
            bbox = node.bbox[0] if node.bbox else None

            for idx, elem in enumerate(node.elements):
                if not isinstance(elem, ImageElement):
                    continue

                # Create unique element ID
                element_id = f"{self._safe_slug(document_title, 'doc')}_image_p{page}_{idx}"

                # Normalize bbox coordinates
                coords = None
                if bbox:
                    coords = {
                        "x": bbox.x0 / bbox.page_width,
                        "y": bbox.y0 / bbox.page_height,
                        "width": (bbox.x1 - bbox.x0) / bbox.page_width,
                        "height": (bbox.y1 - bbox.y0) / bbox.page_height,
                        "page_height": bbox.page_height,
                        "page_width": bbox.page_width,
                    }

                visual_elem = VisualElement(
                    element_id=element_id,
                    element_type="image",
                    page_number=page,
                    document_id=document_title,
                    image_base64=elem.image,  # Already base64-encoded PNG
                    coordinates=coords,
                    caption=elem.text if elem.text else None,
                )

                images_by_page.setdefault(page, []).append(visual_elem)

        return images_by_page

    def _convert_nodes_to_sections(
        self,
        nodes: List[Node],
        document_title: str,
        images_by_page: Dict[int, List[VisualElement]],
    ) -> List[SectionText]:
        """
        Convert OpenParse nodes to SectionText format with spatial relationships.

        Args:
            nodes: List of OpenParse nodes
            document_title: Document title for fallback
            images_by_page: Images organized by page for linking

        Returns:
            List of SectionText objects
        """
        sections = []

        for i, node in enumerate(nodes):
            # Skip image nodes (already processed separately)
            if "image" in node.variant:
                continue

            # Extract text
            text = node.text.strip() if node.text else ""
            if not text:
                continue

            # Determine node type
            node_type = self._get_node_type(node)

            # Create section title
            if node_type == "heading":
                title = text[:100]  # Use heading text as title
            elif "table" in node.variant:
                title = f"{document_title} - Table on Page {node.start_page + 1}"
            else:
                title = f"{document_title} - Section {i+1}"

            # Create slug
            slug = self._safe_slug(
                text[:50] if node_type == "heading" else f"section-{i}", f"section-{i}"
            )

            # Calculate page numbers (OpenParse uses 0-indexed, we use 1-indexed)
            start_page = node.start_page + 1
            end_page = node.end_page + 1
            page_numbers = list(range(start_page, end_page + 1))

            # Extract bbox and calculate text height
            bbox_dict = None
            text_height = None
            if node.bbox:
                first_bbox = node.bbox[0]
                text_height = first_bbox.y1 - first_bbox.y0

                # Normalize coordinates for storage
                bbox_dict = {
                    "x": first_bbox.x0 / first_bbox.page_width,
                    "y": first_bbox.y0 / first_bbox.page_height,
                    "width": (first_bbox.x1 - first_bbox.x0) / first_bbox.page_width,
                    "height": (first_bbox.y1 - first_bbox.y0) / first_bbox.page_height,
                    "page": first_bbox.page,
                    "page_height": first_bbox.page_height,
                    "page_width": first_bbox.page_width,
                }

            # Find related images on the same page
            image_refs = []
            page = node.start_page
            if page in images_by_page:
                for img in images_by_page[page]:
                    image_refs.append(img.element_id)

                    # Determine spatial position (above/below)
                    if bbox_dict and img.coordinates:
                        img.spatial_position = self._determine_spatial_position(
                            bbox_dict, img.coordinates
                        )
                        # Link this chunk to the image
                        img.related_chunk_ids.append(slug)

            # Build comprehensive metadata
            metadata = {
                "parser": "openparse",
                "node_type": node_type,
                "variant": list(node.variant),
            }

            # Add table-specific metadata
            if "table" in node.variant and node.elements:
                metadata["table_format"] = "markdown"
                metadata["table_element_count"] = len(node.elements)

            # Add node attributes that exist
            if hasattr(node, "tokens") and node.tokens:
                metadata["tokens"] = node.tokens

            # Create SectionText with all enhanced fields
            section = SectionText(
                section_slug=slug,
                title=title,
                text=text,
                start_page=start_page,
                end_page=end_page,
                page_numbers=page_numbers,
                bbox=bbox_dict,
                node_type=node_type,
                image_refs=image_refs,
                text_height=text_height,
                metadata=metadata,
            )

            sections.append(section)

        return sections

    def _get_node_type(self, node: Node) -> str:
        """
        Determine node type from OpenParse node.

        Args:
            node: OpenParse node

        Returns:
            Node type string: "text", "heading", "table", "list", "image"
        """
        # Check variant first
        if "table" in node.variant:
            return "table"
        if "image" in node.variant:
            return "image"

        # Check for node_type attribute (BasicIngestionPipeline provides this)
        if hasattr(node, "node_type") and node.node_type:
            return node.node_type

        # Fall back to heuristic heading detection using bbox height
        if node.bbox:
            text_height = node.bbox[0].y1 - node.bbox[0].y0
            text_len = len(node.text.strip())

            # Large text with short content is likely a heading
            if text_height > 15 and text_len < 100:
                return "heading"

        return "text"

    def _determine_spatial_position(
        self, text_bbox: Dict[str, float], image_bbox: Dict[str, float]
    ) -> str:
        """
        Determine if image is above or below text chunk.

        Args:
            text_bbox: Normalized text bounding box
            image_bbox: Normalized image bounding box

        Returns:
            "above" | "below" | "inline"
        """
        # Compare y coordinates (remember: bottom-left origin in PDF)
        text_y_bottom = text_bbox["y"]
        image_y_bottom = image_bbox["y"]

        # If image bottom is higher than text bottom, image is above
        if image_y_bottom > text_y_bottom + 0.05:  # 5% threshold
            return "above"
        elif image_y_bottom < text_y_bottom - 0.05:
            return "below"
        else:
            return "inline"

    def _safe_slug(self, text: str, fallback: str, *, limit: int = 64) -> str:
        """
        Create a safe slug from text.

        Args:
            text: Text to slugify
            fallback: Fallback slug if text is invalid
            limit: Maximum slug length

        Returns:
            Safe slug string
        """
        slug = "".join(ch if ch.isalnum() else "-" for ch in text.lower()).strip("-")
        slug = "-".join(filter(None, slug.split("-")))
        return slug[:limit] or fallback


OPENPARSE_PARSER_METADATA = ParserMetadata(
    name="openparse",
    content_types=("application/pdf",),
    requires=("openparse",),
    description="Fast semantic PDF parsing with OpenParse - tables, images, and rich metadata",
)


__all__ = ["OpenParseAdapter", "OPENPARSE_PARSER_METADATA"]
