"""Visual element extraction and contextual linking service.

This module extracts images and tables from parsed documents and creates
smart contextual links between visual elements and text chunks based on:
1. Page proximity (same page or adjacent pages)
2. Text references (e.g., "Figure 3", "Table 2", "see image")

User requirement: "any chunk before and after an image or that references
an image or a table we make sure to put a pointer to it in the metadata
so if that chunk is surfaced we also surface that image and or table with it"
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Set

from ..models.common import ChunkRecord, VisualElement

logger = logging.getLogger(__name__)


@dataclass
class VisualExtractionResult:
    """Result of visual element extraction and linking."""

    visual_elements: List[VisualElement]
    chunks_with_visuals: List[ChunkRecord]  # Chunks with updated metadata containing visual links


class VisualLinker:
    """Extract and link visual elements to text chunks."""

    def __init__(self) -> None:
        # Regex patterns for detecting visual references in text
        self._figure_pattern = re.compile(
            r'\b(?:figure|fig\.?|image|diagram|chart|graph|illustration)\s*(\d+|[A-Z]|\d+[A-Z]?)\b',
            re.IGNORECASE
        )
        self._table_pattern = re.compile(
            r'\btable\s*(\d+|[A-Z]|\d+[A-Z]?)\b',
            re.IGNORECASE
        )
        self._see_pattern = re.compile(
            r'\b(?:see|shown\s+in|depicted\s+in|illustrated\s+in|refer\s+to)\s+(?:figure|fig\.?|table|image|diagram)',
            re.IGNORECASE
        )

    def extract_and_link(
        self,
        *,
        elements: Sequence[object],
        chunks: Sequence[ChunkRecord],
        document_id: str,
    ) -> VisualExtractionResult:
        """
        Extract visual elements from unstructured elements and link them to chunks.

        Args:
            elements: Raw unstructured elements (may include images/tables with Base64 data)
            chunks: Generated text chunks
            document_id: Document identifier

        Returns:
            VisualExtractionResult with visual elements and updated chunks
        """
        # Extract visual elements from unstructured elements
        visual_elements = self._extract_visual_elements(elements, document_id)

        if not visual_elements:
            logger.info(f"No visual elements found in document '{document_id}'")
            return VisualExtractionResult(
                visual_elements=[],
                chunks_with_visuals=list(chunks),
            )

        logger.info(
            f"Extracted {len(visual_elements)} visual elements from document '{document_id}': "
            f"{sum(1 for v in visual_elements if v.element_type == 'Image')} images, "
            f"{sum(1 for v in visual_elements if v.element_type == 'Table')} tables"
        )

        # Link visual elements to chunks
        chunks_with_visuals = self._link_visuals_to_chunks(
            visual_elements=visual_elements,
            chunks=chunks,
        )

        # Track which chunks reference which visuals
        linked_count = sum(
            1 for chunk in chunks_with_visuals
            if chunk.metadata.get("visual_element_ids")
        )

        logger.info(
            f"Linked visual elements to {linked_count}/{len(chunks)} chunks "
            f"in document '{document_id}'"
        )

        return VisualExtractionResult(
            visual_elements=visual_elements,
            chunks_with_visuals=chunks_with_visuals,
        )

    def _extract_visual_elements(
        self,
        elements: Sequence[object],
        document_id: str,
    ) -> List[VisualElement]:
        """Extract images and tables with Base64 data from unstructured elements."""
        visual_elements: List[VisualElement] = []
        element_counters: Dict[str, int] = {"Image": 0, "Table": 0}

        for element in elements:
            # Check if this is an image or table element
            category = (getattr(element, "category", "") or "").strip()
            if category not in {"Image", "Table"}:
                continue

            # Extract metadata
            metadata = getattr(element, "metadata", None)
            if not metadata:
                continue

            # Get Base64 image data from metadata
            image_base64 = getattr(metadata, "image_base64", None)
            if not image_base64:
                logger.debug(f"Skipping {category} element without Base64 data")
                continue

            # Get page number
            page_number = getattr(metadata, "page_number", None)
            if page_number is None:
                logger.warning(f"Skipping {category} element without page number")
                continue

            # Generate element ID
            element_counters[category] += 1
            element_id = f"{document_id}_{category.lower()}_page{page_number}_{element_counters[category]}"

            # Extract coordinates if available
            coordinates = None
            if hasattr(metadata, "coordinates"):
                coords_obj = metadata.coordinates
                if coords_obj:
                    coordinates = {
                        "x1": getattr(coords_obj, "x1", None),
                        "y1": getattr(coords_obj, "y1", None),
                        "x2": getattr(coords_obj, "x2", None),
                        "y2": getattr(coords_obj, "y2", None),
                    }

            # Extract caption if available (from nearby text elements)
            caption = None
            text = getattr(element, "text", "")
            if text and text.strip():
                caption = text.strip()

            # Create visual element
            visual_element = VisualElement(
                element_id=element_id,
                element_type=category,
                image_base64=image_base64,
                page_number=page_number,
                document_id=document_id,
                coordinates=coordinates,
                caption=caption,
                related_chunk_ids=[],  # Will be populated during linking
            )

            visual_elements.append(visual_element)
            logger.debug(
                f"Extracted {category} element '{element_id}' from page {page_number}"
            )

        return visual_elements

    def _link_visuals_to_chunks(
        self,
        *,
        visual_elements: List[VisualElement],
        chunks: Sequence[ChunkRecord],
    ) -> List[ChunkRecord]:
        """
        Link visual elements to chunks based on proximity and references.

        Strategy:
        1. Page proximity: Link visuals to chunks on same/adjacent pages
        2. Text references: Link visuals mentioned in chunk text
        3. Bidirectional: Update both chunk metadata and visual.related_chunk_ids
        """
        # Index visuals by page for proximity matching
        visuals_by_page: Dict[int, List[VisualElement]] = {}
        for visual in visual_elements:
            page = visual.page_number
            if page not in visuals_by_page:
                visuals_by_page[page] = []
            visuals_by_page[page].append(visual)

        # Process each chunk and find related visuals
        updated_chunks: List[ChunkRecord] = []

        for chunk in chunks:
            related_visual_ids: Set[str] = set()

            # Strategy 1: Page proximity
            # Link visuals on same page or adjacent pages (±1 page)
            chunk_pages = self._get_chunk_pages(chunk)
            for page in chunk_pages:
                # Check current page and adjacent pages
                for page_offset in [-1, 0, 1]:
                    check_page = page + page_offset
                    if check_page in visuals_by_page:
                        for visual in visuals_by_page[check_page]:
                            related_visual_ids.add(visual.element_id)

            # Strategy 2: Text references
            # Detect explicit references to figures/tables in chunk text
            referenced_visuals = self._detect_visual_references(
                chunk.text,
                visual_elements,
                chunk_pages,
            )
            related_visual_ids.update(referenced_visuals)

            # Update chunk metadata with visual links
            if related_visual_ids:
                chunk_metadata = dict(chunk.metadata)
                chunk_metadata["visual_element_ids"] = sorted(related_visual_ids)

                updated_chunk = replace(chunk, metadata=chunk_metadata)
                updated_chunks.append(updated_chunk)

                # Bidirectional: Update visual elements with chunk references
                for visual in visual_elements:
                    if visual.element_id in related_visual_ids:
                        if chunk.chunk_id not in visual.related_chunk_ids:
                            visual.related_chunk_ids.append(chunk.chunk_id)

                logger.debug(
                    f"Linked {len(related_visual_ids)} visuals to chunk '{chunk.chunk_id}' "
                    f"(pages {chunk.start_page}-{chunk.end_page})"
                )
            else:
                # No visual links - keep original chunk
                updated_chunks.append(chunk)

        return updated_chunks

    def _get_chunk_pages(self, chunk: ChunkRecord) -> Set[int]:
        """Get all pages covered by this chunk."""
        pages: Set[int] = set()

        # Use page_numbers from metadata if available (from enhanced SectionText)
        if "page_numbers" in chunk.metadata:
            page_numbers = chunk.metadata["page_numbers"]
            if isinstance(page_numbers, list):
                pages.update(page_numbers)

        # Fallback: Use start_page to end_page range
        if not pages and chunk.start_page and chunk.end_page:
            pages.update(range(chunk.start_page, chunk.end_page + 1))

        return pages

    def _detect_visual_references(
        self,
        text: str,
        visual_elements: List[VisualElement],
        chunk_pages: Set[int],
    ) -> Set[str]:
        """
        Detect explicit references to visual elements in text.

        Detects patterns like:
        - "Figure 3", "Fig. 2", "Image 1"
        - "Table 4", "Table A"
        - "see figure", "shown in table", "refer to diagram"

        Returns:
            Set of visual element IDs referenced in the text
        """
        referenced_ids: Set[str] = set()

        # Strategy 1: Detect figure references and match to nearby images
        figure_matches = self._figure_pattern.findall(text)
        if figure_matches or self._see_pattern.search(text):
            # Link to images on same/nearby pages
            for visual in visual_elements:
                if visual.element_type == "Image" and visual.page_number in chunk_pages:
                    referenced_ids.add(visual.element_id)

        # Strategy 2: Detect table references and match to nearby tables
        table_matches = self._table_pattern.findall(text)
        if table_matches:
            # Link to tables on same/nearby pages
            for visual in visual_elements:
                if visual.element_type == "Table" and visual.page_number in chunk_pages:
                    referenced_ids.add(visual.element_id)

        return referenced_ids


__all__ = ["VisualLinker", "VisualExtractionResult"]
