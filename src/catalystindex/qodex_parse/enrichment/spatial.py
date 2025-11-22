"""
Spatial Enrichment Layer - adds navigation relationships to chunks

Enriches chunks with spatial metadata including:
- Sequential edges (prev/next in document order)
- Spatial edges (above/below/left/right on page)
- Hierarchical edges (heading parent relationships)
- Page grouping

This is an INTERNAL enrichment layer during parsing, not the external graph.
"""

from typing import List, Optional, Dict, Any
from ..core.schemas import QodexChunk, Bbox, SpatialMetadata, ChunkType


class SpatialGraphBuilder:
    """
    Build spatial navigation graph from chunks.

    Creates pointer references between chunks based on:
    1. Document order (prev/next)
    2. Page layout (above/below/left/right)
    3. Heading hierarchy (parent headings)
    4. Page grouping (same page chunks)
    """

    def __init__(
        self,
        spatial_threshold: float = 0.5,  # Overlap threshold for spatial relationships
    ):
        """
        Args:
            spatial_threshold: Minimum overlap for spatial relationships (0-1)
        """
        self.spatial_threshold = spatial_threshold

    def build_graph(self, chunks: List[QodexChunk]) -> List[QodexChunk]:
        """
        Build spatial graph by updating chunk metadata with pointer references.

        Args:
            chunks: List of QodexChunk objects

        Returns:
            Updated chunks with spatial metadata populated
        """
        if not chunks:
            return chunks

        # Sort chunks by page and vertical position (top to bottom, left to right)
        sorted_chunks = self._sort_chunks_by_position(chunks)

        # Build sequential edges (prev/next in document order)
        self._build_sequential_edges(sorted_chunks)

        # Build spatial edges (above/below/left/right)
        self._build_spatial_edges(sorted_chunks)

        # Build hierarchical edges (heading parents)
        self._build_heading_hierarchy(sorted_chunks)

        # Build page grouping
        self._build_page_groups(sorted_chunks)

        return sorted_chunks

    def _sort_chunks_by_position(self, chunks: List[QodexChunk]) -> List[QodexChunk]:
        """
        Sort chunks by page number and position (top to bottom, left to right).

        For vertical text flow:
        - Higher y coordinates = top of page (PDF bottom-left origin)
        - Sort by page, then by y (descending), then by x (ascending)
        """
        return sorted(
            chunks,
            key=lambda c: (
                c.spatial.page,
                -c.spatial.bbox.y1,  # Descending (top first)
                c.spatial.bbox.x0,  # Ascending (left first)
            )
        )

    def _build_sequential_edges(self, sorted_chunks: List[QodexChunk]) -> None:
        """
        Build prev/next edges in document order.

        Modifies chunks in place.
        """
        for i, chunk in enumerate(sorted_chunks):
            # Set prev_chunk_id
            if i > 0:
                chunk.spatial.prev_chunk_id = sorted_chunks[i - 1].chunk_id

            # Set next_chunk_id
            if i < len(sorted_chunks) - 1:
                chunk.spatial.next_chunk_id = sorted_chunks[i + 1].chunk_id

    def _build_spatial_edges(self, chunks: List[QodexChunk]) -> None:
        """
        Build above/below/left/right edges based on bounding box positions.

        For each chunk, find the nearest chunk in each direction.
        """
        for chunk in chunks:
            # Find nearest chunks in each direction
            above = self._find_nearest_above(chunk, chunks)
            below = self._find_nearest_below(chunk, chunks)
            left = self._find_nearest_left(chunk, chunks)
            right = self._find_nearest_right(chunk, chunks)

            # Update spatial metadata
            if above:
                chunk.spatial.above_chunk_id = above.chunk_id
            if below:
                chunk.spatial.below_chunk_id = below.chunk_id
            if left:
                chunk.spatial.left_chunk_id = left.chunk_id
            if right:
                chunk.spatial.right_chunk_id = right.chunk_id

    def _find_nearest_above(
        self,
        chunk: QodexChunk,
        all_chunks: List[QodexChunk]
    ) -> Optional[QodexChunk]:
        """Find nearest chunk above this one"""
        candidates = [
            c for c in all_chunks
            if c.chunk_id != chunk.chunk_id
            and c.spatial.bbox.is_above(chunk.spatial.bbox)
            and c.spatial.bbox.overlaps_horizontal(chunk.spatial.bbox, self.spatial_threshold)
        ]

        if not candidates:
            return None

        # Return closest (minimum vertical distance)
        return min(
            candidates,
            key=lambda c: chunk.spatial.bbox.y0 - c.spatial.bbox.y1
        )

    def _find_nearest_below(
        self,
        chunk: QodexChunk,
        all_chunks: List[QodexChunk]
    ) -> Optional[QodexChunk]:
        """Find nearest chunk below this one"""
        candidates = [
            c for c in all_chunks
            if c.chunk_id != chunk.chunk_id
            and c.spatial.bbox.is_below(chunk.spatial.bbox)
            and c.spatial.bbox.overlaps_horizontal(chunk.spatial.bbox, self.spatial_threshold)
        ]

        if not candidates:
            return None

        # Return closest (minimum vertical distance)
        return min(
            candidates,
            key=lambda c: c.spatial.bbox.y0 - chunk.spatial.bbox.y1
        )

    def _find_nearest_left(
        self,
        chunk: QodexChunk,
        all_chunks: List[QodexChunk]
    ) -> Optional[QodexChunk]:
        """Find nearest chunk to the left"""
        candidates = [
            c for c in all_chunks
            if c.chunk_id != chunk.chunk_id
            and c.spatial.bbox.is_left_of(chunk.spatial.bbox)
            and c.spatial.bbox.overlaps_vertical(chunk.spatial.bbox, self.spatial_threshold)
        ]

        if not candidates:
            return None

        # Return closest (minimum horizontal distance)
        return min(
            candidates,
            key=lambda c: chunk.spatial.bbox.x0 - c.spatial.bbox.x1
        )

    def _find_nearest_right(
        self,
        chunk: QodexChunk,
        all_chunks: List[QodexChunk]
    ) -> Optional[QodexChunk]:
        """Find nearest chunk to the right"""
        candidates = [
            c for c in all_chunks
            if c.chunk_id != chunk.chunk_id
            and c.spatial.bbox.is_right_of(chunk.spatial.bbox)
            and c.spatial.bbox.overlaps_vertical(chunk.spatial.bbox, self.spatial_threshold)
        ]

        if not candidates:
            return None

        # Return closest (minimum horizontal distance)
        return min(
            candidates,
            key=lambda c: c.spatial.bbox.x0 - chunk.spatial.bbox.x1
        )

    def _build_heading_hierarchy(self, chunks: List[QodexChunk]) -> None:
        """
        Build heading hierarchy by linking chunks to their parent heading.

        For each chunk, find the most recent heading before it.
        """
        current_heading_stack: List[QodexChunk] = []  # Stack of nested headings

        for chunk in chunks:
            # If this is a heading, update the stack
            if chunk.type == ChunkType.HEADING:
                # Get current heading level (default to 3 if None)
                current_level = chunk.spatial.heading_level or 3

                # Pop headings of same or lower level
                while current_heading_stack:
                    stack_level = current_heading_stack[-1].spatial.heading_level or 3
                    if stack_level >= current_level:
                        current_heading_stack.pop()
                    else:
                        break

                # Add this heading to stack
                current_heading_stack.append(chunk)

            # If this is not a heading, link to current heading
            else:
                if current_heading_stack:
                    parent_heading = current_heading_stack[-1]
                    chunk.spatial.heading_parent_id = parent_heading.chunk_id

    def _build_page_groups(self, chunks: List[QodexChunk]) -> None:
        """
        Group chunks by page.

        For each chunk, collect IDs of all other chunks on same page.
        """
        # Build page index
        page_index: Dict[int, List[str]] = {}
        for chunk in chunks:
            page = chunk.spatial.page
            if page not in page_index:
                page_index[page] = []
            page_index[page].append(chunk.chunk_id)

        # Assign page groups to chunks
        for chunk in chunks:
            page = chunk.spatial.page
            # All chunks on page except self
            chunk.spatial.same_page_chunks = [
                cid for cid in page_index[page]
                if cid != chunk.chunk_id
            ]


class NavigationGraph:
    """
    Utility class for navigating the spatial graph.

    Provides convenience methods for graph traversal.
    """

    def __init__(self, chunks: List[QodexChunk]):
        """
        Args:
            chunks: List of chunks with spatial graph built
        """
        self.chunks = chunks
        self.chunk_index = {c.chunk_id: c for c in chunks}

    def get_chunk(self, chunk_id: str) -> Optional[QodexChunk]:
        """Get chunk by ID"""
        return self.chunk_index.get(chunk_id)

    def get_next(self, chunk_id: str) -> Optional[QodexChunk]:
        """Get next chunk in document order"""
        chunk = self.get_chunk(chunk_id)
        if chunk and chunk.spatial.next_chunk_id:
            return self.get_chunk(chunk.spatial.next_chunk_id)
        return None

    def get_prev(self, chunk_id: str) -> Optional[QodexChunk]:
        """Get previous chunk in document order"""
        chunk = self.get_chunk(chunk_id)
        if chunk and chunk.spatial.prev_chunk_id:
            return self.get_chunk(chunk.spatial.prev_chunk_id)
        return None

    def get_above(self, chunk_id: str) -> Optional[QodexChunk]:
        """Get chunk above"""
        chunk = self.get_chunk(chunk_id)
        if chunk and chunk.spatial.above_chunk_id:
            return self.get_chunk(chunk.spatial.above_chunk_id)
        return None

    def get_below(self, chunk_id: str) -> Optional[QodexChunk]:
        """Get chunk below"""
        chunk = self.get_chunk(chunk_id)
        if chunk and chunk.spatial.below_chunk_id:
            return self.get_chunk(chunk.spatial.below_chunk_id)
        return None

    def get_left(self, chunk_id: str) -> Optional[QodexChunk]:
        """Get chunk to the left"""
        chunk = self.get_chunk(chunk_id)
        if chunk and chunk.spatial.left_chunk_id:
            return self.get_chunk(chunk.spatial.left_chunk_id)
        return None

    def get_right(self, chunk_id: str) -> Optional[QodexChunk]:
        """Get chunk to the right"""
        chunk = self.get_chunk(chunk_id)
        if chunk and chunk.spatial.right_chunk_id:
            return self.get_chunk(chunk.spatial.right_chunk_id)
        return None

    def get_heading_parent(self, chunk_id: str) -> Optional[QodexChunk]:
        """Get parent heading"""
        chunk = self.get_chunk(chunk_id)
        if chunk and chunk.spatial.heading_parent_id:
            return self.get_chunk(chunk.spatial.heading_parent_id)
        return None

    def get_page_chunks(self, page: int) -> List[QodexChunk]:
        """Get all chunks on a page"""
        return [c for c in self.chunks if c.spatial.page == page]

    def get_heading_children(self, heading_id: str) -> List[QodexChunk]:
        """Get all chunks under a heading"""
        return [
            c for c in self.chunks
            if c.spatial.heading_parent_id == heading_id
        ]

    def reconstruct_page(self, page: int) -> str:
        """
        Reconstruct page text in reading order.

        Returns:
            Text content of page in proper reading order
        """
        page_chunks = self.get_page_chunks(page)

        # Sort by position (top to bottom, left to right)
        sorted_chunks = sorted(
            page_chunks,
            key=lambda c: (
                -c.spatial.bbox.y1,  # Top first
                c.spatial.bbox.x0,   # Left first
            )
        )

        # Join text
        return "\n\n".join(c.text for c in sorted_chunks)
