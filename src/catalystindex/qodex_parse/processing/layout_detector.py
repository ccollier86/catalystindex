"""
Layout Detection Utility

Analyzes PDF page bounding boxes to detect layout patterns:
- Single column (normal)
- Two columns (dual column)
- Multi-column (3+)
- Complex/mixed layouts

Returns standardized labels for routing to appropriate parsers.
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class LayoutType(Enum):
    """
    Standardized layout classifications.

    Note: Tables and images inline are NOT considered complex -
    they're normal features that standard parsers handle well.
    COMPLEX is reserved for truly unusual layouts (irregular columns,
    mixed column counts, overlapping text, etc.)
    """
    SINGLE_COLUMN = "single_column"
    TWO_COLUMN = "two_column"
    MULTI_COLUMN = "multi_column"  # 3+ columns
    COMPLEX = "complex"  # Irregular columns, mixed patterns, overlapping text
    UNKNOWN = "unknown"


@dataclass
class LayoutAnalysis:
    """Result of layout detection"""
    layout_type: LayoutType
    confidence: float  # 0.0 to 1.0
    num_columns: int
    column_boundaries: List[Tuple[float, float]]  # List of (x_start, x_end) for each column
    metadata: Dict[str, Any]  # Additional info about the layout


class LayoutDetector:
    """
    Detects page layout patterns from bounding boxes.

    Strategy:
    1. Analyze x-coordinate distribution of text blocks
    2. Detect vertical clustering (columns)
    3. Check for alignment patterns
    4. Classify layout type
    """

    def __init__(
        self,
        column_gap_threshold: float = 20.0,  # Minimum gap between columns
        alignment_tolerance: float = 15.0,    # How close x-coords need to be to consider "aligned"
        min_blocks_per_column: int = 2,       # Minimum blocks to consider a column
    ):
        self.column_gap_threshold = column_gap_threshold
        self.alignment_tolerance = alignment_tolerance
        self.min_blocks_per_column = min_blocks_per_column

    def analyze_page(self, blocks: List[Dict[str, Any]], page_width: float, page_height: float) -> LayoutAnalysis:
        """
        Analyze a page's layout from its text blocks.

        Args:
            blocks: List of text blocks with bbox info (from PyMuPDF)
            page_width: Page width in points
            page_height: Page height in points

        Returns:
            LayoutAnalysis with classification and metadata
        """
        if not blocks:
            return LayoutAnalysis(
                layout_type=LayoutType.UNKNOWN,
                confidence=0.0,
                num_columns=0,
                column_boundaries=[],
                metadata={"reason": "no_blocks"}
            )

        # Step 1: Extract x-coordinates (left edges of blocks)
        x_starts = [block['x0'] for block in blocks]
        x_ends = [block['x1'] for block in blocks]

        # Step 2: Detect columns by clustering x_starts
        columns = self._detect_columns(x_starts, x_ends, page_width)

        # Step 3: Validate columns (must have enough blocks)
        valid_columns = self._validate_columns(columns, blocks)

        # Step 4: Classify layout
        layout_analysis = self._classify_layout(valid_columns, blocks, page_width, page_height)

        return layout_analysis

    def _detect_columns(
        self,
        x_starts: List[float],
        x_ends: List[float],
        page_width: float
    ) -> List[Tuple[float, float]]:
        """
        Detect column boundaries by looking at x-coordinate distribution.

        Simple approach:
        - Look at where blocks start (x_starts)
        - Find distinct clusters with significant gaps between them
        - If blocks cluster around 2 distinct x positions → 2 columns
        - If blocks span similar x range → 1 column
        """
        if not x_starts:
            return []

        # Find major gaps in x-coordinates
        sorted_starts = sorted(x_starts)

        # Look for large gaps (indicating column separation)
        gaps = []
        for i in range(len(sorted_starts) - 1):
            gap_size = sorted_starts[i + 1] - sorted_starts[i]
            if gap_size > self.column_gap_threshold:
                # This is a significant gap
                mid_point = (sorted_starts[i] + sorted_starts[i + 1]) / 2
                gaps.append(mid_point)

        # If no significant gaps, it's single column
        if not gaps:
            min_x = min(x_starts)
            max_x = max(x_ends)
            return [(min_x, max_x)]

        # Build columns based on gaps
        columns = []
        prev_split = 0

        for gap in gaps:
            columns.append((prev_split, gap))
            prev_split = gap

        # Last column extends to page edge
        columns.append((prev_split, page_width))

        return columns

    def _validate_columns(
        self,
        columns: List[Tuple[float, float]],
        blocks: List[Dict[str, Any]]
    ) -> List[Tuple[float, float]]:
        """
        Validate that detected columns actually contain enough blocks.
        Filter out false positives.
        """
        valid_columns = []

        for col_start, col_end in columns:
            # Count blocks in this column
            blocks_in_column = sum(
                1 for block in blocks
                if col_start <= block['x0'] <= col_end
            )

            if blocks_in_column >= self.min_blocks_per_column:
                valid_columns.append((col_start, col_end))

        return valid_columns

    def _classify_layout(
        self,
        columns: List[Tuple[float, float]],
        blocks: List[Dict[str, Any]],
        page_width: float,
        page_height: float
    ) -> LayoutAnalysis:
        """
        Classify the layout type based on detected columns.

        Note: We don't classify as COMPLEX just because there are tables
        or images - those are normal and handled by standard parsers.
        COMPLEX is for truly irregular layouts with inconsistent column
        structures or overlapping text.
        """
        num_columns = len(columns)

        # Calculate confidence based on how clear the column structure is
        confidence = self._calculate_confidence(columns, blocks, page_width)

        # Classify based on number of columns
        # Note: Low confidence on multi-column might indicate COMPLEX,
        # but for now we keep it simple
        if num_columns == 0:
            layout_type = LayoutType.UNKNOWN
        elif num_columns == 1:
            layout_type = LayoutType.SINGLE_COLUMN
        elif num_columns == 2:
            layout_type = LayoutType.TWO_COLUMN
        elif num_columns >= 3:
            layout_type = LayoutType.MULTI_COLUMN
        else:
            # This else clause should never be hit given the logic above
            layout_type = LayoutType.COMPLEX

        # Future: Could detect COMPLEX by checking for:
        # - Very low confidence with multiple columns (irregular layout)
        # - Overlapping bounding boxes
        # - Mixed column counts within page

        # Build metadata
        metadata = {
            "page_width": page_width,
            "page_height": page_height,
            "num_blocks": len(blocks),
            "blocks_per_column": [
                sum(1 for b in blocks if col_start <= b['x0'] <= col_end)
                for col_start, col_end in columns
            ] if columns else [],
        }

        return LayoutAnalysis(
            layout_type=layout_type,
            confidence=confidence,
            num_columns=num_columns,
            column_boundaries=columns,
            metadata=metadata
        )

    def _calculate_confidence(
        self,
        columns: List[Tuple[float, float]],
        blocks: List[Dict[str, Any]],
        page_width: float
    ) -> float:
        """
        Calculate confidence score for the layout detection.

        Higher confidence when:
        - Blocks are evenly distributed across columns
        - Column widths are similar
        - Blocks align well within columns
        """
        if not columns or not blocks:
            return 0.0

        # Factor 1: Block distribution evenness (0.0 to 1.0)
        blocks_per_col = [
            sum(1 for b in blocks if col_start <= b['x0'] <= col_end)
            for col_start, col_end in columns
        ]

        if not blocks_per_col or max(blocks_per_col) == 0:
            return 0.0

        # Calculate variance in block distribution
        avg_blocks = sum(blocks_per_col) / len(blocks_per_col)
        variance = sum((x - avg_blocks) ** 2 for x in blocks_per_col) / len(blocks_per_col)
        distribution_score = 1.0 / (1.0 + variance / (avg_blocks + 1))

        # Factor 2: Column width consistency (0.0 to 1.0)
        col_widths = [col_end - col_start for col_start, col_end in columns]
        avg_width = sum(col_widths) / len(col_widths)
        width_variance = sum((w - avg_width) ** 2 for w in col_widths) / len(col_widths)
        width_score = 1.0 / (1.0 + width_variance / (avg_width + 1))

        # Combined confidence (weighted average)
        confidence = 0.6 * distribution_score + 0.4 * width_score

        return min(1.0, max(0.0, confidence))


def detect_layout(blocks: List[Dict[str, Any]], page_width: float, page_height: float) -> LayoutAnalysis:
    """
    Convenience function for layout detection - single page.

    Args:
        blocks: List of text blocks with bbox info
        page_width: Page width in points
        page_height: Page height in points

    Returns:
        LayoutAnalysis object with classification
    """
    detector = LayoutDetector()
    return detector.analyze_page(blocks, page_width, page_height)


def detect_document_layout(pdf_path: str, start_page: int, end_page: int) -> LayoutType:
    """
    Analyze a sample of pages from a PDF and return the overall layout classification.

    This is the main utility function - pass it a PDF path and page range,
    it returns a LayoutType that can be used to decide parsing strategy.

    Args:
        pdf_path: Path to PDF file
        start_page: Start page number (0-indexed)
        end_page: End page number (0-indexed, inclusive)

    Returns:
        LayoutType enum value indicating the document's layout pattern

    Example:
        layout = detect_document_layout("doc.pdf", 30, 35)
        if layout == LayoutType.TWO_COLUMN:
            # Use two-column parser
            pass
        elif layout == LayoutType.SINGLE_COLUMN:
            # Use standard parser
            pass
    """
    import fitz

    doc = fitz.open(pdf_path)
    detector = LayoutDetector()

    # Collect analyses from all pages in sample
    analyses = []

    for page_num in range(start_page, min(end_page + 1, len(doc))):
        page = doc[page_num]
        page_width = page.rect.width
        page_height = page.rect.height

        # Extract blocks
        text_dict = page.get_text("dict")
        blocks = []

        for block in text_dict["blocks"]:
            if block.get("type") == 0:  # Text block
                bbox = block.get("bbox", (0, 0, 0, 0))
                blocks.append({
                    "x0": bbox[0],
                    "y0": bbox[1],
                    "x1": bbox[2],
                    "y1": bbox[3],
                })

        # Analyze this page
        analysis = detector.analyze_page(blocks, page_width, page_height)
        analyses.append(analysis)

    doc.close()

    # Determine consensus layout type
    # Check for pattern switching (indicates COMPLEX layout)
    layout_types = [analysis.layout_type for analysis in analyses]

    if not layout_types:
        return LayoutType.UNKNOWN

    # Get unique layout types (excluding UNKNOWN)
    unique_types = set(lt for lt in layout_types if lt != LayoutType.UNKNOWN)

    # If we see 2+ different layout types, it's a COMPLEX layout
    # (pattern switching, hybrid pages, mixed column counts)
    if len(unique_types) >= 2:
        return LayoutType.COMPLEX

    # If only one type found, return it
    if unique_types:
        return list(unique_types)[0]

    # All pages were UNKNOWN
    return LayoutType.UNKNOWN


# Example usage and testing
if __name__ == "__main__":
    import fitz

    pdf_path = "/Volumes/code-bank/sandbox/qodex_parse_standalone/testdocs/Addiction Treatment Planner.pdf"

    # Test the main utility function
    print("Testing detect_document_layout utility:")
    print("=" * 60)

    # First, show per-page analysis to visualize the pattern
    print("\nPer-Page Layout Analysis:")
    print("-" * 60)

    doc = fitz.open(pdf_path)
    detector = LayoutDetector()

    for page_num in range(28, min(39, len(doc))):
        page = doc[page_num]
        page_width = page.rect.width
        page_height = page.rect.height

        # Extract blocks
        text_dict = page.get_text("dict")
        blocks = []

        for block in text_dict["blocks"]:
            if block.get("type") == 0:  # Text block
                bbox = block.get("bbox", (0, 0, 0, 0))
                blocks.append({
                    "x0": bbox[0],
                    "y0": bbox[1],
                    "x1": bbox[2],
                    "y1": bbox[3],
                })

        # Analyze this page
        analysis = detector.analyze_page(blocks, page_width, page_height)
        print(f"  Page {page_num + 1}: {analysis.layout_type.value:15s} ({analysis.num_columns} columns, confidence: {analysis.confidence:.2f})")

    doc.close()

    print("\n" + "=" * 60)

    # Test on pages 29-39 (0-indexed: 28-38)
    layout_type = detect_document_layout(pdf_path, start_page=28, end_page=38)

    print(f"\nOverall Detection: {layout_type.value.upper()}")
    print("\nRouting Decision:")
    print(f"  - {LayoutType.SINGLE_COLUMN.value}: Use standard parser")
    print(f"  - {LayoutType.TWO_COLUMN.value}: Use two-column hierarchical parser")
    print(f"  - {LayoutType.MULTI_COLUMN.value}: Use multi-column parser")
    print(f"  - {LayoutType.COMPLEX.value}: Use complex layout parser (pattern switching, hybrid pages)")
    print("=" * 60)
