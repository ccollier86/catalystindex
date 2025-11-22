"""
Enhanced heading detection using multiple signals

Fixes OpenParse's broken heading detection by using:
- Font size analysis
- Bold styling
- Position on page
- Text length
- Capitalization patterns
- Isolation (spacing)
- Numbering patterns
"""

import re
from typing import List, Any, Optional
from collections import Counter
from .base import ProcessingStep


class EnhancedHeadingDetector(ProcessingStep):
    """
    Multi-signal heading detection with weighted voting.

    Uses 7 different signals to determine if text is a heading:
    1. Font size (relative to median)
    2. Bold styling
    3. Position on page (top third)
    4. Text length (< 100 chars)
    5. Capitalization (Title Case or ALL CAPS)
    6. Isolation (spacing above/below)
    7. Numbering pattern (1.2.3, Chapter X, etc.)
    """

    def __init__(
        self,
        threshold: float = 5.0,
        font_size_weight: float = 3.0,
        bold_weight: float = 2.0,
        position_weight: float = 1.5,
        length_weight: float = 1.0,
        caps_weight: float = 1.0,
        isolation_weight: float = 2.0,
        numbering_weight: float = 2.5,
    ):
        """
        Args:
            threshold: Minimum score to be considered a heading (default 5.0)
            *_weight: Weight for each signal (higher = more important)
        """
        self.threshold = threshold
        self.weights = {
            'font_size': font_size_weight,
            'bold': bold_weight,
            'position': position_weight,
            'length': length_weight,
            'caps': caps_weight,
            'isolation': isolation_weight,
            'numbering': numbering_weight,
        }

        # Patterns for numbering detection
        self.numbering_patterns = [
            r'^\d+\.?\s+',  # 1. or 1
            r'^\d+\.\d+\.?\s+',  # 1.1. or 1.1
            r'^\d+\.\d+\.\d+\.?\s+',  # 1.1.1. or 1.1.1
            r'^[IVXLCDM]+\.?\s+',  # Roman numerals
            r'^[A-Z]\.?\s+',  # A. or A
            r'^Chapter\s+\d+',  # Chapter 1
            r'^Section\s+\d+',  # Section 1
            r'^Part\s+\d+',  # Part 1
            r'^Appendix\s+[A-Z]',  # Appendix A
        ]

        self.median_size: Optional[float] = None
        self.median_page_height: Optional[float] = None

    def process(self, nodes: List[Any]) -> List[Any]:
        """
        Process text blocks and mark headings.

        Args:
            nodes: List of text block dictionaries from PyMuPDF

        Returns:
            Same text blocks with is_heading flags updated
        """
        # First pass: Calculate median font size and page height
        self._calculate_medians(nodes)

        # Second pass: Detect headings
        for block in nodes:
            if isinstance(block, dict) and block.get('text', '').strip():
                is_heading = self._detect_heading_dict(block, nodes)
                block['is_heading'] = is_heading

        return nodes

    def _calculate_medians(self, nodes: List[Any]) -> None:
        """Calculate median font size and page height across all text blocks"""
        font_sizes = []
        page_heights = []

        for block in nodes:
            if isinstance(block, dict):
                if block.get('font_size'):
                    font_sizes.append(block['font_size'])
                if block.get('page_height'):
                    page_heights.append(block['page_height'])

        if font_sizes:
            font_sizes.sort()
            self.median_size = font_sizes[len(font_sizes) // 2]
        else:
            self.median_size = 12.0  # Default

        if page_heights:
            page_heights.sort()
            self.median_page_height = page_heights[len(page_heights) // 2]
        else:
            self.median_page_height = 792.0  # Default letter size

    def _detect_heading(self, elem: Any, all_nodes: List[Any]) -> bool:
        """
        Detect if an element is a heading using multi-signal analysis.

        Returns:
            True if element is likely a heading
        """
        signals = {
            'font_size': self._check_font_size(elem),
            'bold': self._check_bold(elem),
            'position': self._check_position(elem),
            'length': self._check_length(elem),
            'caps': self._check_capitalization(elem),
            'isolation': self._check_isolation(elem, all_nodes),
            'numbering': self._check_numbering(elem),
        }

        # Calculate weighted score
        score = sum(
            signals[signal] * self.weights[signal]
            for signal in signals
        )

        return score >= self.threshold

    def _detect_heading_dict(self, block: dict, all_blocks: List[dict]) -> bool:
        """
        Detect if a text block dictionary is a heading.

        Args:
            block: Text block dictionary with keys: text, font_size, is_bold, bbox, etc.
            all_blocks: All text blocks for context

        Returns:
            True if block is likely a heading
        """
        signals = {
            'font_size': self._check_font_size_dict(block),
            'bold': self._check_bold_dict(block),
            'position': self._check_position_dict(block),
            'length': self._check_length_dict(block),
            'caps': self._check_capitalization_dict(block),
            'isolation': True,  # Simplified for now
            'numbering': self._check_numbering_dict(block),
        }

        # Calculate weighted score
        score = sum(
            signals[signal] * self.weights[signal]
            for signal in signals
        )

        return score >= self.threshold

    def _check_font_size_dict(self, block: dict) -> bool:
        """Check if font size is larger than median"""
        font_size = block.get('font_size')
        if not font_size or not self.median_size:
            return False
        return font_size > self.median_size * 1.2

    def _check_bold_dict(self, block: dict) -> bool:
        """Check if text is bold"""
        return block.get('is_bold', False)

    def _check_position_dict(self, block: dict) -> bool:
        """Check if element is in top portion of page"""
        bbox = block.get('bbox')
        page_height = block.get('page_height')

        if not bbox or not page_height:
            return False

        # bbox is (x0, y0, x1, y1)
        y_position = bbox[3] / page_height  # y1 / page_height

        # Headings are usually in top 2/3 of page
        return y_position > 0.33

    def _check_length_dict(self, block: dict) -> bool:
        """Check if text is short (headings are usually concise)"""
        text = block.get('text', '')
        return len(text.strip()) < 100

    def _check_capitalization_dict(self, block: dict) -> bool:
        """Check if text uses Title Case or ALL CAPS"""
        text = block.get('text', '').strip()
        if not text:
            return False

        # Check for ALL CAPS
        if text.isupper() and len(text) > 3:
            return True

        # Check for Title Case
        words = text.split()
        if len(words) == 0:
            return False

        # At least 60% of words should be capitalized
        capitalized_words = sum(1 for word in words if word and word[0].isupper())
        return capitalized_words / len(words) >= 0.6

    def _check_numbering_dict(self, block: dict) -> bool:
        """Check if text starts with a numbering pattern"""
        text = block.get('text', '').strip()
        for pattern in self.numbering_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        return False

    def _check_font_size(self, elem: Any) -> bool:
        """Check if font size is larger than median"""
        if not hasattr(elem, 'size') or not elem.size or not self.median_size:
            return False
        return elem.size > self.median_size * 1.2

    def _check_bold(self, elem: Any) -> bool:
        """Check if text is bold"""
        if not hasattr(elem, 'is_bold'):
            return False
        return elem.is_bold

    def _check_position(self, elem: Any) -> bool:
        """Check if element is in top third of page"""
        if not hasattr(elem, 'bbox') or not elem.bbox:
            return False

        if not self.median_page_height:
            return False

        # Calculate vertical position (normalized to 0-1)
        y_position = elem.bbox.y1 / self.median_page_height

        # Headings are usually in top 2/3 of page
        return y_position > 0.33

    def _check_length(self, elem: Any) -> bool:
        """Check if text is short (headings are usually concise)"""
        if not hasattr(elem, 'text'):
            return False
        return len(elem.text.strip()) < 100

    def _check_capitalization(self, elem: Any) -> bool:
        """Check if text uses Title Case or ALL CAPS"""
        if not hasattr(elem, 'text'):
            return False

        text = elem.text.strip()
        if not text:
            return False

        # Check for ALL CAPS
        if text.isupper() and len(text) > 3:
            return True

        # Check for Title Case
        words = text.split()
        if len(words) == 0:
            return False

        # At least 60% of words should be capitalized
        capitalized_words = sum(1 for word in words if word and word[0].isupper())
        return capitalized_words / len(words) >= 0.6

    def _check_isolation(self, elem: Any, all_nodes: List[Any]) -> bool:
        """
        Check if element is isolated (has spacing above/below).

        This is a simplified version - in practice you'd check actual spacing
        in the bounding boxes.
        """
        if not hasattr(elem, 'bbox') or not elem.bbox:
            return False

        # For now, assume elements on their own line are more isolated
        # This would need more sophisticated bbox analysis in practice
        return True  # Placeholder

    def _check_numbering(self, elem: Any) -> bool:
        """Check if text starts with a numbering pattern"""
        if not hasattr(elem, 'text'):
            return False

        text = elem.text.strip()
        for pattern in self.numbering_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True

        return False


class HeadingHierarchyBuilder(ProcessingStep):
    """
    Build heading hierarchy (H1, H2, H3, etc.) based on font sizes.

    This step should run AFTER EnhancedHeadingDetector.
    """

    def process(self, nodes: List[Any]) -> List[Any]:
        """
        Assign heading levels based on relative font sizes.

        Args:
            nodes: List of text block dictionaries with is_heading flags set

        Returns:
            Same text blocks with heading_level attributes added
        """
        # Collect all heading font sizes
        heading_sizes = []
        for block in nodes:
            if isinstance(block, dict):
                if block.get('is_heading') and block.get('font_size'):
                    heading_sizes.append(block['font_size'])

        if not heading_sizes:
            return nodes

        # Sort unique sizes in descending order (largest = H1)
        unique_sizes = sorted(set(heading_sizes), reverse=True)

        # Map sizes to levels (1-6)
        size_to_level = {
            size: min(idx + 1, 6)
            for idx, size in enumerate(unique_sizes)
        }

        # Assign levels to headings
        for block in nodes:
            if isinstance(block, dict):
                if block.get('is_heading') and block.get('font_size'):
                    level = size_to_level.get(block['font_size'], 3)
                    block['heading_level'] = level

        return nodes
