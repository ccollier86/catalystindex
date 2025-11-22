"""PDF section splitting using TOC extraction and intelligent fallbacks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

logger = logging.getLogger(__name__)


@dataclass
class PDFSection:
    """Represents a section of a PDF with page range and metadata."""

    start_page: int  # 0-indexed
    end_page: int  # 0-indexed, inclusive
    toc_title: str | None
    toc_level: int | None  # 1=chapter, 2=subsection, etc.
    section_type: str  # "toc_based" | "header_based" | "page_based"
    page_count: int


class PDFSectionSplitter:
    """Intelligently split PDFs into sections using TOC, headers, or pages."""

    def __init__(
        self,
        *,
        max_pages_per_section: int = 10,
        min_pages_per_section: int = 1,
    ) -> None:
        """
        Initialize the PDF section splitter.

        Args:
            max_pages_per_section: Maximum pages per section for page-based fallback
            min_pages_per_section: Minimum pages per section
        """
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) is required for PDF section splitting. Install with: pip install pymupdf")

        self._max_pages_per_section = max_pages_per_section
        self._min_pages_per_section = min_pages_per_section

    def split_pdf(self, pdf_bytes: bytes, document_title: str) -> List[PDFSection]:
        """
        Split PDF into sections using intelligent strategy selection.

        Strategy priority:
        1. TOC extraction (if available)
        2. Font-based header detection (if no TOC)
        3. Adaptive page-level sectioning (fallback)

        Args:
            pdf_bytes: PDF file content as bytes
            document_title: Document title for logging

        Returns:
            List of PDFSection objects representing the split
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)

        logger.info(
            f"Splitting PDF '{document_title}' ({total_pages} pages) using intelligent section detection"
        )

        # Strategy 1: Try TOC extraction
        toc = doc.get_toc()
        if toc and len(toc) > 0:
            sections = self._split_by_toc(toc, total_pages, document_title)
            if sections:
                logger.info(
                    f"Successfully extracted {len(sections)} sections from TOC for '{document_title}'"
                )
                doc.close()
                return sections

        # Strategy 2: Try font-based header detection
        logger.info(f"No TOC found for '{document_title}', attempting font-based header detection")
        sections = self._split_by_headers(doc, document_title)
        if sections and len(sections) > 1:
            logger.info(
                f"Detected {len(sections)} sections using font-based headers for '{document_title}'"
            )
            doc.close()
            return sections

        # Strategy 3: Fallback to adaptive page-level sectioning
        logger.info(
            f"No headers detected for '{document_title}', using adaptive page-level sectioning"
        )
        sections = self._split_by_pages(total_pages, document_title)
        doc.close()
        return sections

    def _split_by_toc(
        self, toc: List[Tuple], total_pages: int, document_title: str
    ) -> List[PDFSection]:
        """
        Split PDF by Table of Contents entries.

        TOC format from PyMuPDF: [(level, title, page), ...]
        where page is 1-indexed.

        Args:
            toc: TOC entries from PyMuPDF
            total_pages: Total number of pages in document
            document_title: Document title for logging

        Returns:
            List of PDFSection objects
        """
        sections: List[PDFSection] = []

        # Filter to top-level TOC entries (level 1 and 2)
        # This prevents over-segmentation from deep subsections
        filtered_toc = [entry for entry in toc if entry[0] <= 2]

        if not filtered_toc:
            logger.warning(f"TOC for '{document_title}' has no level 1-2 entries, falling back")
            return []

        for i, entry in enumerate(filtered_toc):
            level, title, start_page_1indexed = entry
            start_page = start_page_1indexed - 1  # Convert to 0-indexed

            # Determine end page (start of next section - 1, or last page)
            if i + 1 < len(filtered_toc):
                end_page = filtered_toc[i + 1][2] - 2  # Next section start - 1, convert to 0-indexed
            else:
                end_page = total_pages - 1

            # Skip invalid ranges
            if start_page >= total_pages or end_page < start_page:
                logger.warning(
                    f"Skipping invalid TOC entry for '{document_title}': '{title}' (pages {start_page}-{end_page})"
                )
                continue

            page_count = end_page - start_page + 1

            sections.append(
                PDFSection(
                    start_page=start_page,
                    end_page=end_page,
                    toc_title=title.strip() if title else None,
                    toc_level=level,
                    section_type="toc_based",
                    page_count=page_count,
                )
            )

        # Validate coverage - ensure we cover the whole document
        if sections:
            if sections[0].start_page > 0:
                # Add intro section
                sections.insert(
                    0,
                    PDFSection(
                        start_page=0,
                        end_page=sections[0].start_page - 1,
                        toc_title="Introduction",
                        toc_level=1,
                        section_type="toc_based",
                        page_count=sections[0].start_page,
                    ),
                )

        return sections

    def _split_by_headers(self, doc: fitz.Document, document_title: str) -> List[PDFSection]:
        """
        Split PDF by detecting headers using font size analysis.

        Looks for text with larger font sizes than surrounding text,
        which typically indicates section headers.

        Args:
            doc: PyMuPDF document object
            document_title: Document title for logging

        Returns:
            List of PDFSection objects
        """
        headers: List[Tuple[int, str, float]] = []  # (page_num, text, font_size)

        # Analyze first 50 pages to find headers (performance optimization)
        max_pages_to_analyze = min(50, len(doc))

        for page_num in range(max_pages_to_analyze):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if "lines" not in block:
                    continue

                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        font_size = span["size"]

                        # Heuristics for headers:
                        # - Font size > 12pt
                        # - Short text (< 100 chars)
                        # - Not all caps (those are usually footers/headers)
                        # - Has some uppercase letters
                        if (
                            font_size > 12
                            and len(text) < 100
                            and not text.isupper()
                            and any(c.isupper() for c in text)
                        ):
                            headers.append((page_num, text, font_size))

        if not headers:
            logger.info(f"No font-based headers detected for '{document_title}'")
            return []

        # Filter headers - keep only those with font size in top 30%
        if len(headers) > 3:
            font_sizes = sorted([h[2] for h in headers], reverse=True)
            threshold = font_sizes[int(len(font_sizes) * 0.3)]
            headers = [h for h in headers if h[2] >= threshold]

        # Create sections from headers
        sections: List[PDFSection] = []
        total_pages = len(doc)

        for i, (page_num, header_text, _) in enumerate(headers):
            start_page = page_num

            # Determine end page
            if i + 1 < len(headers):
                end_page = headers[i + 1][0] - 1
            else:
                end_page = total_pages - 1

            if end_page < start_page:
                continue

            page_count = end_page - start_page + 1

            sections.append(
                PDFSection(
                    start_page=start_page,
                    end_page=end_page,
                    toc_title=header_text,
                    toc_level=1,  # All detected headers treated as level 1
                    section_type="header_based",
                    page_count=page_count,
                )
            )

        return sections

    def _split_by_pages(self, total_pages: int, document_title: str) -> List[PDFSection]:
        """
        Adaptive page-level sectioning fallback.

        Splits document into sections based on size:
        - Small docs (< 50 pages): 5 pages per section
        - Medium docs (50-150 pages): 10 pages per section
        - Large docs (> 150 pages): 15 pages per section

        Args:
            total_pages: Total number of pages
            document_title: Document title for logging

        Returns:
            List of PDFSection objects
        """
        # Adaptive section size based on document length
        if total_pages < 50:
            pages_per_section = 5
        elif total_pages < 150:
            pages_per_section = 10
        else:
            pages_per_section = 15

        # Respect configured limits
        pages_per_section = max(
            self._min_pages_per_section,
            min(pages_per_section, self._max_pages_per_section),
        )

        sections: List[PDFSection] = []
        section_num = 1

        for start_page in range(0, total_pages, pages_per_section):
            end_page = min(start_page + pages_per_section - 1, total_pages - 1)
            page_count = end_page - start_page + 1

            sections.append(
                PDFSection(
                    start_page=start_page,
                    end_page=end_page,
                    toc_title=f"Section {section_num} (pages {start_page + 1}-{end_page + 1})",
                    toc_level=1,
                    section_type="page_based",
                    page_count=page_count,
                )
            )
            section_num += 1

        logger.info(
            f"Created {len(sections)} page-based sections for '{document_title}' "
            f"({pages_per_section} pages per section)"
        )

        return sections


__all__ = ["PDFSection", "PDFSectionSplitter"]
