from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import BytesIO
from typing import Iterable, List, Sequence, Tuple

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from .base import ParserAdapter, ParserMetadata
from ..models.common import SectionText
from ..config.settings import get_settings


@dataclass
class _SectionAccumulator:
    slug: str
    title: str
    texts: List[str] = field(default_factory=list)
    start_page: int = 1
    end_page: int = 1
    element_categories: set[str] = field(default_factory=set)
    element_count: int = 0
    # Enhanced metadata for intelligent PDF splitting
    toc_title: str | None = None
    toc_level: int | None = None
    section_type: str = "unknown"
    page_numbers: List[int] = field(default_factory=list)

    def add(self, element, text: str) -> None:
        self.texts.append(text)
        metadata = getattr(element, "metadata", None)
        page = getattr(metadata, "page_number", None)
        if page:
            if self.element_count == 0:
                self.start_page = page
            self.end_page = max(self.end_page, page)
            if page not in self.page_numbers:
                self.page_numbers.append(page)
        category = (getattr(element, "category", "") or "").lower()
        if category:
            self.element_categories.add(category)
        self.element_count += 1

    def to_section(self) -> SectionText:
        metadata = {
            "parser": "unstructured",
            "element_count": self.element_count,
            "categories": sorted(self.element_categories),
        }
        return SectionText(
            section_slug=self.slug,
            title=self.title,
            text="\n\n".join(self.texts).strip(),
            start_page=self.start_page,
            end_page=self.end_page,
            metadata=metadata,
            page_numbers=sorted(self.page_numbers) if self.page_numbers else list(range(self.start_page, self.end_page + 1)),
            toc_title=self.toc_title,
            toc_level=self.toc_level,
            section_type=self.section_type,
        )


class UnstructuredParserAdapter(ParserAdapter):
    """Parser that leverages unstructured.* partitioners for rich formats."""

    def __init__(self, *, strategy: str = "fast") -> None:
        self._strategy = strategy

    def parse(
        self,
        content: bytes | str,
        *,
        document_title: str,
        content_type: str | None = None,
    ) -> Iterable[SectionText]:
        payload = content.encode("utf-8") if isinstance(content, str) else content
        kind = self._determine_kind(content_type, payload)
        elements = self._partition_elements(kind, payload, document_title)
        sections = self._aggregate_sections(elements, document_title)
        if not sections:
            slug = self._safe_slug(document_title or "document", fallback="section-0")
            return [
                SectionText(
                    section_slug=slug,
                    title=document_title or "Document",
                    text=payload.decode("utf-8", errors="ignore"),
                    metadata={"parser": "unstructured", "strategy": self._strategy},
                )
            ]
        return sections

    def parse_with_elements(
        self,
        content: bytes | str,
        *,
        document_title: str,
        content_type: str | None = None,
    ) -> Tuple[List[SectionText], List[object]]:
        """
        Parse content and return both sections AND raw elements.

        This method preserves the raw unstructured elements (which include
        Base64-encoded images/tables) alongside the aggregated sections.
        This enables visual element extraction and contextual linking.

        Returns:
            Tuple of (sections, raw_elements)
        """
        payload = content.encode("utf-8") if isinstance(content, str) else content
        kind = self._determine_kind(content_type, payload)
        elements = self._partition_elements(kind, payload, document_title)

        # Store elements as list (we need them for visual extraction)
        elements_list = list(elements) if not isinstance(elements, list) else elements

        sections = self._aggregate_sections(elements_list, document_title)
        if not sections:
            slug = self._safe_slug(document_title or "document", fallback="section-0")
            sections = [
                SectionText(
                    section_slug=slug,
                    title=document_title or "Document",
                    text=payload.decode("utf-8", errors="ignore"),
                    metadata={"parser": "unstructured", "strategy": self._strategy},
                )
            ]

        return sections, elements_list

    # -- helpers -------------------------------------------------------------

    def _determine_kind(self, content_type: str | None, payload: bytes) -> str:
        if content_type:
            lowered = content_type.lower()
            for keyword, kind in (
                ("pdf", "pdf"),
                ("msword", "docx"),
                ("word", "docx"),
                ("presentation", "pptx"),
                ("powerpoint", "pptx"),
                ("excel", "xlsx"),
                ("spreadsheet", "xlsx"),
                ("html", "html"),
                ("markdown", "html"),
            ):
                if keyword in lowered:
                    return kind
        return self._kind_from_signature(payload)

    def _kind_from_signature(self, payload: bytes) -> str:
        if payload.startswith(b"%PDF-"):
            return "pdf"
        if payload.startswith(b"PK\x03\x04"):
            window = payload[:4096]
            if b"word/" in window:
                return "docx"
            if b"ppt/" in window:
                return "pptx"
            if b"xl/" in window:
                return "xlsx"
        snippet = payload[:256].lower()
        if snippet.startswith(b"<!doctype html") or b"<html" in snippet:
            return "html"
        return "auto"

    def _partition_elements(self, kind: str, payload: bytes, document_title: str):
        buffer = BytesIO(payload)
        filename = f"{self._safe_slug(document_title or 'document', 'document')}.{self._extension_for_kind(kind)}"
        logging.info(f"Parsing {kind} document '{document_title}' with strategy='{self._strategy}'")
        try:
            if kind == "pdf":
                # PDF parsing now handled by OpenParseAdapter
                raise RuntimeError(
                    f"PDF parsing should use OpenParseAdapter, not UnstructuredParserAdapter. "
                    f"Document '{document_title}' was incorrectly routed to unstructured."
                )
            if kind == "docx":
                from unstructured.partition.docx import partition_docx

                return partition_docx(file=buffer, metadata_filename=filename)
            if kind == "pptx":
                from unstructured.partition.pptx import partition_pptx

                return partition_pptx(file=buffer, metadata_filename=filename)
            if kind == "xlsx":
                from unstructured.partition.xlsx import partition_xlsx

                return partition_xlsx(file=buffer, metadata_filename=filename)
            if kind == "html":
                from unstructured.partition.html import partition_html

                return partition_html(file=buffer, metadata_filename=filename)
            from unstructured.partition.auto import partition

            return partition(
                file=buffer,
                metadata_filename=filename,
                strategy=self._strategy,
            )
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency errors
            logging.warning("unstructured.dependencies_missing", exc_info=True)
            if self._strategy == "fast":
                raise RuntimeError(
                    f"Unstructured partition failed with 'fast' strategy for {kind} document '{document_title}'. "
                    "Missing required dependencies for fast text extraction. "
                    "OCR fallback is disabled. Document requires manual review or alternative parser."
                ) from exc
            return self._fallback_sections(payload, document_title)
        except Exception as exc:  # pragma: no cover - fallback path
            logging.warning("unstructured.partition_failed", exc_info=True)
            if self._strategy == "fast":
                raise RuntimeError(
                    f"Unstructured partition failed with 'fast' strategy for {kind} document '{document_title}'. "
                    "Text extraction failed. OCR fallback is disabled. "
                    "Document may require OCR (hi_res strategy) or alternative parser (OpenParse). "
                    "Stopping ingestion to allow manual review."
                ) from exc
            return self._fallback_sections(payload, document_title)

    def _partition_pdf_parallel(self, payload: bytes, document_title: str, filename: str):
        """
        Partition PDF using parallel processing for speed + accuracy.

        Strategy:
        1. Split PDF into sections using PDFSectionSplitter (TOC/headers/pages)
        2. Process each section in parallel with hi_res strategy
        3. Enable image/table extraction for visual elements
        4. Aggregate all elements preserving metadata

        Args:
            payload: PDF bytes
            document_title: Document title for logging
            filename: Filename for unstructured metadata

        Returns:
            List of unstructured elements from all sections
        """
        from unstructured.partition.pdf import partition_pdf
        from .pdf_splitter import PDFSectionSplitter

        # Get parallel processing config (reuse LLM enrichment settings)
        settings = get_settings()
        max_workers = getattr(settings.jobs.worker, "llm_max_workers", 6) or 6

        # Split PDF into sections
        splitter = PDFSectionSplitter()
        pdf_sections = splitter.split_pdf(payload, document_title)

        logging.info(
            f"Split PDF '{document_title}' into {len(pdf_sections)} sections using {pdf_sections[0].section_type if pdf_sections else 'unknown'} strategy. "
            f"Processing in parallel with {max_workers} workers (hi_res + image extraction enabled)."
        )

        # For small PDFs (< 3 sections), use single-threaded to avoid overhead
        if len(pdf_sections) <= 2:
            logging.info(f"PDF '{document_title}' has {len(pdf_sections)} sections, using single-threaded processing")
            return self._partition_pdf_section(payload, 0, len(fitz.open(stream=payload, filetype="pdf")), filename)

        # Parallel processing for larger PDFs
        all_elements = []
        section_futures = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all sections for processing
            for pdf_section in pdf_sections:
                future = executor.submit(
                    self._partition_pdf_section,
                    payload,
                    pdf_section.start_page,
                    pdf_section.end_page + 1,  # partition_pdf uses exclusive end
                    filename,
                    pdf_section.toc_title,
                    pdf_section.toc_level,
                    pdf_section.section_type,
                )
                section_futures[future] = pdf_section

            # Collect results as they complete
            for future in as_completed(section_futures):
                pdf_section = section_futures[future]
                try:
                    elements = future.result()
                    logging.info(
                        f"Processed section '{pdf_section.toc_title or f'pages {pdf_section.start_page + 1}-{pdf_section.end_page + 1}'}': "
                        f"{len(elements)} elements extracted"
                    )
                    all_elements.extend(elements)
                except Exception as exc:
                    logging.error(
                        f"Failed to process PDF section '{pdf_section.toc_title or f'pages {pdf_section.start_page + 1}-{pdf_section.end_page + 1}'}' "
                        f"for document '{document_title}': {exc}",
                        exc_info=True,
                    )
                    # Continue with other sections - don't fail entire document

        logging.info(
            f"Parallel PDF processing complete for '{document_title}': "
            f"{len(all_elements)} total elements from {len(pdf_sections)} sections"
        )

        return all_elements

    def _partition_pdf_section(
        self,
        pdf_bytes: bytes,
        start_page: int,
        end_page: int,
        filename: str,
        toc_title: str | None = None,
        toc_level: int | None = None,
        section_type: str = "unknown",
    ):
        """
        Process a single PDF section with hi_res + image extraction.

        Args:
            pdf_bytes: Full PDF bytes
            start_page: Start page (0-indexed)
            end_page: End page (exclusive, for partition_pdf)
            filename: Filename for metadata
            toc_title: TOC title if from TOC
            toc_level: TOC level if from TOC
            section_type: Section type (toc_based, header_based, page_based)

        Returns:
            List of unstructured elements with enhanced metadata
        """
        from unstructured.partition.pdf import partition_pdf

        # Extract just this section's pages
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        section_pdf = fitz.open()  # Create new empty PDF

        for page_num in range(start_page, end_page):
            if page_num < len(doc):
                section_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)

        # Convert section to bytes
        section_bytes = section_pdf.tobytes()
        doc.close()
        section_pdf.close()

        # Process section with hi_res + image extraction
        buffer = BytesIO(section_bytes)
        elements = partition_pdf(
            file=buffer,
            metadata_filename=filename,
            strategy=self._strategy,
            extract_image_block_types=["Image", "Table"],  # Extract images and tables
            extract_image_block_to_payload=True,  # Get Base64 in metadata
        )

        # Enhance elements with section metadata
        for element in elements:
            if hasattr(element, "metadata"):
                # Store TOC info if available
                if toc_title:
                    element.metadata.toc_title = toc_title
                if toc_level:
                    element.metadata.toc_level = toc_level
                element.metadata.section_type = section_type

                # Adjust page numbers to be relative to full document
                if hasattr(element.metadata, "page_number") and element.metadata.page_number:
                    element.metadata.page_number += start_page

        return elements

    def _aggregate_sections(self, elements: Sequence[object], document_title: str) -> List[SectionText]:
        sections: List[SectionText] = []
        current: _SectionAccumulator | None = None
        fallback_index = 0
        for element in elements:
            text = getattr(element, "text", "") or ""
            text = text.strip()
            if not text:
                continue

            # Extract TOC metadata from element if available
            metadata = getattr(element, "metadata", None)
            toc_title = getattr(metadata, "toc_title", None) if metadata else None
            toc_level = getattr(metadata, "toc_level", None) if metadata else None
            section_type = getattr(metadata, "section_type", "unknown") if metadata else "unknown"

            if self._is_heading(element):
                if current and current.texts:
                    sections.append(current.to_section())
                slug = self._safe_slug(text, fallback=f"section-{fallback_index}")
                fallback_index += 1
                current = _SectionAccumulator(
                    slug=slug,
                    title=text,
                    toc_title=toc_title,
                    toc_level=toc_level,
                    section_type=section_type,
                )
                current.add(element, text)
                continue
            if not current:
                slug = self._safe_slug(document_title or "document", fallback=f"section-{fallback_index}")
                fallback_index += 1
                current = _SectionAccumulator(
                    slug=slug,
                    title=document_title or "Document",
                    toc_title=toc_title,
                    toc_level=toc_level,
                    section_type=section_type,
                )
            current.add(element, text)
        if current and current.texts:
            sections.append(current.to_section())
        return sections

    def _safe_slug(self, text: str, fallback: str, *, limit: int = 64) -> str:
        slug = "".join(ch if ch.isalnum() else "-" for ch in text.lower()).strip("-")
        slug = "-".join(filter(None, slug.split("-")))
        return slug[:limit] or fallback

    def _is_heading(self, element) -> bool:
        category = (getattr(element, "category", "") or "").lower()
        if category in {"title", "heading", "section"}:
            return True
        metadata = getattr(element, "metadata", None)
        if metadata and getattr(metadata, "category_depth", None) == 0:
            return True
        return False

    def _extension_for_kind(self, kind: str) -> str:
        return {
            "pdf": "pdf",
            "docx": "docx",
            "pptx": "pptx",
            "xlsx": "xlsx",
            "html": "html",
        }.get(kind, "md")

    def _fallback_sections(self, payload: bytes, title: str) -> List[SectionText]:
        try:
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(payload))
            text_parts: List[str] = []
            for page in reader.pages:
                try:
                    text_parts.append(page.extract_text() or "")
                except Exception:  # pragma: no cover - pypdf edge case
                    continue
            text = "\n".join(filter(None, text_parts)) or payload.decode("utf-8", errors="ignore")
        except Exception:
            text = payload.decode("utf-8", errors="ignore")
        slug = self._safe_slug(title or "document", fallback="section-fallback")
        return [
            SectionText(
                section_slug=slug,
                title=title or "Document",
                text=text,
                metadata={"parser": "fallback"},
            )
        ]


UNSTRUCTURED_PARSER_METADATA = ParserMetadata(
    name="unstructured",
    content_types=(
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/html",
        "text/markdown",
    ),
    requires=("unstructured",),
    description="Full-featured parser using unstructured partitioners",
)

__all__ = ["UnstructuredParserAdapter", "UNSTRUCTURED_PARSER_METADATA"]
