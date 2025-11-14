from __future__ import annotations

import logging
from dataclasses import dataclass, field
from io import BytesIO
from typing import Iterable, List, Sequence

from .base import ParserAdapter, ParserMetadata
from ..models.common import SectionText


@dataclass
class _SectionAccumulator:
    slug: str
    title: str
    texts: List[str] = field(default_factory=list)
    start_page: int = 1
    end_page: int = 1
    element_categories: set[str] = field(default_factory=set)
    element_count: int = 0

    def add(self, element, text: str) -> None:
        self.texts.append(text)
        metadata = getattr(element, "metadata", None)
        page = getattr(metadata, "page_number", None)
        if page:
            if self.element_count == 0:
                self.start_page = page
            self.end_page = max(self.end_page, page)
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
        )


class UnstructuredParserAdapter(ParserAdapter):
    """Parser that leverages unstructured.* partitioners for rich formats."""

    def __init__(self, *, strategy: str = "hi_res") -> None:
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
        try:
            if kind == "pdf":
                from unstructured.partition.pdf import partition_pdf

                return partition_pdf(file=buffer, metadata_filename=filename)
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
            return self._fallback_sections(payload, document_title)
        except Exception:  # pragma: no cover - fallback path
            logging.warning("unstructured.partition_failed", exc_info=True)
            return self._fallback_sections(payload, document_title)

    def _aggregate_sections(self, elements: Sequence[object], document_title: str) -> List[SectionText]:
        sections: List[SectionText] = []
        current: _SectionAccumulator | None = None
        fallback_index = 0
        for element in elements:
            text = getattr(element, "text", "") or ""
            text = text.strip()
            if not text:
                continue
            if self._is_heading(element):
                if current and current.texts:
                    sections.append(current.to_section())
                slug = self._safe_slug(text, fallback=f"section-{fallback_index}")
                fallback_index += 1
                current = _SectionAccumulator(slug=slug, title=text)
                current.add(element, text)
                continue
            if not current:
                slug = self._safe_slug(document_title or "document", fallback=f"section-{fallback_index}")
                fallback_index += 1
                current = _SectionAccumulator(slug=slug, title=document_title or "Document")
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
