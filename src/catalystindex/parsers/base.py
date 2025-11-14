from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Iterable, Tuple

from ..models.common import SectionText


class ParserAdapter(ABC):
    """Base class for document parsers."""

    @abstractmethod
    def parse(
        self,
        content: bytes | str,
        *,
        document_title: str,
        content_type: str | None = None,
    ) -> Iterable[SectionText]:
        raise NotImplementedError


@dataclass(frozen=True)
class ParserMetadata:
    """Describes a parser's supported types and runtime requirements."""

    name: str
    content_types: Tuple[str, ...]
    requires: Tuple[str, ...] = ()
    description: str | None = None


class PlainTextParser(ParserAdapter):
    """Simple parser for plain text content."""

    def parse(
        self,
        content: bytes | str,
        *,
        document_title: str,
        content_type: str | None = None,
    ) -> Iterable[SectionText]:
        text = content.decode("utf-8") if isinstance(content, bytes) else str(content)
        yield SectionText(
            section_slug="body",
            title=document_title,
            text=text,
            metadata={"parser": "plain_text"},
        )


class HTMLParserAdapter(ParserAdapter):
    """HTML parser that sanitises markup using a lightweight stripper."""

    def parse(
        self,
        content: bytes | str,
        *,
        document_title: str,
        content_type: str | None = None,
    ) -> Iterable[SectionText]:
        html = content.decode("utf-8") if isinstance(content, bytes) else str(content)
        sanitizer = _HTMLSanitizer()
        sanitizer.feed(html)
        sanitized = sanitizer.get_text()
        yield SectionText(
            section_slug="html_body",
            title=document_title,
            text=sanitized,
            metadata={
                "parser": "html",
                "sanitized": True,
                "source_length": len(html),
            },
        )


class PDFParserStub(ParserAdapter):
    """Parser stub capturing PDF artifact references for downstream processing."""

    def parse(
        self,
        content: bytes | str,
        *,
        document_title: str,
        content_type: str | None = None,
    ) -> Iterable[SectionText]:
        artifact_pointer = content if isinstance(content, str) else "inline-pdf"
        yield SectionText(
            section_slug="pdf_artifact",
            title=document_title,
            text="",
            metadata={
                "parser": "pdf_stub",
                "artifact_reference": artifact_pointer,
                "requires": "pdfminer.six",
            },
        )


class OCRParserStub(ParserAdapter):
    """Parser stub for OCR inputs, capturing the raw artifact pointer."""

    def parse(
        self,
        content: bytes | str,
        *,
        document_title: str,
        content_type: str | None = None,
    ) -> Iterable[SectionText]:
        artifact_pointer = content if isinstance(content, str) else "inline-ocr"
        yield SectionText(
            section_slug="ocr_artifact",
            title=document_title,
            text="",
            metadata={
                "parser": "ocr_stub",
                "artifact_reference": artifact_pointer,
                "requires": "tesseract",
            },
        )


class _HTMLSanitizer(HTMLParser):
    """Simple HTML sanitizer that strips tags and collapses whitespace."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:  # pragma: no cover - exercised via parse
        text = unescape(data)
        if text.strip():
            self._parts.append(text.strip())

    def get_text(self) -> str:
        return "\n".join(self._parts)


__all__ = [
    "ParserAdapter",
    "ParserMetadata",
    "PlainTextParser",
    "HTMLParserAdapter",
    "PDFParserStub",
    "OCRParserStub",
]
