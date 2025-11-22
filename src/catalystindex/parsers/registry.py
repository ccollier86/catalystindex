from __future__ import annotations

from typing import Dict

from .base import (
    HTMLParserAdapter,
    OCRParserStub,
    ParserAdapter,
    ParserMetadata,
    PlainTextParser,
)
from .unstructured_adapter import UNSTRUCTURED_PARSER_METADATA, UnstructuredParserAdapter
from .qodex_adapter import QODEX_PARSER_METADATA, QodexParserAdapter
# Keep OpenParse available for backwards compatibility (deprecated)
from .openparse_adapter import OPENPARSE_PARSER_METADATA, OpenParseAdapter


class ParserRegistry:
    """Registry that maps parser names to adapters."""

    def __init__(self) -> None:
        self._parsers: Dict[str, ParserAdapter] = {}
        self._metadata: Dict[str, ParserMetadata] = {}

    def register(self, name: str, adapter: ParserAdapter, *, metadata: ParserMetadata | None = None) -> None:
        self._parsers[name] = adapter
        if metadata:
            self._metadata[name] = metadata

    def resolve(self, name: str) -> ParserAdapter:
        try:
            return self._parsers[name]
        except KeyError as exc:
            raise KeyError(f"Parser '{name}' is not registered") from exc

    def metadata(self, name: str) -> ParserMetadata | None:
        return self._metadata.get(name)

    def list_parsers(self) -> Dict[str, ParserMetadata]:
        return dict(self._metadata)

def default_registry() -> ParserRegistry:
    registry = ParserRegistry()
    registry.register(
        "plain_text",
        PlainTextParser(),
        metadata=ParserMetadata(
            name="plain_text",
            content_types=("text/plain",),
            description="Plain text ingestion with minimal processing",
        ),
    )
    registry.register(
        "html",
        HTMLParserAdapter(),
        metadata=ParserMetadata(
            name="html",
            content_types=("text/html",),
            requires=("firecrawl", "playwright"),
            description="HTML parser that sanitizes DOM fetched via Firecrawl/Playwright",
        ),
    )
    # Use Qodex-Parse for PDF parsing (universal + modern)
    # IMPORTANT: Use mode="full" for enhanced metadata (keywords, search_terms, semantic neighbors)
    import os
    qodex_adapter = QodexParserAdapter(
        mode="full",  # Full mode with LLM-powered metadata
        openai_key=os.getenv("OPENAI_API_KEY"),
    )
    registry.register("pdf", qodex_adapter, metadata=QODEX_PARSER_METADATA)

    # Keep OpenParse available for backwards compatibility (deprecated)
    openparse_adapter = OpenParseAdapter()
    registry.register("openparse", openparse_adapter, metadata=OPENPARSE_PARSER_METADATA)

    # Keep unstructured for other formats
    unstructured_adapter = UnstructuredParserAdapter()
    registry.register("unstructured", unstructured_adapter, metadata=UNSTRUCTURED_PARSER_METADATA)
    registry.register(
        "docx",
        UnstructuredParserAdapter(strategy="fast"),
        metadata=ParserMetadata(
            name="docx",
            content_types=(
                "application/msword",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            requires=("unstructured",),
            description="DOC/DOCX parser using unstructured partition",
        ),
    )
    registry.register(
        "pptx",
        UnstructuredParserAdapter(strategy="fast"),
        metadata=ParserMetadata(
            name="pptx",
            content_types=(
                "application/vnd.ms-powerpoint",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ),
            requires=("unstructured",),
            description="PPT parser using unstructured partition",
        ),
    )
    registry.register(
        "xlsx",
        UnstructuredParserAdapter(strategy="fast"),
        metadata=ParserMetadata(
            name="xlsx",
            content_types=(
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
            requires=("unstructured",),
            description="Spreadsheet parser using unstructured partition",
        ),
    )
    registry.register(
        "ocr",
        OCRParserStub(),
        metadata=ParserMetadata(
            name="ocr",
            content_types=("image/png", "image/jpeg"),
            requires=("tesseract", "pytesseract"),
            description="OCR parser stub capturing scanned document artifacts",
        ),
    )
    return registry


__all__ = ["ParserRegistry", "default_registry"]
