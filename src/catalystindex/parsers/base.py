from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable
from ..models.common import SectionText


class ParserAdapter(ABC):
    """Base class for document parsers."""

    @abstractmethod
    def parse(self, content: bytes | str, *, document_title: str) -> Iterable[SectionText]:
        raise NotImplementedError


class PlainTextParser(ParserAdapter):
    """Simple parser for plain text content."""

    def parse(self, content: bytes | str, *, document_title: str) -> Iterable[SectionText]:
        text = content.decode("utf-8") if isinstance(content, bytes) else str(content)
        yield SectionText(
            section_slug="body",
            title=document_title,
            text=text,
            metadata={"parser": "plain_text"},
        )

__all__ = ["ParserAdapter", "PlainTextParser"]
