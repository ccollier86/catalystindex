from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol, runtime_checkable


@dataclass(slots=True)
class AcquisitionResult:
    """Represents retrieved document content before parsing."""

    content: bytes
    content_type: str | None
    source_uri: str | None
    parser_hint: str | None
    metadata: Dict[str, object]


@runtime_checkable
class URLFetcher(Protocol):
    """Protocol describing URL-based acquisition backends."""

    def fetch(
        self,
        url: str,
        *,
        metadata: Dict[str, object] | None = None,
        parser_hint: str | None = None,
    ) -> AcquisitionResult:
        ...


class AcquisitionService:
    """Fetches document sources from uploads, URLs, or inline payloads."""

    def __init__(self, url_fetcher: URLFetcher | None = None) -> None:
        self._url_fetcher = url_fetcher

    def acquire(
        self,
        *,
        source_type: str,
        content: bytes | str | None,
        content_uri: str | None,
        metadata: Dict[str, object] | None = None,
        parser_hint: str | None = None,
    ) -> AcquisitionResult:
        normalized_type = (source_type or "inline").lower()
        metadata = dict(metadata or {})
        if normalized_type in {"inline", "upload"}:
            if content is None:
                raise ValueError("Inline or upload sources must include content")
            payload = content.encode("utf-8") if isinstance(content, str) else content
            metadata.setdefault("source_type", normalized_type)
            return AcquisitionResult(
                content=payload,
                content_type=metadata.get("content_type") or "text/plain",
                source_uri=None,
                parser_hint=parser_hint,
                metadata=metadata,
            )
        if normalized_type == "url":
            if not content_uri:
                raise ValueError("URL sources must provide content_uri")
            fetcher = self._url_fetcher
            if fetcher is None:
                raise RuntimeError("URL acquisition requested but no URL fetcher is configured")
            result = fetcher.fetch(content_uri, metadata=metadata, parser_hint=parser_hint)
            result.metadata.setdefault("source_type", normalized_type)
            result.metadata.setdefault("fetched_uri", content_uri)
            if result.source_uri is None:
                result.source_uri = content_uri
            return result
        raise ValueError(f"Unsupported source_type '{source_type}'")


__all__ = ["AcquisitionResult", "AcquisitionService", "URLFetcher"]

