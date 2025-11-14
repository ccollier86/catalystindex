from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, runtime_checkable

try:  # pragma: no cover - optional dependency
    import magic  # type: ignore
except Exception:  # pragma: no cover
    magic = None


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
            name_hint = self._preferred_name(metadata)
            content_type = metadata.get("content_type") or self._detect_content_type(
                payload,
                metadata,
                source_hint=name_hint,
            )
            metadata.setdefault("content_type", content_type)
            inferred_parser = parser_hint or self._parser_hint_from_content_type(content_type)
            return AcquisitionResult(
                content=payload,
                content_type=content_type,
                source_uri=None,
                parser_hint=inferred_parser,
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
            if not result.content_type:
                result.content_type = self._detect_content_type(
                    result.content,
                    result.metadata,
                    source_hint=content_uri,
                )
            if not result.parser_hint:
                result.parser_hint = self._parser_hint_from_content_type(result.content_type)
            return result
        raise ValueError(f"Unsupported source_type '{source_type}'")

    def _detect_content_type(
        self,
        payload: bytes,
        metadata: Dict[str, object],
        *,
        source_hint: str | None = None,
    ) -> str:
        filename = source_hint or self._preferred_name(metadata)
        if filename:
            guess, _ = mimetypes.guess_type(filename)
            if guess:
                return guess
        signature_guess = self._guess_from_signatures(payload)
        if signature_guess:
            return signature_guess
        if magic:  # pragma: no branch - depends on optional package
            try:
                detected = magic.from_buffer(payload, mime=True)
                if detected:
                    return detected
            except Exception:  # pragma: no cover - safety
                pass
        text_sample = payload[:256]
        if self._looks_like_html(text_sample):
            return "text/html"
        if self._looks_like_text(text_sample):
            return "text/plain"
        return "application/octet-stream"

    def _preferred_name(self, metadata: Dict[str, object]) -> Optional[str]:
        for key in ("filename", "file_name", "original_filename"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    def _guess_from_signatures(self, payload: bytes) -> Optional[str]:
        if payload.startswith(b"%PDF-"):
            return "application/pdf"
        if payload.startswith(b"PK\x03\x04"):
            # Office files are ZIPs with identifying folders
            slice_window = payload[:4096]
            if b"word/" in slice_window:
                return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            if b"ppt/" in slice_window:
                return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            if b"xl/" in slice_window:
                return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            return "application/zip"
        if payload.startswith(b"<!DOCTYPE html") or payload.lstrip().lower().startswith(b"<html"):
            return "text/html"
        return None

    def _looks_like_text(self, sample: bytes) -> bool:
        return all(32 <= byte <= 126 or byte in {9, 10, 13} for byte in sample)

    def _looks_like_html(self, sample: bytes) -> bool:
        lowered = sample.lower()
        return b"<html" in lowered or b"<body" in lowered

    def _parser_hint_from_content_type(self, content_type: Optional[str]) -> Optional[str]:
        if not content_type:
            return None
        lowered = content_type.lower()
        if "pdf" in lowered:
            return "pdf"
        if "html" in lowered:
            return "html"
        if "word" in lowered or "msword" in lowered:
            return "docx"
        if "presentation" in lowered or "powerpoint" in lowered:
            return "pptx"
        if "excel" in lowered or "spreadsheet" in lowered:
            return "xlsx"
        if lowered.startswith("text/"):
            return "plain_text"
        return None


__all__ = ["AcquisitionResult", "AcquisitionService", "URLFetcher"]
