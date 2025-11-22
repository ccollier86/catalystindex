from __future__ import annotations

import logging
import mimetypes
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, runtime_checkable

try:  # pragma: no cover - optional dependency
    import magic  # type: ignore
except Exception:  # pragma: no cover
    magic = None

logger = logging.getLogger(__name__)


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

            # Handle base64-encoded content (common for API uploads)
            if isinstance(content, str):
                payload, was_base64 = self._decode_if_base64(content)
                if was_base64:
                    logger.info(f"Decoded base64 content ({len(content)} chars → {len(payload)} bytes)")
                else:
                    payload = content.encode("utf-8")
            else:
                payload = content
                was_base64 = False

            metadata.setdefault("source_type", normalized_type)
            name_hint = self._preferred_name(metadata)

            # Use provided content_type if available, otherwise detect
            provided_content_type = metadata.get("content_type")
            if provided_content_type and isinstance(provided_content_type, str):
                logger.info(f"Using provided content_type: {provided_content_type}")
                content_type = str(provided_content_type)  # Ensure it's a string
            else:
                content_type = self._detect_content_type(payload, metadata, source_hint=name_hint)
                logger.info(f"Detected content_type: {content_type} (base64={was_base64}, size={len(payload)} bytes)")

            metadata.setdefault("content_type", content_type)
            inferred_parser = parser_hint or self._parser_hint_from_content_type(content_type)
            logger.info(f"Acquisition: content_type='{content_type}', parser_hint='{inferred_parser}', source={normalized_type}")
            return AcquisitionResult(
                content=payload,
                content_type=content_type,
                source_uri=None,
                parser_hint=inferred_parser,
                metadata=metadata,
            )
        if normalized_type == "local_file":
            if content_uri:
                path = content_uri
            elif isinstance(content, str):
                path = content
            else:
                raise ValueError("local_file sources must provide a file path in content or content_uri")
            try:
                with open(path, "rb") as fh:
                    payload = fh.read()
            except FileNotFoundError as exc:
                raise ValueError(f"Local file not found: {path}") from exc
            metadata.setdefault("source_type", normalized_type)
            metadata.setdefault("filename", path.rsplit("/", 1)[-1])
            content_type = metadata.get("content_type") or self._detect_content_type(payload, metadata, source_hint=path)
            metadata.setdefault("content_type", content_type)
            inferred_parser = parser_hint or self._parser_hint_from_content_type(content_type)
            return AcquisitionResult(
                content=payload,
                content_type=content_type,
                source_uri=path,
                parser_hint=inferred_parser,
                metadata=metadata,
            )
        if normalized_type == "s3":
            if not content_uri:
                raise ValueError("s3 sources must provide content_uri (e.g., s3://bucket/key)")
            bucket, key = self._parse_s3_uri(content_uri)
            try:
                import boto3  # type: ignore
            except Exception as exc:
                raise RuntimeError("boto3 is required for s3 acquisition") from exc
            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=bucket, Key=key)
            payload = obj["Body"].read()
            metadata.setdefault("source_type", normalized_type)
            metadata.setdefault("filename", key.rsplit("/", 1)[-1])
            content_type = metadata.get("content_type") or obj.get("ContentType") or self._detect_content_type(
                payload,
                metadata,
                source_hint=key,
            )
            metadata.setdefault("content_type", content_type)
            inferred_parser = parser_hint or self._parser_hint_from_content_type(content_type)
            return AcquisitionResult(
                content=payload,
                content_type=content_type,
                source_uri=content_uri,
                parser_hint=inferred_parser,
                metadata=metadata,
            )
        if normalized_type == "url":
            if not content_uri:
                raise ValueError("URL sources must provide content_uri")
            fetcher = self._url_fetcher
            if fetcher is None:
                raise RuntimeError("URL acquisition requested but no URL fetcher is configured")
            result = self._fetch_with_fallback(fetcher, content_uri, metadata, parser_hint)
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

    def _fetch_with_fallback(
        self,
        fetcher: URLFetcher,
        url: str,
        metadata: Dict[str, object],
        parser_hint: str | None,
    ) -> AcquisitionResult:
        try:
            return fetcher.fetch(url, metadata=metadata, parser_hint=parser_hint)
        except Exception as exc:
            # Firecrawl cannot reach host-only addresses; fall back to direct HTTP in those cases.
            if fetcher.__class__.__name__ == "FirecrawlFetcher" or self._is_internal_url(url):
                from .firecrawl import HttpURLFetcher  # local import to avoid circular dependency
                http_fetcher = HttpURLFetcher()
                return http_fetcher.fetch(url, metadata=metadata, parser_hint=parser_hint)
            raise

    def _is_internal_url(self, url: str) -> bool:
        lowered = url.lower()
        return any(host in lowered for host in ("localhost", "127.0.0.1", "host.docker.internal"))

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

    def _parse_s3_uri(self, uri: str) -> tuple[str, str]:
        if not uri.lower().startswith("s3://"):
            raise ValueError(f"Invalid s3 uri: {uri}")
        parts = uri[5:].split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid s3 uri: {uri}")
        return parts[0], parts[1]

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

    def _decode_if_base64(self, content: str) -> tuple[bytes, bool]:
        """
        Attempt to decode content as base64.

        Returns:
            (decoded_bytes, was_base64) - decoded bytes and whether it was base64

        If content is base64-encoded binary data (PDF, DOCX, etc.), decode it.
        Otherwise, return content as UTF-8 bytes.
        """
        import base64

        # Quick heuristic: base64 typically has no newlines/spaces and uses base64 chars
        # Real documents are usually much larger when base64-encoded
        stripped = content.strip()
        if len(stripped) < 100:
            # Too short to be a base64-encoded document
            return content.encode("utf-8"), False

        # Check if it looks like base64: only contains base64 characters
        base64_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
        # Allow some whitespace that might be present
        sample = stripped[:1000].replace("\n", "").replace("\r", "").replace(" ", "")
        if not all(c in base64_chars for c in sample):
            # Contains non-base64 characters
            return content.encode("utf-8"), False

        # Try to decode
        try:
            decoded = base64.b64decode(stripped, validate=True)

            # Check if decoded data looks like binary format (PDF, Office docs, etc.)
            # by checking magic bytes
            if decoded.startswith(b"%PDF-"):
                logger.debug("Detected base64-encoded PDF")
                return decoded, True
            elif decoded.startswith(b"PK\x03\x04"):
                logger.debug("Detected base64-encoded ZIP/Office document")
                return decoded, True
            elif decoded[:4] in (b"\xD0\xCF\x11\xE0", b"\x50\x4B\x03\x04"):
                logger.debug("Detected base64-encoded Office document (legacy)")
                return decoded, True
            else:
                # Decoded but doesn't look like a document format we recognize
                # Might be text that happened to be valid base64
                logger.debug(f"Decoded as base64 but unrecognized format (first 16 bytes: {decoded[:16].hex()})")
                # If it decoded and is significantly different in size, trust it
                if len(decoded) < len(content) * 0.5:  # base64 is ~133% of original
                    return decoded, True
                return content.encode("utf-8"), False

        except Exception as e:
            # Not valid base64 or decode failed
            logger.debug(f"Base64 decode failed: {e}")
            return content.encode("utf-8"), False


__all__ = ["AcquisitionResult", "AcquisitionService", "URLFetcher"]
