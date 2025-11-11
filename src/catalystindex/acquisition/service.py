from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict


@dataclass(slots=True)
class AcquisitionResult:
    """Represents retrieved document content before parsing."""

    content: bytes
    content_type: str | None
    source_uri: str | None
    parser_hint: str | None
    metadata: Dict[str, object]


class AcquisitionService:
    """Fetches document sources from uploads, URLs, or inline payloads."""

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
            try:
                with urllib.request.urlopen(content_uri) as response:  # nosec - trusted runtime configuration
                    body = response.read()
                    content_type = response.headers.get_content_type()
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Failed to fetch content from {content_uri}: {exc.reason}") from exc
            metadata.update({"source_type": normalized_type, "fetched_uri": content_uri})
            inferred_parser = parser_hint
            if not inferred_parser and content_type:
                if "html" in content_type:
                    inferred_parser = "html"
                elif "pdf" in content_type:
                    inferred_parser = "pdf"
            return AcquisitionResult(
                content=body,
                content_type=content_type,
                source_uri=content_uri,
                parser_hint=inferred_parser,
                metadata=metadata,
            )
        raise ValueError(f"Unsupported source_type '{source_type}'")


__all__ = ["AcquisitionResult", "AcquisitionService"]

