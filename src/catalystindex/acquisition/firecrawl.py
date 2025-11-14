from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional

from .service import AcquisitionResult, URLFetcher


@dataclass(slots=True)
class HttpURLFetcher(URLFetcher):
    """Fallback fetcher that retrieves content directly over HTTP."""

    user_agent: str = "catalyst-index-ingestion/1.0"

    def fetch(
        self,
        url: str,
        *,
        metadata: Dict[str, object] | None = None,
        parser_hint: str | None = None,
    ) -> AcquisitionResult:
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(request) as response:  # nosec - trusted runtime configuration
                body = response.read()
                content_type = response.headers.get_content_type()
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to fetch content from {url}: {exc.reason}") from exc
        inferred_parser = parser_hint
        if not inferred_parser and content_type:
            lowered = content_type.lower()
            if "html" in lowered:
                inferred_parser = "html"
            elif "pdf" in lowered:
                inferred_parser = "pdf"
        meta = dict(metadata or {})
        meta.setdefault("payload_size", len(body))
        return AcquisitionResult(
            content=body,
            content_type=content_type,
            source_uri=url,
            parser_hint=inferred_parser,
            metadata=meta,
        )


class FirecrawlFetcher(URLFetcher):
    """Firecrawl-backed fetcher that normalizes responses into AcquisitionResult."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.firecrawl.dev",
        timeout: int = 60,
        scrape_format: str = "markdown",
    ) -> None:
        if not api_key:
            raise ValueError("FirecrawlFetcher requires a non-empty api_key")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._scrape_format = scrape_format
        self._opener = urllib.request.build_opener()

    def fetch(
        self,
        url: str,
        *,
        metadata: Dict[str, object] | None = None,
        parser_hint: str | None = None,
    ) -> AcquisitionResult:
        payload = json.dumps({"url": url, "format": self._scrape_format}).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self._base_url}/v1/scrape",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": "catalyst-index-ingestion/1.0",
            },
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                raw = response.read().decode("utf-8") or "{}"
        except urllib.error.HTTPError as exc:
            error_payload = exc.read().decode("utf-8") or exc.reason
            raise RuntimeError(f"Firecrawl request failed ({exc.code}): {error_payload}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Firecrawl request failed: {exc.reason}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise RuntimeError("Firecrawl returned invalid JSON") from exc

        document = data.get("data") or data
        content: Optional[str] = None
        if isinstance(document, dict):
            content = document.get("content") or document.get(self._scrape_format)
            content_type = (
                document.get("content_type")
                or document.get("contentType")
                or document.get("mime_type")
                or document.get("mimeType")
            )
            file_type = (document.get("file_type") or document.get("fileType") or "").lower()
            metadata_payload = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
        else:
            content_type = None
            file_type = ""
            metadata_payload = {}

        if content is None:
            raise RuntimeError("Firecrawl response did not include content")

        body = content.encode("utf-8")
        content_type = self._normalize_content_type(content_type, file_type)
        inferred_parser = parser_hint or self._parser_hint_from_metadata(content_type, file_type)
        meta = dict(metadata_payload)
        if metadata:
            meta.update(metadata)
        meta.setdefault("payload_size", len(body))
        if file_type:
            meta.setdefault("firecrawl_file_type", file_type)
        return AcquisitionResult(
            content=body,
            content_type=content_type,
            source_uri=url,
            parser_hint=inferred_parser,
            metadata=meta,
        )

    def _normalize_content_type(self, content_type: Optional[str], file_type: str) -> Optional[str]:
        if content_type:
            return content_type
        mapping = {
            "pdf": "application/pdf",
            "doc": "application/msword",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "ppt": "application/vnd.ms-powerpoint",
            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "xls": "application/vnd.ms-excel",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "html": "text/html",
        }
        if file_type and file_type in mapping:
            return mapping[file_type]
        if self._scrape_format.lower() == "markdown":
            return "text/markdown"
        if self._scrape_format.lower() == "html":
            return "text/html"
        return None

    def _parser_hint_from_metadata(self, content_type: Optional[str], file_type: str) -> Optional[str]:
        lowered = (content_type or "").lower()
        if "pdf" in lowered or file_type == "pdf":
            return "pdf"
        if "html" in lowered or file_type == "html":
            return "html"
        if any(token in lowered for token in ("msword", "word")) or file_type in {"doc", "docx"}:
            return "docx"
        if "powerpoint" in lowered or file_type in {"ppt", "pptx"}:
            return "pptx"
        if "excel" in lowered or file_type in {"xls", "xlsx"}:
            return "xlsx"
        if lowered.startswith("text/"):
            return "plain_text"
        return None


__all__ = ["FirecrawlFetcher", "HttpURLFetcher"]
