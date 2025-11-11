from __future__ import annotations

import json
from typing import Dict, List, Optional

from urllib import request as urllib_request
from urllib.error import HTTPError

from .models import Chunk, GenerationResult, IngestionJob, SearchResult


class CatalystIndexClient:
    """HTTP client for Catalyst Index services."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    def ingest_document(
        self,
        *,
        document_id: str,
        title: str,
        content: str,
        schema: str | None = None,
    ) -> IngestionJob:
        payload = self._post(
            "/ingest/document",
            {
                "document_id": document_id,
                "document_title": title,
                "content": content,
                "schema": schema,
            },
        )
        chunks = [
            Chunk(
                chunk_id=item["chunk_id"],
                section_slug=item["section_slug"],
                text=item["text"],
                chunk_tier=item["chunk_tier"],
                start_page=item["start_page"],
                end_page=item["end_page"],
                metadata=item["metadata"],
            )
            for item in payload["chunks"]
        ]
        return IngestionJob(document_id=payload["document_id"], policy=payload["policy"], chunks=chunks)

    def search(
        self,
        *,
        query: str,
        economy_mode: bool = False,
        limit: int | None = None,
        filters: Dict[str, str] | None = None,
    ) -> List[SearchResult]:
        payload = self._post(
            "/search/query",
            {
                "query": query,
                "economy_mode": economy_mode,
                "limit": limit,
                "filters": filters,
            },
        )
        return [
            SearchResult(
                chunk_id=item["chunk_id"],
                score=item["score"],
                chunk_tier=item["chunk_tier"],
                section_slug=item["section_slug"],
                track=item["track"],
            )
            for item in payload["results"]
        ]

    def generate_summary(self, *, query: str, limit: int = 6) -> GenerationResult:
        payload = self._post("/generate/summary", {"query": query, "limit": limit})
        return GenerationResult(
            summary=payload["summary"],
            citations=payload["citations"],
            chunk_count=payload["chunk_count"],
        )

    def _post(self, path: str, payload: Dict[str, object]) -> Dict[str, object]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            url=f"{self._base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json", **self._headers()},
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8") or "{}"
                return json.loads(body)
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8") or "{}"
            try:
                detail = json.loads(error_body)
            except json.JSONDecodeError:
                detail = {"detail": error_body}
            raise RuntimeError(f"Request failed ({exc.code}): {detail}") from exc


__all__ = ["CatalystIndexClient"]
