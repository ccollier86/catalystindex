from __future__ import annotations

import json
from typing import Dict, List, Optional

from urllib import request as urllib_request
from urllib.error import HTTPError

from .models import (
    ArtifactRef,
    Chunk,
    GenerationResult,
    IngestionDocument,
    IngestionJob,
    IngestionJobSummary,
    SearchResult,
)


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
        content: Optional[str] = None,
        schema: Optional[str] = None,
        parser_hint: Optional[str] = None,
        source_type: str = "inline",
        content_uri: Optional[str] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> IngestionJob:
        request_payload: Dict[str, object] = {
            "document_id": document_id,
            "document_title": title,
            "schema": schema,
            "parser_hint": parser_hint,
            "source_type": source_type,
            "metadata": metadata or {},
        }
        if content is not None:
            request_payload["content"] = content
        if content_uri is not None:
            request_payload["content_uri"] = content_uri
        payload = self._post("/ingest/document", request_payload)
        return self._parse_job(payload)

    def ingest_bulk(self, documents: List[Dict[str, object]]) -> IngestionJobSummary:
        payload = self._post("/ingest/bulk", {"documents": documents})
        return IngestionJobSummary(
            job_id=payload["job_id"],
            status=payload["status"],
            document_count=payload["submitted"],
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            error=payload.get("error"),
        )

    def list_ingestion_jobs(self) -> List[IngestionJobSummary]:
        payload = self._get("/ingest/jobs")
        return [
            IngestionJobSummary(
                job_id=item["job_id"],
                status=item["status"],
                document_count=item["document_count"],
                created_at=item["created_at"],
                updated_at=item["updated_at"],
                error=item.get("error"),
            )
            for item in payload
        ]

    def get_ingestion_job(self, job_id: str) -> IngestionJob:
        payload = self._get(f"/ingest/jobs/{job_id}")
        return self._parse_job(payload)

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

    def _parse_job(self, payload: Dict[str, object]) -> IngestionJob:
        documents_payload = payload.get("documents")
        if not documents_payload and payload.get("document"):
            documents_payload = [payload["document"]]
        documents = [self._parse_document(item) for item in documents_payload or []]
        return IngestionJob(
            job_id=payload["job_id"],
            status=payload["status"],
            documents=documents,
            created_at=payload.get("created_at", ""),
            updated_at=payload.get("updated_at", ""),
            error=payload.get("error"),
        )

    def _parse_document(self, payload: Dict[str, object]) -> IngestionDocument:
        artifact_payload = payload.get("artifact")
        artifact = None
        if artifact_payload:
            artifact = ArtifactRef(
                uri=artifact_payload.get("uri"),
                content_type=artifact_payload.get("content_type"),
            )
        chunks = [self._parse_chunk(item) for item in payload.get("chunks", [])]
        return IngestionDocument(
            document_id=payload["document_id"],
            status=payload["status"],
            policy=payload.get("policy"),
            chunk_count=payload.get("chunk_count", len(chunks)),
            parser=payload.get("parser"),
            artifact=artifact,
            metadata=payload.get("metadata", {}),
            chunks=chunks,
            error=payload.get("error"),
        )

    def _parse_chunk(self, payload: Dict[str, object]) -> Chunk:
        return Chunk(
            chunk_id=payload["chunk_id"],
            section_slug=payload["section_slug"],
            text=payload["text"],
            chunk_tier=payload["chunk_tier"],
            start_page=payload["start_page"],
            end_page=payload["end_page"],
            metadata=payload.get("metadata", {}),
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

    def _get(self, path: str) -> Dict[str, object] | List[Dict[str, object]]:
        req = urllib_request.Request(
            url=f"{self._base_url}{path}",
            headers={"Accept": "application/json", **self._headers()},
            method="GET",
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
