from __future__ import annotations

import json
from typing import Dict, List, Optional

from urllib import request as urllib_request
from urllib.error import HTTPError

from .models import (
    ArtifactRef,
    Chunk,
    FeedbackReceipt,
    GenerationResult,
    IngestionDocument,
    IngestionJob,
    IngestionJobSummary,
    SearchDebug,
    SearchResult,
    SearchResultsEnvelope,
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
        mode: str = "economy",
        limit: int | None = None,
        tracks: Optional[List[Dict[str, object]]] = None,
        filters: Optional[Dict[str, object]] = None,
        alias: Optional[Dict[str, object]] = None,
        debug: bool = False,
    ) -> SearchResultsEnvelope:
        body: Dict[str, object] = {"query": query, "mode": mode, "debug": debug}
        if limit is not None:
            body["limit"] = limit
        if tracks:
            body["tracks"] = tracks
        if filters:
            body["filters"] = filters
        if alias:
            body["alias"] = alias
        payload = self._post("/search/query", body)
        results = [self._parse_search_result(item) for item in payload.get("results", [])]
        track_map = {track["name"]: track.get("retrieved", 0) for track in payload.get("tracks", [])}
        debug_payload = None
        if payload.get("debug"):
            dbg = payload["debug"]
            debug_payload = SearchDebug(
                raw_query=dbg.get("raw_query", ""),
                expanded_query=dbg.get("expanded_query", ""),
                alias_terms=list(dbg.get("alias_terms", [])),
                intent=dbg.get("intent"),
                mode=dbg.get("mode", mode),
                tracks=list(dbg.get("tracks", [])),
            )
        return SearchResultsEnvelope(
            mode=payload.get("mode", mode),
            tracks=track_map,
            results=results,
            debug=debug_payload,
        )

    def generate_summary(self, *, query: str, limit: int = 6) -> GenerationResult:
        payload = self._post("/generate/summary", {"query": query, "limit": limit})
        return GenerationResult(
            summary=payload["summary"],
            citations=payload["citations"],
            chunk_count=payload["chunk_count"],
        )

    def submit_feedback(
        self,
        *,
        query: str,
        chunk_ids: List[str],
        positive: bool = True,
        comment: Optional[str] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> FeedbackReceipt:
        body: Dict[str, object] = {
            "query": query,
            "chunk_ids": chunk_ids,
            "positive": positive,
        }
        if comment is not None:
            body["comment"] = comment
        if metadata is not None:
            body["metadata"] = metadata
        payload = self._post("/feedback", body)
        return FeedbackReceipt(
            status=payload.get("status", "recorded"),
            positive=payload.get("positive", positive),
            chunk_ids=list(payload.get("chunk_ids", [])),
            recorded_at=payload.get("recorded_at", ""),
            comment=payload.get("comment"),
            metadata=payload.get("metadata", {}),
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
                metadata=artifact_payload.get("metadata", {}),
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

    def _parse_search_result(self, payload: Dict[str, object]) -> SearchResult:
        return SearchResult(
            chunk_id=payload["chunk_id"],
            score=payload["score"],
            track=payload.get("track", "text"),
            chunk_tier=payload.get("chunk_tier", "semantic"),
            section_slug=payload.get("section_slug", ""),
            start_page=payload.get("start_page", 1),
            end_page=payload.get("end_page", 1),
            summary=payload.get("summary"),
            key_terms=list(payload.get("key_terms", [])),
            requires_previous=payload.get("requires_previous", False),
            prev_chunk_id=payload.get("prev_chunk_id"),
            confidence_note=payload.get("confidence_note"),
            bbox_pointer=payload.get("bbox_pointer"),
            metadata=payload.get("metadata", {}),
            vision_context=payload.get("vision_context"),
            explanation=payload.get("explanation"),
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
