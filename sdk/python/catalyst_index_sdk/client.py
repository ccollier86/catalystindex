from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

from urllib import request as urllib_request
from urllib.error import HTTPError

from .models import (
    ArtifactRef,
    Chunk,
    DependencyMetrics,
    FeedbackAnalytics,
    FeedbackItemAnalytics,
    FeedbackReceipt,
    GenerationMetrics,
    GenerationResult,
    IngestionDocument,
    IngestionJob,
    IngestionJobStatus,
    IngestionJobSummary,
    IngestionMetrics,
    JobMetrics,
    LatencySummary,
    SearchDebug,
    SearchResult,
    SearchResultsEnvelope,
    SearchMetrics,
    FeedbackMetrics,
    TelemetryMetrics,
    TelemetryExporter,
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

    def get_ingestion_job_status(self, job_id: str) -> IngestionJobStatus:
        payload = self._get(f"/ingest/jobs/{job_id}/status")
        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected job status payload")
        return self._parse_ingestion_status(payload)

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
        analytics_payload = payload.get("analytics")
        analytics = (
            self._parse_feedback_analytics(analytics_payload)
            if isinstance(analytics_payload, dict)
            else None
        )
        return FeedbackReceipt(
            status=payload.get("status", "recorded"),
            positive=payload.get("positive", positive),
            chunk_ids=list(payload.get("chunk_ids", [])),
            recorded_at=payload.get("recorded_at", ""),
            comment=payload.get("comment"),
            metadata=payload.get("metadata", {}),
            analytics=analytics,
        )

    def get_feedback_analytics(self) -> FeedbackAnalytics:
        payload = self._get("/feedback/analytics")
        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected analytics payload")
        return self._parse_feedback_analytics(payload)

    def get_ingestion_status(self, job_id: str) -> IngestionJobStatus:
        payload = self._get(f"/ingest/jobs/{job_id}/status")
        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected ingestion status payload")
        return self._parse_ingestion_status(payload)

    def poll_ingestion_job(
        self,
        job_id: str,
        *,
        interval: float = 2.0,
        timeout: float = 60.0,
    ) -> IngestionJobStatus:
        deadline = time.time() + timeout
        status = self.get_ingestion_status(job_id)
        terminal = {"succeeded", "failed", "partial"}
        while status.status.lower() not in terminal and time.time() < deadline:
            time.sleep(interval)
            status = self.get_ingestion_status(job_id)
        return status

    def get_telemetry_metrics(self) -> TelemetryMetrics:
        payload = self._get("/telemetry/metrics")
        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected telemetry payload")
        return self._parse_telemetry_metrics(payload)

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

    def _parse_feedback_analytics(self, payload: Dict[str, object]) -> FeedbackAnalytics:
        return FeedbackAnalytics(
            total_positive=int(payload.get("total_positive", 0) or 0),
            total_negative=int(payload.get("total_negative", 0) or 0),
            feedback_ratio=float(payload.get("feedback_ratio", 0.0) or 0.0),
            generated_at=payload.get("generated_at", ""),
            chunks=[
                self._parse_feedback_item(item)
                for item in payload.get("chunks", [])
                if isinstance(item, dict)
            ],
            queries=[
                self._parse_feedback_item(item)
                for item in payload.get("queries", [])
                if isinstance(item, dict)
            ],
        )

    def _parse_feedback_item(self, payload: Dict[str, object]) -> FeedbackItemAnalytics:
        return FeedbackItemAnalytics(
            identifier=str(payload.get("identifier", "")),
            positive=int(payload.get("positive", 0) or 0),
            negative=int(payload.get("negative", 0) or 0),
            score=int(payload.get("score", 0) or 0),
            last_feedback_at=payload.get("last_feedback_at"),
            last_comment=payload.get("last_comment"),
        )

    def _parse_latency_summary(self, payload: Dict[str, object]) -> LatencySummary:
        return LatencySummary(
            count=int(payload.get("count", 0) or 0),
            avg=float(payload.get("avg", 0.0) or 0.0),
            p50=float(payload.get("p50", 0.0) or 0.0),
            p95=float(payload.get("p95", 0.0) or 0.0),
            max=float(payload.get("max", 0.0) or 0.0),
        )

    def _parse_telemetry_metrics(self, payload: Dict[str, object]) -> TelemetryMetrics:
        ingestion_payload = payload.get("ingestion", {}) or {}
        jobs_payload = payload.get("jobs", {}) or {}
        search_payload = payload.get("search", {}) or {}
        generation_payload = payload.get("generation", {}) or {}
        feedback_payload = payload.get("feedback", {}) or {}
        dependency_payload = payload.get("dependencies", {}) or {}
        exporter_payload = payload.get("exporter", {}) or {}
        return TelemetryMetrics(
            ingestion=IngestionMetrics(
                chunks=int(ingestion_payload.get("chunks", 0) or 0),
                latency_ms=self._parse_latency_summary(
                    ingestion_payload.get("latency_ms", {}) or {}
                ),
            ),
            jobs=JobMetrics(
                total=int(jobs_payload.get("total", 0) or 0),
                by_status={
                    str(key): int(value)
                    for key, value in (jobs_payload.get("by_status", {}) or {}).items()
                },
                failed_documents=int(jobs_payload.get("failed_documents", 0) or 0),
            ),
            search=SearchMetrics(
                requests=int(search_payload.get("requests", 0) or 0),
                economy_requests=int(search_payload.get("economy_requests", 0) or 0),
                premium_requests=int(search_payload.get("premium_requests", 0) or 0),
                latency_ms=self._parse_latency_summary(
                    search_payload.get("latency_ms", {}) or {}
                ),
            ),
            generation=GenerationMetrics(
                requests=int(generation_payload.get("requests", 0) or 0),
                latency_ms=self._parse_latency_summary(
                    generation_payload.get("latency_ms", {}) or {}
                ),
            ),
            feedback=FeedbackMetrics(
                positive=int(feedback_payload.get("positive", 0) or 0),
                negative=int(feedback_payload.get("negative", 0) or 0),
                ratio=float(feedback_payload.get("ratio", 0.0) or 0.0),
                latency_ms=self._parse_latency_summary(
                    feedback_payload.get("latency_ms", {}) or {}
                ),
            ),
            dependencies=DependencyMetrics(
                failures={
                    str(key): int(value)
                    for key, value in (dependency_payload.get("failures", {}) or {}).items()
                },
                retries={
                    str(key): int(value)
                    for key, value in (dependency_payload.get("retries", {}) or {}).items()
                },
            ),
            exporter=TelemetryExporter(
                enabled=bool(exporter_payload.get("enabled", False)),
                running=bool(exporter_payload.get("running", False)),
                address=exporter_payload.get("address"),
                port=exporter_payload.get("port"),
            ),
        )

    def _parse_ingestion_status(self, payload: Dict[str, object]) -> IngestionJobStatus:
        return IngestionJobStatus(
            job_id=payload.get("job_id", ""),
            status=payload.get("status", ""),
            documents_total=int(payload.get("documents_total", 0) or 0),
            documents_completed=int(payload.get("documents_completed", 0) or 0),
            documents_failed=int(payload.get("documents_failed", 0) or 0),
            updated_at=payload.get("updated_at", ""),
            error=payload.get("error"),
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
