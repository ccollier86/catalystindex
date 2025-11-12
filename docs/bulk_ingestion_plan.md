# Bulk Ingestion Orchestrator Plan

## Goals
- Accept mixed submissions (uploads + URLs + Firecrawl site crawls) in one request.
- Immediately start acquisition (download / crawl) while tracking progress.
- Use queues with retry/backoff for each stage (download, parse, embed, etc.).
- Surface job-level progress (e.g., "Downloading 5/10", "Ingested 1/10", overall % complete).

## Architecture
1. **Submission intake**
   - `/ingest/bulk` accepts an array of items: `{type: upload|url|site, source, metadata}`.
   - Create a `BulkJobRecord` with stages per document.

2. **Acquisition queue**
   - Each item enqueues an "acquire" task:
     - `upload`: file already present; mark download stage complete.
     - `url`: fetch via HTTP/Firecrawl; retry with exponential backoff.
     - `site`: Firecrawl crawl, storing each discovered document as a sub-task.
   - Update job status as downloads finish/fail.

3. **Ingestion queue**
   - Once acquisition completes for a document, enqueue the existing ingestion worker (parse → chunk → embed → index).
   - Record per-document progress (download %, ingestion %, finished).

4. **Job tracking API**
   - `GET /ingest/jobs/{id}` returns aggregated progress:
     ```json
     {
       "download": {"completed":5, "total":10, "percent":50},
       "ingestion": {"completed":1, "total":10, "percent":10},
       "overall_percent": 30,
       "documents": [...],
       "failures": [...]
     }
     ```
   - `GET /ingest/jobs/{id}/stream` (optional SSE/WebSocket) pushes updates for UI.

5. **Retries & alerts**
   - Acquisition retries (network errors, Firecrawl 5xx) with configurable intervals.
   - Ingestion retries handled by existing worker settings.
   - On repeated failure, mark document as `FAILED` and note reason.

6. **Notifications / Hooks**
   - Optional webhook per job when stages change (download complete, ingestion complete, overall done).
   - Metrics: `bulk_job_download_progress`, `bulk_job_ingestion_progress` exposed via telemetry.

## Implementation Steps
1. Extend job schema to include `download_status` and `overall_percent`.
2. Build acquisition dispatcher + worker tasks (reuse RQ with a new queue).
3. Update job APIs to merge download + ingestion progress.
4. Provide CLI command (`catalystctl bulk-status <job_id>`) to inspect live progress.
5. Add tests covering mixed submissions, retry scenarios, and progress reporting.
