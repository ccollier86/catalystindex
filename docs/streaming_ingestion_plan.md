# Streaming / Structured Data Ingestion Plan

## Objective
Provide a templated way to sync structured sources (databases, tables, collections) into the RAG pipeline continuously, supporting pub/sub, polling, or webhooks.

## Source Templates
1. **Database Polling**
   - Define a connector (Postgres/Mongo/etc.) with `table`, `primary_key`, `updated_at` columns.
   - Poll at interval, detect new/changed rows, generate documents (row-based or chunked by column groups).

2. **Webhooks / Pub/Sub**
   - Expose `/ingest/stream` endpoint to receive JSON payloads (e.g., CRM updates).
   - Optionally hook into Kafka/Redis Streams via a worker that consumes messages and enqueues ingestion.

3. **File Feeds**
   - Watch S3 buckets or object storage events (via SNS/SQS) and ingest new files as they land.

## Pipeline Flow
- Each source template produces `DocumentSubmission` entries.
- Reuse acquisition + ingestion queues so structured data flows through parsing/chunking like any other doc.
- Store source metadata (table name, row ID, change version) in chunk metadata for traceability.

## Configuration
- `CATALYST_STREAMING__sources` config describing each feed:
  ```yaml
  - type: postgres
    dsn: ...
    table: knowledge_base
    updated_at_column: updated_at
    interval_seconds: 60
  - type: webhook
    path: /ingest/stream/crm
    secret: ...
  ```
- Templates define how to render rows into textual sections (column mapping, templates).

## Monitoring
- Track per-source lag, last processed timestamp, and failure counts.
- Expose via telemetry (`stream_source_lag_seconds`, `stream_source_failures`).

## Implementation Steps
1. Build source registry + interface (`poll() -> [DocumentSubmission]`).
2. Implement Postgres polling connector (others can follow).
3. Add `/ingest/stream/<source>` webhook endpoint with signing secret.
4. Add streaming worker that loops over configured sources and enqueues documents.
5. Update docs/CLI to show how to enable streaming feeds.
