## Logging, Auditing, and Metrics Specification

### 1. Objectives
- Provide complete observability across ingestion, search, and generation pipelines.
- Support compliance (HIPAA-ready), tenant isolation, and forensic analysis.
- Enable the EVOLVE improvement loop via structured metrics and feedback capture.

### 2. Telemetry Streams

| Stream | Purpose | Sink |
| --- | --- | --- |
| **Operational logs** | Request metadata, timing, errors, retries | Central log system (e.g., OpenTelemetry → Elastic / Loki) |
| **Audit events** | Security-relevant actions (ingests, searches, generation, policy changes) | Append-only audit store (immutable S3 bucket or WORM DB) |
| **Metrics** | Latency, throughput, precision@k, cost tracking | Prometheus / Grafana |
| **Feedback records** | User thumbs up/down, comments | Postgres “feedback” table feeding EVOLVE jobs |

### 3. Log Schema (Operational)
```json
{
  "timestamp": "...",
  "service": "search-api",
  "request_id": "uuid",
  "tenant": { "org_id": "...", "workspace_id": "..." },
  "user_id": "...",
  "endpoint": "/search",
  "latency_ms": 183,
  "status_code": 200,
  "error": null,
  "payload_summary": {
    "query_length": 82,
    "chunks_returned": 6,
    "economy_mode": false
  }
}
```
- Mask sensitive inputs (PHI) before logging.
- Include retry counts and upstream dependencies (Unstructured, Qdrant) in `payload_summary`.

### 4. Audit Events
Tracked actions:
1. **Ingestion**: who uploaded what, policy applied, completion time, artifact checksums.
2. **Search**: query metadata (hashed text), filters, tier usage.
3. **Generation**: prompt template ID, model used, citation list.
4. **Policy updates**: schema changes, overrides, manual chunk adjustments.

Audit record example:
```json
{
  "event": "INGEST_COMPLETED",
  "timestamp": "...",
  "actor": "user_123",
  "tenant": { "org_id": "...", "workspace_id": "..." },
  "object": { "document_id": "ptsd.pdf", "policy": "dsm5" },
  "details": { "chunks": 42, "embedding_count": 84, "vision_track": false }
}
```
- Store in an immutable log (S3 object lock or dedicated audit DB).
- Provide export/report tooling for compliance reviews.

### 5. Metrics & KPIs
| Metric | Description |
| --- | --- |
| `ingestion_latency_seconds` | Distribution per document type |
| `search_latency_seconds` | Total + Qdrant sub-calls + reranker time |
| `precision_at_k`, `recall_at_k` | Per diagnosis group (from scenario tests & feedback labels) |
| `economy_vs_premium_ratio` | Traffic split for capacity planning |
| `generation_success_rate` | Error rate for `/generate/summary` |
| `feedback_positive_ratio` | % thumbs up vs down |
| `cost_per_query` | Estimated embedding + LLM spend |

Metrics exported via Prometheus/OpenTelemetry; dashboards in Grafana with tenant filters.

### 6. Alerting
- **Ingestion failures**: error rate > 5% or job backlog > threshold.
- **Search latency**: P95 > SLA (e.g., 600 ms) for 5 minutes.
- **Generation errors**: >2% failure within 10 minutes.
- **Security events**: repeated unauthorized attempts, cross-tenant access.

Alerts routed to PagerDuty/Slack with runbooks referencing log queries.

### 7. Privacy & Retention
- Redact PHI and user content in logs/audits unless tenant opts in for encrypted storage.
- Retention policies:
  - Operational logs: 30 days (hot), 12 months (cold archive).
  - Audit logs: 7 years (configurable per compliance).
  - Metrics: 13 months rolling.
- Provide tooling for tenant-level export/delete (to honor data residency requirements).

### 8. SDK Instrumentation
- SDKs emit structured logs (debug mode) and expose hooks for client apps to capture metrics.
- Provide middleware to automatically attach request IDs and propagate tracing headers.

### 9. EVOLVE Integration
- Nightly EVOLVE job consumes feedback + metrics to update term index, chunk policies, and scenario test baselines.
- Summary reports generated weekly (top failing queries, latency outliers, model costs).
