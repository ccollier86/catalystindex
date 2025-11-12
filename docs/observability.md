# Observability & Dashboards

Catalyst Index exposes metrics through `/telemetry/metrics` (JSON) and optional Prometheus exporters (see `MetricsRecorder`). This guide covers both options.

## Prometheus + Grafana

1. Enable metrics exporter in settings:
   ```bash
   export CATALYST_FEATURES__enable_metrics=true
   export CATALYST_METRICS_EXPORTER_PORT=9464
   ```
2. Start the API; Prometheus can now scrape `http://<host>:9464/`.
3. Import the following Grafana dashboard JSON snippet (save to a file first):

```json
{
  "title": "Catalyst Index",
  "panels": [
    {"type": "stat", "title": "Ingestion Jobs", "targets": [{"expr": "sum(catalystindex_ingestion_jobs_total)"}]},
    {"type": "gauge", "title": "Job Failures", "targets": [{"expr": "sum(catalystindex_ingestion_job_failed_documents_total)"}]},
    {"type": "graph", "title": "Search Latency", "targets": [{"expr": "rate(catalystindex_search_latency_seconds_sum[5m]) / rate(catalystindex_search_latency_seconds_count[5m])"}]},
    {"type": "stat", "title": "Feedback Ratio", "targets": [{"expr": "sum(catalystindex_feedback_events_total{positive=\"True\"}) / sum(catalystindex_feedback_events_total)"}]}
  ]
}
```

Adjust metric names if you change the namespace.

## CLI Snapshots

Use `scripts/catalystctl.py`:

```bash
python scripts/catalystctl.py --base-url http://localhost:8000 --token <JWT> telemetry
```

The command prints the same structure returned by `/telemetry/metrics`, including `jobs.total`, `jobs.by_status`, and `jobs.failed_documents`.

## Alerts

Suggested Prometheus alerts:

- `catalystindex_ingestion_job_failed_documents_total` increasing rapidly.
- `catalystindex_dependency_failures_total{dependency="firecrawl"}` above threshold.
- Search latency histogram p95 exceeding SLA.

Add these rules to your Alertmanager configuration to get notified when ingestion or acquisition fails.
