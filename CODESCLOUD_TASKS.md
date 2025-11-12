# Codex Cloud Delivery Tasks

This plan realigns the remaining Catalyst Index milestones around the updated Stage 2–5 priorities.
Each task is sized for an independent Codex Cloud workstream with clear success criteria and
call-outs to the relevant modules/spec sections.

## Stage 2 — Ingestion Pipeline Completion
Tighten asynchronous ingestion defaults, remote acquisition, and retry observability.

### Task 2A · Promote queue-backed ingestion as the default path
- **Objective:** Make the Redis/RQ dispatcher the primary ingestion execution mode while retaining a documented synchronous fallback.
- **Key Work:**
  - Replace `InMemoryIngestionJobStore` with a Redis/Postgres-backed implementation in `src/catalystindex/services/ingestion_jobs.py` and persist job/document state (`queued`, `running`, `partial`, `succeeded`, `failed`).
  - Enable the RQ worker in production profiles (`src/catalystindex/config/settings.py` and startup wiring in `src/catalystindex/api/dependencies.py`), including dependency diagnostics at startup with actionable error messages when Redis/RQ is unavailable.
  - Update `/ingest/bulk` in `src/catalystindex/api/routes.py` to enqueue per-document jobs by default, returning job handles that poll the durable store.
  - Document the synchronous fallback path (dev/testing) in `DELIVERY_PLAN.md` and inline docstrings.
- **Specs:** `ingestion_pipeline_spec.md §§52-88`, `logs_audits_metrics_spec.md §§32-44`.

### Task 2B · Harden remote acquisition & artifact persistence
- **Objective:** Align remote acquisition with Firecrawl/S3 requirements and ensure artifacts are durably stored with metadata.
- **Key Work:**
  - Integrate the Firecrawl client within `src/catalystindex/acquisition/` to replace inline HTTP fetches, handling auth and rate limits per spec.
  - Enforce configuration validation so production profiles require Firecrawl credentials and S3 (or configured object store) endpoints; fail fast otherwise.
  - Extend `ArtifactStore` (`src/catalystindex/artifacts/store.py`) to persist URIs + metadata and write references into ingestion job records.
  - Update ingestion coordinator (`src/catalystindex/services/ingestion.py`) to surface artifact URIs via job status APIs and SDK models (`sdk/python/catalyst_index_sdk`).
  - Harden failure paths with retries and artifact cleanup according to `ingestion_pipeline_spec.md §§62-74`.
- **Specs:** `ingestion_pipeline_spec.md §§62-74`, `security_tenancy_spec.md §§12-24`.

### Task 2C · Instrument retry/backoff metrics & dashboards
- **Objective:** Provide end-to-end observability for ingestion retries, queue depth, and external failures.
- **Key Work:**
  - Expand `MetricsRecorder` (`src/catalystindex/telemetry/logger.py`) to expose Prometheus counters/histograms for retry counts, backoff durations, queue depth, Firecrawl/Qdrant failures.
  - Add a `/metrics` exporter route (FastAPI middleware or dedicated endpoint) and document scraping configuration.
  - Supply Grafana dashboard JSON + runbook under `docs/observability/` covering ingestion retry heatmaps and queue saturation alerts.
  - Update automated tests in `tests/test_telemetry.py` (new) to assert metric emission on retry scenarios.
- **Specs:** `logs_audits_metrics_spec.md §§20-60`, `ingestion_pipeline_spec.md §§70-88`.

## Stage 3 — Retrieval & Search Productionization
Finalize Qdrant-based hybrid search and premium reranking.

### Task 3A · Enable Qdrant dense+sparse fusion by default
- **Objective:** Make Qdrant the default vector backend with full payload filtering and hybrid retrieval support.
- **Key Work:**
  - Configure `QdrantVectorStore` (`src/catalystindex/storage/vector_store.py`) to create collections with dense + sparse vectors, tenant/policy/chunk-tier payload schemas, and enforce default usage via settings.
  - Update `SearchService` (`src/catalystindex/services/search.py`) to apply ICD/diagnosis filters, tier constraints, and sparse boosting with regression tests in `tests/test_search_service.py`.
  - Add integration smoke tests using dockerized Qdrant under `tests/integration/` to validate dense-only vs dense+sparse queries.
  - Remove the in-memory vector store from production settings, keeping it only for explicit dev overrides.
- **Specs:** `search_spec.md §§41-84`, `security_tenancy_spec.md §§12-24`.

### Task 3B · Production-grade premium reranker
- **Objective:** Deliver the premium search path with an external reranker, resilience hooks, and observability.
- **Key Work:**
  - Integrate Cohere ReRank or OpenAI responses in `src/catalystindex/services/search.py`, including credential validation in `src/catalystindex/config/settings.py`.
  - Implement health checks and graceful degradation (fallback to base ranking) with telemetry on degradation events.
  - Add scenario tests in `tests/test_search_service.py` (or new premium suite) to cover reranker success, timeout, and failure paths.
  - Update SDK response models to surface reranker provenance/debug metadata.
- **Specs:** `search_spec.md §§84-100`, `sdk_spec.md §§10-24`.

## Stage 4 — Observability, Feedback & SDK Experience
Close the relevance loop and equip operators with tooling.

### Task 4A · Feedback-driven relevance tuning
- **Objective:** Use feedback signals to adapt retrieval relevance and expose tuning data.
- **Key Work:**
  - Enhance `FeedbackService` (`src/catalystindex/services/feedback.py`) to aggregate feedback, update TermIndex weights, and push payload boosts into Qdrant via `QdrantVectorStore` APIs.
  - Add a debug feedback analytics endpoint (`src/catalystindex/api/routes.py`) returning aggregated scores per term/document, consumed by SDK models.
  - Ensure updates are tenant-scoped and include audit logging per `security_tenancy_spec.md`.
- **Specs:** `search_spec.md §§78-90`, `logs_audits_metrics_spec.md §§60-80`.

### Task 4B · Metrics dashboards & runbooks
- **Objective:** Provide operational dashboards and documented procedures aligned with observability specs.
- **Key Work:**
  - Publish Grafana dashboards (JSON) covering ingestion/search/generation/feedback KPIs under `docs/observability/`.
  - Write alert runbooks detailing triggers, investigation steps, and escalation paths referencing telemetry metrics.
  - Link runbooks/dashboards from `DELIVERY_PLAN.md` and README.
- **Specs:** `logs_audits_metrics_spec.md §§30-70`.

### Task 4C · CLI & SDK tooling uplift
- **Objective:** Deliver first-class tooling so operators can exercise new APIs without manual HTTP calls.
- **Key Work:**
  - Extend the Python SDK (`sdk/python/catalyst_index_sdk`) with helpers for job status polling, feedback submission, debug search, and telemetry endpoints.
  - Optionally add a CLI entry point (under `sdk/python/`) leveraging `typer` or `click` for ingest, search, feedback, and metrics commands.
  - Update documentation and examples in `sdk_spec.md` appendices and provide sample notebooks under `docs/examples/`.
- **Specs:** `sdk_spec.md §§6-28`.

## Stage 5 — Validation & Hardening
Restore automated validation and document readiness.

### Task 5A · Restore automated & scenario testing
- **Objective:** Ensure regression coverage across ingestion, retrieval, and feedback scenarios.
- **Key Work:**
  - Bundle `pytest` and integration dependencies in `pyproject.toml` / `requirements-dev.txt`; add CI workflow (GitHub Actions) invoking unit + integration suites.
  - Author end-to-end scenario tests under `tests/scenarios/` covering bulk ingestion through premium search with feedback loops.
  - Capture test data fixtures and document execution instructions in `README.md`.
- **Specs:** `ingestion_pipeline_spec.md §§90-110`, `search_spec.md §§90-100`.

### Task 5B · Security & load validation dossier
- **Objective:** Produce the final readiness package for security and performance.
- **Key Work:**
  - Conduct JWT propagation and tenant isolation review per `security_tenancy_spec.md`, documenting findings and mitigations in `docs/security_review.md`.
  - Run load tests for bulk ingestion, search concurrency, and reranker latency (scripts under `tests/perf/`), capturing metrics snapshots.
  - Summarize results, acceptance criteria, and outstanding risks in a Stage 5 validation report appended to `DELIVERY_PLAN.md`.
- **Specs:** `build_order_spec.md §§50-70`, `security_tenancy_spec.md`, `logs_audits_metrics_spec.md`.

---
These tasks assume prior Stage 1 foundations are stable. Each can proceed in parallel where noted,
with cross-task coordination on shared components (e.g., metrics exporter, Qdrant schema).
