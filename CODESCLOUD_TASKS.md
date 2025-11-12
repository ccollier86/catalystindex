# Codex Cloud Execution Tasks

This checklist translates the outstanding delivery plan work into discrete, non-overlapping tasks you can assign in Codex Cloud. Each task links to the relevant specification sections and includes a definition of done to ensure the resulting pull request merges cleanly without conflict risk.

## ✅ Completed Foundations
- [x] **Core platform wiring** — Qdrant/Redis/artifact backends and parser registry are configurable per environment.
  - Specs: [`build_order_spec.md` §§3-5](build_order_spec.md), [`ingestion_pipeline_spec.md` §§1-4](ingestion_pipeline_spec.md)
  - Verified via: `src/catalystindex/api/dependencies.py`, `src/catalystindex/config/settings.py`

## 🧱 Stage 2 · Ingestion Reliability & Bulk Operations
- [ ] **Task 2.1 — Make asynchronous ingestion the default path**
  - Steps: enable Redis/RQ dispatcher by default, add health check for worker availability, fall back gracefully with warning metrics when queue unreachable.
  - Definition of done: `/ingest/bulk` enqueues jobs automatically; synchronous fallback only used when explicitly disabled; telemetry records dispatcher outages.
  - Specs: [`ingestion_pipeline_spec.md` §§50-78](ingestion_pipeline_spec.md#L50-L78)

- [ ] **Task 2.2 — Harden remote acquisition (Firecrawl/S3) pipeline**
  - Steps: validate dependencies at startup, persist artifacts to configured store, emit retryable errors with exponential backoff, ensure job records contain artifact URIs.
  - Definition of done: remote URL/file sources succeed across retries; job status surfaces permanent vs transient failures; integration tests cover Firecrawl + S3 paths.
  - Specs: [`ingestion_pipeline_spec.md` §§79-90](ingestion_pipeline_spec.md#L79-L90)

- [ ] **Task 2.3 — Job retry metrics & monitoring**
  - Steps: extend telemetry with retry/failure counters, expose `/ingest/jobs/{id}` progress percentages, document Grafana panels.
  - Definition of done: metrics dashboard mock or JSON spec checked into repo; retries observable via Prometheus-compatible metrics; SDK exposes retry counts.
  - Specs: [`logs_audits_metrics_spec.md` §§20-60](logs_audits_metrics_spec.md#L20-L60)

## 🔍 Stage 3 · Production Retrieval & Premium Search
- [ ] **Task 3.1 — Qdrant dense+sparse hybrid rollout**
  - Steps: enable Qdrant backend in default settings, add payload filters for ICD/diagnosis metadata, cover hybrid search with integration tests and fixtures.
  - Definition of done: search requests use Qdrant in both economy/premium modes; unit tests validate payload filters and sparse vector fallback; docs updated with deployment guidance.
  - Specs: [`search_spec.md` §§40-108](search_spec.md#L40-L108)

- [ ] **Task 3.2 — External premium rerankers with guardrails**
  - Steps: integrate Cohere/OpenAI rerankers under feature flags, add credential validation and graceful degradation to embedding reranker.
  - Definition of done: premium mode requires configured reranker, otherwise automatically downgrades with audit log; regression tests cover both paths.
  - Specs: [`search_spec.md` §§109-148](search_spec.md#L109-L148)

- [ ] **Task 3.3 — Track-aware metrics and debug surfacing**
  - Steps: record per-track latency/scores, expose debug payload via SDK, document troubleshooting workflow.
  - Definition of done: telemetry distinguishes text/vision/hybrid tracks, SDK returns debug bundle behind flag, docs provide example analysis session.
  - Specs: [`search_spec.md` §§149-178](search_spec.md#L149-L178)

## 📈 Stage 4 · Feedback Loop & Observability
- [ ] **Task 4.1 — Feedback-driven relevance tuning**
  - Steps: adjust TermIndex weights and Qdrant payload scores based on feedback API, surface analytics via `/feedback/analytics`.
  - Definition of done: feedback events change future search ordering in tests; analytics endpoint exposes aggregate stats over time.
  - Specs: [`logs_audits_metrics_spec.md` §§61-92](logs_audits_metrics_spec.md#L61-L92)

- [ ] **Task 4.2 — CLI & SDK enhancements**
  - Steps: add `catalystctl` commands for job status, search debug, feedback submission; expand Python SDK with streaming updates and CLI wrappers.
  - Definition of done: CLI documented with usage examples; automated tests cover CLI argument parsing; README updated.
  - Specs: [`sdk_spec.md` §§10-75](sdk_spec.md#L10-L75)

- [ ] **Task 4.3 — Monitoring dashboards & runbooks**
  - Steps: provide Grafana dashboard JSON, document alert thresholds, create troubleshooting runbook for ingestion/search incidents.
  - Definition of done: dashboards committed under `docs/monitoring`, runbook covers common failure modes with response steps.
  - Specs: [`logs_audits_metrics_spec.md` §§93-120](logs_audits_metrics_spec.md#L93-L120)

## 🛡️ Stage 5 · Validation & Hardening
- [ ] **Task 5.1 — Restore automated test coverage**
  - Steps: add pytest to dev requirements, ensure CI pipeline runs `python -m pytest`, write regression tests for new ingestion/search flows.
  - Definition of done: CI green on a clean checkout; tests cover async ingestion, Qdrant search, feedback impact.
  - Specs: [`ingestion_pipeline_spec.md` §§91-104](ingestion_pipeline_spec.md#L91-L104)

- [ ] **Task 5.2 — Load and security validation**
  - Steps: execute load tests (bulk ingest + high-QPS search), perform JWT/Qdrant ACL review, document results in `/docs/validation`.
  - Definition of done: load/security reports checked in; open issues tracked; mitigation plan for any findings.
  - Specs: [`security_tenancy_spec.md`](security_tenancy_spec.md), [`build_order_spec.md` §6](build_order_spec.md#L120-L150)

Assign each unchecked item as an independent Codex Cloud task to ensure clean, conflict-free pull requests that collectively deliver full spec compliance.
