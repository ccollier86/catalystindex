# Catalyst Index Delivery Plan

This plan organizes the remaining work into stages with explicit dependencies and notes on which workstreams can proceed in parallel.

### Clickable Execution Checklist
- [ ] **Stage 1:** Complete parser registry hardening and tenant-aware term index persistence (`build_order_spec.md`, `ingestion_pipeline_spec.md`).
- [ ] **Stage 2:** Default to async ingestion with artifact durability and retry telemetry (`ingestion_pipeline_spec.md`, `logs_audits_metrics_spec.md`).
- [ ] **Stage 3:** Validate Qdrant hybrid retrieval + premium rerankers in production configs (`search_spec.md`, `security_tenancy_spec.md`).
- [ ] **Stage 4:** Ship monitoring dashboards, feedback-driven ranking, and CLI/SDK tooling (`logs_audits_metrics_spec.md`, `sdk_spec.md`).
- [ ] **Stage 5:** Finish automated testing, load/security validation, and documentation updates (`ingestion_pipeline_spec.md`, `build_order_spec.md`).

## Stage 1 – Core Platform Foundations
**Dependencies:** None (baseline).
**Goal:** Replace mock components with production-ready abstractions described in `build_order_spec.md §§3-5` and `ingestion_pipeline_spec.md §1-4`.

Execution checklist:
- [x] Configurable vector/term/artifact backends exposed through dependency wiring (`src/catalystindex/api/dependencies.py:1-220`).
- [x] App settings publish production toggles for vector store, term index, acquisition, and feature flags (`src/catalystindex/config/settings.py:1-68`).
- [ ] Harden parser registry with HTML + PDF/OCR adapters and dependency metadata (`ingestion_pipeline_spec.md:12-38`).
- [ ] Finalize Redis/Postgres term index backend with TTL + namespace isolation (`search_spec.md:20-37`).
- [ ] Document JWT passthrough and tenant filter requirements for each backend (`build_order_spec.md:34-48`).

## Stage 2 – Ingestion Pipeline & Bulk Jobs
**Dependencies:** Stage 1 vector store + parser updates.
**Goal:** End-to-end ingestion with bulk support, retries, and artifact capture per `ingestion_pipeline_spec.md:50-90` and `logs_audits_metrics_spec.md:20-60`.

Execution checklist:
- [x] API contracts for `/ingest/document`, `/ingest/bulk`, and job inspection endpoints (`src/catalystindex/api/routes.py:192-452`).
- [x] Coordinator orchestration covering acquisition, policy resolution, artifact storage, and metrics (`src/catalystindex/services/ingestion_jobs.py:641-780`).
- [ ] Make Redis/RQ dispatcher the default path with guardrails when background workers are unavailable (`ingestion_pipeline_spec.md:70-88`).
- [ ] Enforce Firecrawl/S3 dependency checks and persist remote artifacts + metadata (`ingestion_pipeline_spec.md:62-74`).
- [ ] Emit retry/backoff metrics + structured logs for per-document failures (`logs_audits_metrics_spec.md:32-50`).
- [ ] Provide CLI/SDK helpers for monitoring queued/running jobs without blocking API workers (`sdk_spec.md:12-20`).

## Stage 3 – Retrieval & Search Enhancements
**Dependencies:** Stage 1 vector store (for Qdrant queries); Stage 2 alias persistence for better expansion.
**Goal:** Spec-compliant search with economy/premium modes, rerankers, and multimodal tracks as defined in `search_spec.md` and reinforced in `sdk_spec.md:6-18`.

Execution checklist:
- [x] Expanded search API request/response models with multi-track + debug support (`src/catalystindex/api/routes.py:130-310`).
- [x] Intent-aware query pipeline with alias expansion and RRF fusion scaffolding (`src/catalystindex/services/search.py:13-210`).
- [ ] Exercise Qdrant vector store with dense+sparse payload filters across tracks, backed by integration tests (`search_spec.md:66-84`).
- [ ] Validate premium rerankers (Cohere/OpenAI) end-to-end with credential guardrails and fallbacks (`search_spec.md:84-100`).
- [ ] Document operational playbook for track configuration + tenant ACL enforcement (`security_tenancy_spec.md:18-40`).

## Stage 4 – Observability, Feedback, and SDK
**Dependencies:** Stages 2–3 outputs.
**Goal:** Provide operational visibility, user feedback loops, and client tooling per `logs_audits_metrics_spec.md`, `security_tenancy_spec.md:40-55`, and `sdk_spec.md`.

Execution checklist:
- [x] Telemetry + audit logging for ingestion/search/generation/feedback (`src/catalystindex/telemetry/logger.py:1-220`).
- [x] Feedback capture + analytics endpoints with TermIndex updates (`src/catalystindex/services/feedback.py:1-220`).
- [ ] Surface Prometheus dashboards and alerting runbooks for ingestion/search pipelines (`logs_audits_metrics_spec.md:60-90`).
- [ ] Feed feedback analytics back into alias weights + search ranking knobs (`search_spec.md:78-90`).
- [ ] Deliver CLI tooling + SDK samples for job monitoring, search debugging, and feedback loops (`sdk_spec.md:6-28`).

## Stage 5 – Hardening & Validation
**Dependencies:** Completion of Stages 1–4.
**Goal:** Ensure system meets spec and is production-ready, covering acceptance criteria in `build_order_spec.md:50-70`, `security_tenancy_spec.md`, and test requirements in `ingestion_pipeline_spec.md:90-110`.

Execution checklist:
- [ ] Restore automated regression suite (ensure `pytest` dependency available locally/CI) (`ingestion_pipeline_spec.md:90-98`).
- [ ] Run documented load & scenario tests covering async ingestion + Qdrant hybrid search (`search_spec.md:78-91`).
- [ ] Complete security review + ACL verification for Qdrant/Redis integrations (`security_tenancy_spec.md:10-55`).
- [ ] Publish updated runbooks + spec alignment addendum once validation is complete (`build_order_spec.md:60-70`).

---
This document should be kept in sync with implementation progress; mark tasks complete and adjust dependencies as we deliver new capabilities.
