# Catalyst Index Delivery Plan

This plan organizes the remaining work into stages with explicit dependencies and notes on which workstreams can proceed in parallel.

## Stage 1 – Core Platform Foundations
**Dependencies:** None (baseline).
**Goal:** Replace mock components with production-ready abstractions described in `build_order_spec.md §§3-5` and `ingestion_pipeline_spec.md §1-4`.

Workstreams (parallel):
- **Vector Store Layer** (`build_order_spec.md:34-48`, `search_spec.md:7-65`)
  - Implement `QdrantVectorStore` (collections, payload schema, dense/sparse config).
  - Wire settings toggle so dev can keep in-memory store.
  - Add tenant filter enforcement + JWT passthrough hooks.
- **Term/Alias Persistence** (`ingestion_pipeline_spec.md:40-48`, `search_spec.md:20-37`)
  - Move `TermIndex` to Redis/Postgres backend with TTL + per-tenant namespaces.
  - Migrate existing APIs to async-safe version.
- **Parser Registry Expansion** (`ingestion_pipeline_spec.md:12-38`)
  - Add HTML parser (Firecrawl/Playwright fetch + DOM sanitizer).
  - Add PDF/OCR parser stubs with artifact references.
  - Register parser metadata (supported content types, required deps).

## Stage 2 – Ingestion Pipeline & Bulk Jobs
**Dependencies:** Stage 1 vector store + parser updates.
**Goal:** End-to-end ingestion with bulk support, retries, and artifact capture per `ingestion_pipeline_spec.md:50-90` and `logs_audits_metrics_spec.md:20-60`.

Workstreams (parallel once Stage 1 ready):
- **API Contracts** (`ingestion_pipeline_spec.md:52-62`)
  - Extend `/ingest/document` with `source_type`, `parser_hint`, `metadata` payloads.
  - Add `/ingest/bulk` + `/ingest/jobs/{id}` + list endpoint.
- **Acquisition + Artifact Store** (`ingestion_pipeline_spec.md:62-74`, `security_tenancy_spec.md:12-24`)
  - Integrate Firecrawl/HTTP fetcher; persist HTML/PDF artifacts (S3/local) with metadata.
  - Wire file reference ingestion (uploads or pointers).
- **Chunking/Enrichment Enhancements** (`ingestion_pipeline_spec.md:24-48`, `generation_spec.md:10-32`)
  - Expand policy definitions (vision, treatment, summary tiers).
  - Add LLM enrichment hooks (async summarization & key terms) with fallback heuristics.
- **Worker Orchestration & Retries** (`ingestion_pipeline_spec.md:70-88`, `logs_audits_metrics_spec.md:32-50`)
  - Introduce Celery/RQ workers, Redis queue, exponential backoff.
  - Define `IngestionJob` persistence, status tracking, retry endpoints.
  - Emit metrics/logs per job/document.

## Stage 3 – Retrieval & Search Enhancements
**Dependencies:** Stage 1 vector store (for Qdrant queries); Stage 2 alias persistence for better expansion.
**Goal:** Spec-compliant search with economy/premium modes, rerankers, and multimodal tracks as defined in `search_spec.md` and reinforced in `sdk_spec.md:6-18`.

Workstreams (parallel):
- **Search API Upgrade** (`search_spec.md:5-40`)
  - Extend request model (mode, tracks, alias controls, filters schema, debug flag).
  - Update response with metadata slices, vision assets, explanations.
- **Query Pipeline** (`search_spec.md:41-65`)
  - Intent detection to pick policies/filters per query type.
  - Configurable alias expansion + ability to disable.
- **Qdrant Retrieval Modes** (`search_spec.md:66-84`, `security_tenancy_spec.md:12-24`)
  - Implement multi-track (text/vision/hybrid) search using Qdrant filters and optional sparse vectors.
  - Add payload prefilters (policy, ICD, diagnosis, chunk tier).
- **Reranking & Fusion** (`search_spec.md:84-100`)
  - Economy path: raw Qdrant ordering.
  - Premium path: pluggable reranker (Cohere/OpenAI) and RRF fusion across tracks.

## Stage 4 – Observability, Feedback, and SDK
**Dependencies:** Stages 2–3 outputs.
**Goal:** Provide operational visibility, user feedback loops, and client tooling per `logs_audits_metrics_spec.md`, `security_tenancy_spec.md:40-55`, and `sdk_spec.md`.

Workstreams (parallel):
- **Telemetry & Audits** (`logs_audits_metrics_spec.md:30-70`)
  - Metrics for ingestion/search latency, Qdrant retries, Firecrawl failures.
  - Structured audit logs covering source_type, parser, policy, job state.
- **Feedback APIs** (`search_spec.md:78-90`, `logs_audits_metrics_spec.md:60-80`)
  - Endpoint to record positive/negative chunk feedback, feeding TermIndex + analytics.
  - Reporting hooks to surface feedback impact.
- **SDK & CLI Updates** (`sdk_spec.md:6-28`)
  - Update Python SDK + CLI to call new ingest/search endpoints (bulk submit, job status, advanced search options).
  - Provide sample notebooks/scripts for QA scenarios.

## Stage 5 – Hardening & Validation
**Dependencies:** Completion of Stages 1–4.
**Goal:** Ensure system meets spec and is production-ready, covering acceptance criteria in `build_order_spec.md:50-70`, `security_tenancy_spec.md`, and test requirements in `ingestion_pipeline_spec.md:90-110`.

Workstreams:
- Scenario + load testing (ingestion batches, search concurrency, rerank latency) per `search_spec.md:78-91`.
- Security review (JWT propagation, tenant filters, artifact access controls) referencing `security_tenancy_spec.md:10-55`.
- Documentation updates (runbooks, spec alignment tables, SDK guides) aligning with all spec docs.

---
This document should be kept in sync with implementation progress; mark tasks complete and adjust dependencies as we deliver new capabilities.
