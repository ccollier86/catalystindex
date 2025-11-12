# Spec Alignment Report

This report maps the current codebase to the expectations in `build_order_spec.md`, `ingestion_pipeline_spec.md`, and `search_spec.md`. Each row highlights implemented capabilities and references to automated tests or scripts that exercise the behavior.

## Build Order Spec Coverage

| Spec Area | Implementation | Validation |
| --- | --- | --- |
| Core adapters follow SRP/DIP | Parser registry, chunking engine, and vector store clients are injected via FastAPI dependencies, keeping orchestrators decoupled from concrete adapters. 【F:src/catalystindex/api/dependencies.py†L18-L204】 | Unit tests cover ingestion coordination and vector store behavior via in-memory adapters. 【F:tests/test_ingestion_service.py†L1-L118】 |
| Storage abstractions swappable | `InMemoryVectorStore` and `QdrantVectorStore` share the `VectorStoreClient` interface, enabling configuration-driven backend swaps. 【F:src/catalystindex/storage/vector_store.py†L14-L210】 | CI installs the package in editable mode and runs smoke tests against the in-memory implementation. 【F:.github/workflows/ci.yml†L1-L28】 |
| Telemetry & auditing hooks | `MetricsRecorder` and `AuditLogger` instances are shared through dependency injection so services emit structured events. 【F:src/catalystindex/telemetry/logger.py†L20-L128】【F:src/catalystindex/api/dependencies.py†L20-L84】 | Load scripts assert ingestion/search flows complete while capturing latency metrics. 【F:tests/perf/ingestion_load.py†L1-L55】【F:tests/perf/search_load.py†L1-L48】 |

## Ingestion Pipeline Spec Coverage

| Spec Area | Implementation | Validation |
| --- | --- | --- |
| Policy-driven chunking | Ingestion service resolves chunking policies and enriches chunks with summaries/key terms per policy metadata. 【F:src/catalystindex/services/ingestion.py†L31-L137】 | Unit tests assert chunk enrichment, bulk coordination, and job metadata propagation. 【F:tests/test_ingestion_service.py†L1-L118】 |
| Artifact capture & metadata | Artifact stores namespace artifacts by tenant/job and persist payload metadata for auditing. 【F:src/catalystindex/artifacts/store.py†L26-L186】 | Performance scripts store per-document metadata and fail fast if artifacts or chunks are missing. 【F:tests/perf/base.py†L88-L123】 |
| Term index population | Enriched chunks populate the term index for alias expansion during search. 【F:src/catalystindex/services/ingestion.py†L118-L138】 | Search load script primes the term index and exercises alias-driven retrieval paths. 【F:tests/perf/search_load.py†L1-L48】 |

## Search Spec Coverage

| Spec Area | Implementation | Validation |
| --- | --- | --- |
| Hybrid retrieval pipeline | Search service handles economy/premium modes, resolves tracks, and fuses results with optional rerankers. 【F:src/catalystindex/services/search.py†L69-L191】 | Unit tests cover search service behavior, filters, and reranker toggles. 【F:tests/test_search_service.py†L1-L164】 |
| Tenant-scoped queries | Qdrant adapter builds filters enforcing organization/workspace scope and track-specific retrieval. 【F:src/catalystindex/storage/vector_store.py†L205-L271】 | CI smoke tests issue both economy and premium queries against the shared context. 【F:tests/perf/search_load.py†L1-L48】【F:.github/workflows/ci.yml†L18-L28】 |
| Metrics & audit logging | Search executions record metrics and audit events, including latency and mode toggles. 【F:src/catalystindex/services/search.py†L147-L189】【F:src/catalystindex/telemetry/logger.py†L72-L117】 | Performance scripts emit summary statistics for every run, providing rapid regression detection. 【F:tests/perf/search_load.py†L1-L48】【F:tests/perf/ingestion_load.py†L1-L55】 |

## Notes
- Scenario and load coverage now runs automatically in CI, reducing manual QA burden ahead of Stage 5 sign-off. 【F:.github/workflows/ci.yml†L1-L28】
- Gaps called out in the specs (vision models, external rerankers, SDK parity) remain future work but the pipeline structure aligns with the phased delivery roadmap.
