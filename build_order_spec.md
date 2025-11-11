## Build Order & Architecture Principles

This document defines the recommended implementation sequence and architecture guardrails so the platform remains composable, testable, and resilient. We explicitly reference **Single Responsibility**, **Separation of Concerns**, and **Dependency Inversion** and provide tips for honoring them at each stage.

### 1. Core Principles & Tips
1. **Single Responsibility (SRP)**
   - *Rule*: A module/class does exactly one thing (e.g., `ParserAdapter`, `ChunkingEngine`, `VectorStoreClient`).
   - *Tips*:
     - Keep public APIs narrow. If a class starts exposing unrelated methods, split it.
     - Put shared utilities (logging helpers, retries) in separate modules, injected where needed.

2. **Separation of Concerns (SoC)**
   - *Rule*: Keep layers distinct (ingestion vs search vs generation vs SDK). Data flow moves between layers via well-defined contracts.
   - *Tips*:
     - Never let SDKs embed business logic; they only call public APIs.
     - Avoid “shortcut” calls (e.g., search service should not reach into ingestion’s DB).
     - Enforce lint/tests that prevent cross-package imports outside approved boundaries.

3. **Dependency Inversion (DIP)**
   - *Rule*: High-level policies depend on abstractions, not concrete classes.
   - *Tips*:
     - Define interfaces/protocols (ParserAdapter, VectorStoreClient, LLMProvider).
     - Register implementations via dependency injection/config. Tests can swap in mocks easily.
     - When adding a new provider, update DI wiring, not consumer code.

### 2. Build Sequence

| Phase | Deliverable | Notes |
| --- | --- | --- |
| 1 | **Schema + Policy Engine** (chunking/policy) | Implement policy resolver first; everything else consumes its JSON output. |
| 2 | **Parsing Adapters** (Docling, MinerU, Firecrawl, vision models) | Each parser exposes the same `SectionText` interface. No downstream code talks to parser-specific APIs. |
| 3 | **Chunking Engine** | Uses policy + SectionText to produce chunks/metadata. Outputs files/artifacts consumed later. |
| 4 | **Ingestion Service** | Orchestrates parsing + chunking, handles caching and artifact storage. Exposes REST/gRPC endpoints. |
| 5 | **Storage Layer** | Qdrant + artifact store (S3). Provide interfaces for vector store operations so economy/premium deployments can swap implementations. |
| 6 | **Search Service** | Implements the hybrid retrieval spec (filters, rerank, vision integration). Depends only on the storage interface. |
| 7 | **Generation Service (optional)** | Simple `/generate/summary` built atop the Search service; no direct dependency on ingestion. |
| 8 | **SDKs (Python/TS)** | Wrap ingestion/search/generation APIs; no business logic beyond client ergonomics. |
| 9 | **Telemetry & Feedback** | Logging/audit/metrics per spec. Hook search + generation responses into feedback pipeline. |

### 3. Module Responsibilities & Interfaces
- **ParserAdapter** interface: `parse(document) -> List[SectionText]`. Concrete adapters: Docling, MinerU OCR, Firecrawl HTML, Vision (ColQwen/Nougat).  
  *Inversion:* chunking engine depends only on `ParserAdapter`.

- **ChunkingEngine**: `generate_chunks(sections, policy) -> List[ChunkRecord]`. No knowledge of storage; returns data structures consumed by ingestion service.

- **VectorStoreClient** interface: `upsert(chunks, embeddings)`, `query(request)`, `delete(ids)`. Implementations: Qdrant premium, Qdrant economy, future providers.

- **SearchOrchestrator**: depends on `VectorStoreClient` + `FilterBuilder` abstractions; does not “know” about HTTP or SDKs.

- **GenerationFacade**: depends on SearchOrchestrator + LLM provider interface; optional.

- **SDKs** consume public HTTP/gRPC APIs; no direct DB access.

### 4. Adapter Rules
1. Every integration point (parsers, vector store, LLM, telemetry) must have a thin adapter implementing a shared interface.
2. Adapters registered via dependency injection; tests can substitute fakes/mocks.
3. No service reaches “around” an adapter to call third-party libraries directly.

### 5. Deployment Layers
- **Core Services**: ingestion, search, generation. Each runs independently; communicate via REST/gRPC.
- **Worker Queues**: optional orchestration layer (Airflow/Temporal) talks only to the ingestion service via adapters.
- **SDKs/Apps**: call public APIs, never internal modules.

### 6. Change Management
- When introducing a new parser/vector store/reranker:
  1. Implement/extend the adapter.
  2. Update configuration/schema to reference the adapter.
  3. Add integration tests verifying interface compliance.
  4. No changes to orchestrator logic unless new capabilities require explicit flags.

### 7. Testing Strategy
- **Unit tests** around policy resolver, chunking engine, filter builders (pure functions).
- **Contract tests** for parser adapters and vector store clients.
- **End-to-end smoke tests** per phase (e.g., parse → chunk → upsert → query) before shipping.

### 8. Versioning & Releases
- Tag releases per module (ingestion-service vX.Y, search-service vX.Y). Use semantic versioning.
- SDK releases aligned with API versions; include compatibility matrix.

### 9. Rollout Checklist
1. Implement policy engine (Phase 1) + document interfaces.
2. Add parser adapters & chunking (Phase 2–3); verify artifacts.
3. Stand up ingestion service + storage (Phase 4–5).
4. Build search service w/ DIP-compliant vector store client (Phase 6).
5. Optional generation service (Phase 7).
6. Release SDKs (Phase 8).
7. Enable telemetry/feedback (Phase 9).

Following this order keeps dependencies flowing upward (high-level policies), ensures each component can be swapped without ripple effects, and leverages adapters to integrate with future tooling.

**Reminder:** during reviews, explicitly check:
- Does each new module have exactly one responsibility?
- Are cross-layer dependencies going through interfaces/adapters?
- Can we swap an implementation (parser, vector store, LLM) without touching orchestrators?
If the answer is “no” to any, refactor before moving to the next phase.
