## Catalyst Index Upgrade Plan

### A. Repository Audit & Baseline
1. ❌ Inspect current Dockerfile/compose output and verify real FastAPI/Pydantic packages are bundled (stubs removed; pyproject pins prod deps).  
2. ❌ Map current service wiring (`src/catalystindex/...`): acquisition, policy resolver, parser registry, chunking, embeddings, search, job store, term index. Document gaps vs. checklist.

### B. Knowledge Base & Tenant Model
3. ❌ Introduce persistent KB catalog in Postgres: table storing `knowledge_base_id`, owner metadata (org/workspace/user), description, document counts, keywords.  
4. ❌ Extend ingestion request models to require/accept `knowledge_base_id`; create KB automatically when missing.  
5. ❌ Update artifacts, chunk metadata, term index, and vector payloads to tag every record with KB ID.  
6. ❌ Add API endpoints to list KBs, view KB metadata (documents, keywords), and enforce ownership checks in ingest/search.

### C. Policy Advisor & Heuristic Removal
7. ❌ Remove the title-based heuristic in `resolve_policy`; resolver must only honor explicit schema/advisor policy names.  
8. ❌ Ensure advisor returns policy name, parser hint, chunk overrides; record them in document metadata (advisor confidence/tags).  
9. ❌ Add dedicated policy templates for major document types (e.g., `ccbhc`) so the advisor can select them cleanly, and design an LLM-guided path for on-the-fly policies when no shared template fits (academic texts, research reports, ops manuals, etc.).  
10. ❌ Build a reusable “LLM-assisted chunking & metadata strategy” module so when a document lacks a matching policy we can synthesize the slicing rules and metadata prompts (and optionally persist them for re-use) instead of falling back to the generic recipe.

### D. Parser & Ingestion Flow Enhancements
11. ❌ Verify Unstructured adapters in Docker image include optional dependencies (pi_heif, etc.). Add fallback extraction (pypdf) when extras missing.  
12. ❌ Ensure parser selection respects advisor hint → MIME → fallback, with warnings logged when falling back.  
13. ❌ Persist structured artifacts (`chunks.json`, `embeddings.json`, etc.) per doc and ensure stage progress is updated for parsed/chunked/embedded/uploaded (with cache-aware rehydration).

### E. Embedding & Vector Store Configuration
14. ❌ Configure API + worker containers to use real embedding provider (OpenAI/Cohere) from `.env`; drop hash provider except in explicit dev-lite mode.  
15. ❌ Map each knowledge base to a Qdrant tenant/collection (isolated per track); dense path live, sparse path available behind `CATALYST_STORAGE__qdrant__sparse_vectors=true` for premium clusters.  
16. ❌ Wire reranker (Cohere/OpenAI) into the premium search pipeline; enable via env (`CATALYST_RERANKER__enabled=true`, provider/model/api keys).

### F. Search & Generation Paths
17. ❌ Update `/search/query` to require KB list (or `["*"]`) and enforce KB ownership before querying vector store (KB IDs propagate to term index + Qdrant filters).  
18. ❌ Confirm search flow: query expansion via Redis term index → dense search → optional sparse + hybrid fusion → rerank → response with citations (env toggles control sparse/rerank in prod).
19. ❌ Ensure generation/feedback endpoints reference the same KB scoping, using the retrieved chunks and recording feedback in Redis/metadata.

### G. Worker / Job Store Enhancements
20. ❌ Use RQ (Redis) for ingestion queue with configurable concurrency; expose env vars for max active docs + queue length backpressure and KB-aware workload limiter.  
21. ❌ Persist job + stage progress in Postgres (`ingestion_jobs`, `ingestion_job_documents`). Include LLM enrichment stages as sub-jobs.  
22. ❌ `/ingest/jobs` endpoints return both job-level summary and per-document stage progress (with knowledge base context).

### H. Bulk Ingestion & Resource Awareness
23. ❌ `/ingest/bulk` handles mixed file/URL arrays; each doc is a child job with stage progress.  
24. ❌ Add scheduler awareness: limit concurrently running docs per worker, add queue backpressure, and record status (“doc 3/10 chunking 60%”).  
25. ❌ Support resource-aware Firecrawl/download tasks so multiple large URLs don’t overload the worker (Firecrawl/HTTP acquisition wired; respects queue/backpressure and max_active docs; supply key to enable Firecrawl).

### I. Telemetry & Observability
26. ❌ Ensure metrics exporter runs in API container (Prometheus). Add counters/histograms for ingestion/search per KB.  
27. ❌ Log audit events for KB creation, ingestion completion, and search queries (with KB IDs).

### J. Proof & Validation
28. ❌ With compose stack running (API 18888, worker, Postgres, Redis, Qdrant), ingest `ccbhc-criteria-2022.pdf` into `kb_ccbhc` via API (artifacts stored under `artifacts/kb_ccbhc/...`).  
29. ❌ Job response succeeded (dev-lite default policy path confirmed); stage progress succeeded, artifacts persisted under KB-scoped artifacts.
30. ❌ Run `/search/query` for a staffing-related question in premium mode targeting `kb_ccbhc`; JSON output returns KB-scoped results.  
31. ❌ Summarize embedding/reranker performance and confirm system meets ≥8/10 across policy accuracy, ingestion reliability, search quality, and observability (prod settings enabled by env for real embeddings/reranker; dev-lite validated ingestion/search/telemetry paths).
