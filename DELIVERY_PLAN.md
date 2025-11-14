## Catalyst Index Upgrade Plan

### A. Repository Audit & Baseline
1. Inspect current Dockerfile/compose output and verify which FastAPI/Pydantic packages are actually bundled (confirm we’re loading the real packages, not the stubs in `fastapi/` and `pydantic/`).  
2. Map current service wiring (`src/catalystindex/...`): acquisition, policy resolver, parser registry, chunking, embeddings, search, job store, term index. Document gaps vs. checklist.

### B. Knowledge Base & Tenant Model
3. Introduce persistent KB catalog in Postgres: table storing `knowledge_base_id`, owner metadata (org/workspace/user), description, document counts, keywords.  
4. Extend ingestion request models to require/accept `knowledge_base_id`; create KB automatically when missing.  
5. Update artifacts, chunk metadata, term index, and vector payloads to tag every record with KB ID.  
6. Add API endpoints to list KBs, view KB metadata (documents, keywords), and enforce ownership checks in ingest/search.

### C. Policy Advisor & Heuristic Removal
7. ✅ Removed the title-based heuristic in `resolve_policy`; resolver now only honors explicit schema/advisor policy names.  
8. Ensure advisor returns policy name, parser hint, chunk overrides; record them in document metadata (advisor confidence/tags).  
9. Add dedicated policy templates for major document types (e.g., `ccbhc`) so the advisor can select them cleanly.

### D. Parser & Ingestion Flow Enhancements
10. Verify Unstructured adapters in Docker image include optional dependencies (pi_heif, etc.). Add fallback extraction (pypdf) when extras missing.  
11. Ensure parser selection respects advisor hint → MIME → fallback, with warnings logged when falling back.  
12. Persist structured artifacts (`chunks.json`, `embeddings.json`, etc.) per doc and ensure stage progress is updated for parsed/chunked/embedded/uploaded.

### E. Embedding & Vector Store Configuration
13. Configure API + worker containers to use real embedding provider (OpenAI/Cohere) from `.env`; drop hash provider except in explicit dev-lite mode.  
14. Map each knowledge base to a Qdrant tenant/collection (or payload filter). Confirm sparse + dense vectors are stored for premium mode.  
15. Wire reranker (Cohere/OpenAI) into the premium search pipeline; ensure economy mode can opt out.

### F. Search & Generation Paths
16. Update `/search/query` to require KB list (or `["*"]`) and enforce KB ownership before querying vector store.  
17. Confirm search flow: query expansion via Redis term index → dense search → sparse + hybrid fusion → rerank → response with citations.  
18. Ensure generation/feedback endpoints reference the same KB scoping, using the retrieved chunks and recording feedback in Redis/metadata.

### G. Worker / Job Store Enhancements
19. Use RQ (Redis) for ingestion queue with configurable concurrency; expose env vars for max active docs.  
20. Persist job + stage progress in Postgres (`ingestion_jobs`, `ingestion_job_documents`). Include LLM enrichment stages as sub-jobs.  
21. `/ingest/jobs` endpoints return both job-level summary and per-document stage progress (with knowledge base context).

### H. Bulk Ingestion & Resource Awareness
22. `/ingest/bulk` handles mixed file/URL arrays; each doc is a child job with stage progress.  
23. Add scheduler awareness: limit concurrently running docs per worker, add queue backpressure, and record status (“doc 3/10 chunking 60%”).  
24. Support resource-aware Firecrawl/download tasks so multiple large URLs don’t overload the worker.

### I. Telemetry & Observability
25. Ensure metrics exporter runs in API container (Prometheus). Add counters/histograms for ingestion/search per KB.  
26. Log audit events for KB creation, ingestion completion, and search queries (with KB IDs).

### J. Proof & Validation
27. With compose stack running (API 18888, worker, Postgres, Redis, Qdrant), ingest `ccbhc-criteria-2022.pdf` into `kb_ccbhc` via API.  
28. Show job response: policy advisor selected `ccbhc` policy (no heuristics), stage progress all succeeded, artifacts live under `artifacts/kb_ccbhc/...`.  
29. Run `/search/query` for a staffing-related question in premium mode targeting `kb_ccbhc`. Provide JSON output showing staffing chunks ranked at top.  
30. Summarize embedding/reranker performance and confirm system meets ≥8/10 across policy accuracy, ingestion reliability, search quality, and observability.
