## Catalyst Index Premium RAG Plan

1. **Dockerized Runtime**
   - API + worker containers plus Postgres (job store), Redis (term index + RQ queue), Qdrant (vector store/KB isolation) under `infrastructure/docker-compose.dev.yml`.
   - Containers load `.env`, override hosts/ports, expose API on 18888, metrics on 9464.

2. **Authentication & Knowledge-Base Ownership**
   - JWT scopes enforced (`ingest:write`, `ingest:read`, `search:read`, etc.).
   - Knowledge bases (`knowledge_base_id`) are the primary tenant concept; each maps to a Qdrant tenant/collection and is tied to org/workspace/user ownership for auth.

3. **Acquisition & Artifacts**
   - Supports inline/upload/url ingestion; Firecrawl optional for URLs.
   - Auto MIME detection + metadata (`content_type`, `payload_size`, `fetched_uri`).
   - Artifacts stored per KB under `artifacts/<kb>/<job>/<doc>` with raw file + metadata JSON + structured artifacts (`chunks.json`, `embeddings.json`, `policy.json`, etc.).

4. **Policy Advisor (No Heuristics)**
   - If schema provided, use it; otherwise call LLM advisor for policy selection, parser hint, chunk overrides.
   - Remove title-based heuristics; log advisor decisions in document metadata.

5. **Parser Layer**
   - Unstructured adapters for PDF/Docx/PPTX/XLSX/HTML with fallback extraction (pypdf etc.) when optional deps missing.
   - Selection uses advisor hint → MIME type → fallback; failures drop to plain-text parser but log warnings.

6. **Chunking & Metadata**
   - `ChunkingEngine` obeys resolved policy (modes, LLM metadata). Term index updated per chunk.
   - Stage-by-stage progress recorded (acquired/parsed/chunked/embedded/uploaded) with timestamps/details.

7. **Embedding & Vector Store**
   - Real embedding provider (OpenAI/Cohere) configured in env; no hash embeddings in premium path.
   - Every KB maps to a Qdrant tenant segment/collection. Premium mode **always** runs dense + sparse embeddings and hybrid search.
   - Reranker (Cohere/OpenAI) enabled for premium responses; economy mode can disable rerank/sparse if desired.

8. **Search & Generation**
   - Search requests include one or more KB IDs (or “all”) and run: query expansion → dense search → sparse search → hybrid fusion → rerank → result set with citations.
   - Generation endpoint reuses search results for the same KB(s); feedback updates term index/metadata counters.

9. **Worker & Job Store**
   - RQ worker in Docker processes ingestion queue (Redis), persisting job status + stage progress in Postgres.
   - APIs `/ingest/jobs*` expose job summaries/details with progress snapshots and artifact URIs.

10. **Telemetry & Logging**
   - Prometheus exporter running in API container; audit logs for ingestion/search events per KB.

11. **Proof Workflow**
   - Bring stack up with `docker compose -f infrastructure/docker-compose.dev.yml up -d`.
   - Ingest `ccbhc-criteria-2022.pdf` into a dedicated KB (e.g., `kb_ccbhc`) via API using a valid JWT.
   - Verify job success, artifacts path, policy advisor output, stage progress, chunk count.
   - Run `/search/query` for staffing question in premium mode targeting that KB; confirm results reference staffing sections.
   - Provide curl commands + responses for ingest and search so the run is reproducible.

12. **Bulk Ingestion & Concurrency Controls**
   - `/ingest/bulk` accepts mixed arrays of files + URLs. Every submission specifies a knowledge base and becomes part of a parent job.
   - Job status surfaces both job-level progress (“4/5 documents complete”) and per-document stage progress (“doc4 parsed 40%, chunking running”).
   - Bulk jobs can throttle concurrency: configurable max active docs per worker, queue-level backpressure, and optional scheduler awareness of system load.
   - Same mechanism supports resource-aware URL crawling (Firecrawl + direct downloads) to avoid overloading the worker or external services.
   - Large documents that require LLM-based metadata/classification run as explicit batch stages (with sub-job IDs) so enrichment progress is tracked separately from base chunking/embedding.
   - When a `knowledge_base_id` doesn’t exist, ingestion creates it automatically and records it in Postgres (owner metadata, document list, keywords). An API endpoint lists KBs, their documents, and summary metadata for clients to browse.
