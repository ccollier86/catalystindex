## SDK Specification (Python & TypeScript)

### 1. Goals
- Provide first-class clients that can drive the entire platform (ingestion → chunking → search → generation → feedback) without hand-writing REST calls.
- Offer consistent ergonomics across Python and TypeScript while respecting each ecosystem’s conventions.
- Support both premium (hybrid + rerank + generation) and economy (Qdrant-only) features via configuration flags.

### 2. Core Modules

| Module | Responsibilities |
| --- | --- |
| `Auth` | Handle JWT issuance/refresh, tenant scoping, API key management. |
| `Ingestion` | Upload files/URLs, check job status, retrieve artifacts (chunks, embeddings). |
| `Policies` | Inspect/override chunking policies and schema metadata. |
| `Search` | Invoke the hybrid retrieval pipeline with sane defaults (filters, rerankers, vision flags). |
| `Generation` | Optional wrapper around `/generate/summary` (and future endpoints). |
| `Feedback` | Submit thumbs-up/down, retrieve analytics (EVOLVE hooks). |

### 3. Ingestion API (SDK)
```python
client.ingest.upload_file(
    path="docs/PTSD.pdf",
    policy="dsm5",
    metadata={"org_id": "...", "workspace_id": "..."}
)
job = client.ingest.wait(job_id)
artifacts = job.list_artifacts()  # chunks.json, embeddings.json, etc.
```
- Support both blocking (`wait`) and async/polling.
- Provide helpers to list artifacts, download JSON, or trigger reprocessing with `force` flags.

### 4. Search API (SDK)
```typescript
const results = await client.search.retrieve({
  query: "PTSD exposure objectives",
  options: {
    tiers: ["criteria","semantic"],
    includeVision: false,
    economyMode: false,
  },
});
```
- Expose filter builders (e.g., `Filter.tenant(jwt)`, `Filter.diagnosis("anxiety")`).
- Allow callers to toggle economy mode (skips external reranker/MMR) vs premium.
- Provide convenience methods for post-processing: `results.toCitations()`, `results.toPromptContext()`.

### 5. Generation API (SDK)
```python
client.generate.summary(
    query="Summarize DSM-5 PTSD criteria.",
    retrieval_options={"max_chunks": 6}
)
```
- Thin wrapper around `/generate/summary`. Optional; apps can skip if they want full control.
- Respect economy flags (disable or auto-select smaller LLM).

### 6. Feedback & Analytics
```typescript
await client.feedback.submit({
  queryId,
  chunkIds: ["ptsd|criteria|2"],
  rating: "thumbs_up",
  comment: "Accurate exposure steps"
});
```
- Provide helper to fetch EVOLVE metrics (precision@k, etc.) for dashboards.

### 7. Configuration & Auth
- Support API key + JWT combo:
  - Python: `Client(api_key="...", jwt=JWT.from_service_account(...))`
  - TypeScript: similar, with browser-friendly storage (if used client-side, rely on session token).
- Allow overriding endpoints (ingestion/search/generation) and timeouts.

### 8. Error Handling & Logging
- Consistent exception hierarchy (`IngestionError`, `SearchError`, etc.).
- Optional debug logging/tracing hooks to correlate SDK calls with server logs.

### 9. Packaging Plan
- **Python**: `pip install clinical-rag-sdk`; typed (PEP 561), synchronous + async clients (using `httpx`).
- **TypeScript**: `npm install @clinical/rag-sdk`; ESM + CJS bundles, works in Node + browser (fetch-based).
- Provide minimal CLI entry point (`python -m clinical_rag ingest file.pdf`) for quick testing.
