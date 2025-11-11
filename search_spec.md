## Retrieval & Search Specification (Qdrant-centric)

### Goals
- Hybrid semantic + lexical retrieval with re-ranking for high-precision answers.
- Respect ingestion policies (text vs vision tracks, chunk tiers, ICD metadata).
- Enforce per-tenant isolation via JWT payload filtering.
- Provide an economy mode (Qdrant-only features) and a premium mode (external rerankers, larger k).

### 1. Query Normalization & Routing
1. **Sanitize** user input (preserve clinical abbreviations, lower noise punctuation).
2. **LLM Query Analyzer** (optional): classify intent (diagnosis lookup, treatment planning, general narrative, vision-required).
3. **Alias expansion** (term index):
   - Fetch top aliases for the predicted document class and append high-confidence synonyms.
   - Example: “behavioral activation” → add “pleasant activities scheduling”.
4. **Track hint**:
   - If the analyzer flags “vision_required” (e.g., mentions “see figure/table”), mark the request to include vision vectors.

### 2. Vector & Hybrid Search
#### 2.1 Named Vectors
- Each Qdrant point can store:
  - `text_dense` (1536-dim) from `text-embedding-3-large`.
  - `text_sparse` (optional BM25/ColBERT-style) for lexical matching.
  - `vision_dense` (e.g., from ColQwen/Nougat) when the document came from the vision track.
- Collections store metadata fields: `chunk_tier`, `diagnosis_group`, `icd_code`, `section_slug`, `tenant_ids`, etc.

#### 2.2 Payload Filters
- **Must** clauses:
  - Tenant enforcement: `{ key: "org_id", match: { value: <jwt_org> } }`, similarly for workspace/user.
  - Chunk tier selections per query type (diagnostic = criteria/semantic, treatment = semantic/window, narrative = window/highlight).
  - Diagnosis group filter if classifier is confident.
- **Should** clauses:
  - Boost specific ICD codes, specifiers, or document titles when relevant.
- **Must_not** clauses:
  - Exclude `highlight` chunks unless specifically requested.
  - Respect user-level blocklists (e.g., previously dismissed chunk IDs).
- **Nested filters**:
  - For arrays such as `criteria[]` or `objectives[]`, use nested filters to match label + text within the same entry (per Qdrant nested filter docs).

#### 2.3 Hybrid Retrieval Plan
- Premium endpoint:
  1. **Prefetch** dense text results (`text_dense`) with k≈24.
  2. **Prefetch** sparse text results (`text_sparse`) using BM25-style sparse vectors (if indexed).
  3. Optional third prefetch for `vision_dense` when query hints at visual content.
  4. Combine prefetches via `fusion: "rrf"` or `dbsf`.
  5. Apply Maximal Marginal Relevance (`mmr`) to ensure diversity (diversity=0.4, candidates=50).
  6. Re-rank top 20 with external cross-encoder (e.g., Cohere ReRank, bge-reranker) for the final top 6 chunks.
- Economy endpoint:
  - Use Qdrant’s built-in RRF fusion and optional re-ranking (if available) without external models.
  - Limit k to 8–10 for cost control.

#### 2.4 Vision Integration
- If `vision_required` or the document is flagged as vision track:
  - Run an additional vector query using `vision_dense`.
  - Fuse results with text hits (RRF).
  - Ensure returned payloads include references to the original figure/page for LLM prompt context.

### 3. Context Stitching & Deduplication
- Auto-fetch `prev_chunk_id` when `requires_previous=true`.
- Deduplicate chunks from the same section if they overlap heavily (based on `chunk_id` prefix and `chunk_tier`).
- Preserve ordering per document to maintain narrative flow.

### 4. Result Payloads
Each retrieval result returns:
- `chunk_id`, `document_title`, `section_slug`, `chunk_tier`.
- `start_page`, `end_page`, `bbox_pointer`.
- LLM metadata: `summary`, `key_terms`, `requires_previous`, `confidence_note`.
- Diagnostic metadata: `diagnosis_group`, `icd_code`, `specifier_flags`.
- Track info: `track: "text"` or `"vision"`, plus `vision_context` if available.

### 5. Feedback & Continuous Learning
- Log `(query_text, chunk_ids, tenant_id, feedback)` for every user reaction (thumbs up/down).
- Nightly job:
  - Adjust alias scores (reinforce positive, decay negative).
  - Flag repeatedly downvoted chunks for re-chunking or policy review.
  - Produce EVOLVE metrics (precision@k, recall@k per diagnosis group).

### 6. Monitoring & Testing
- Scenario suite (updated `scripts/chroma_showcase.py` → Qdrant) runs canonical queries for each diagnosis and compares retrieved chunk tiers/ICD coverage.
- Log Qdrant Query API payloads (prefetch configs, filters) with anonymized query text for debugging.
- Track latency for premium vs economy endpoints (dense+rerank vs dense-only).

### 7. Integration & Multitenancy Notes
- Search API should expose `retrieve(query, options, jwt)` → returns structured chunks plus metadata.
- Always pass the user JWT through to Qdrant so payload filters enforce org/workspace isolation.
- Cache layers:
  - **Embed cache** keyed by `(tenant, query_text)` to avoid re-embedding repeated queries.
  - **Result cache** (short TTL) for identical `(tenant, query_text, filters)` to skip re-running hybrid/rerank when data hasn’t changed.
  - Respect tenant boundaries in cache keys to avoid leaking context.

### 8. Economy Mode Summary
- Single Qdrant query with dense vector (optional sparse) and built-in RRF.
- No external reranker; rely on MMR + score thresholds.
- Smaller k, limited metadata (omit expensive summaries if desired).
- Ideal for low-cost deployments or internal dashboards.
