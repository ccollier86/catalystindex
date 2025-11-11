## Adaptive Ingestion Pipeline & Retrieval Spec

### 1. Document Profiling & Policy Resolution
- Parse each source (PDF/HTML/JSON) for title, headings, and quick structural cues.
- Run `chunking.policy.resolve_policy(document_title, schema_name)` to merge base defaults, domain class templates (DSM, treatment planner, etc.), and per-document overrides.
- If no class is defined, run the lightweight classifier (keyword heuristics + optional 0-shot LLM) to map the document into an archetype (e.g., `trauma_related`, `objective_playbook`, `page_pdf`).
- Policy output includes: allowed chunk modes, token limits, section rules, required metadata fields, and optional `llm_metadata` settings.

### 2. Document Parsing (Multi-Tool Chain)
- **Preferred pipeline (PDF/Scans)**: 
  1. `unstructured.partition` (strategy `hi_res`) or Marker/MinerU for layout + OCR extraction (best tables, figures, scanned DSM pages).
  2. `OpenParse` (semantic pipeline) to build normalized nodes (text + bbox). When OpenParse output is sparse, fall back to the raw Unstructured elements.
  3. Optional PaddleOCR for low-quality scans before feeding into the above.
- **Office formats (DOCX/XLSX/PPTX)**: `unstructured.partition` readers (docx, pptx, spreadsheet) with the same handoff into SectionText.
- **HTML/Markdown**: `BeautifulSoup` or direct ingestion when metadata already structured.
- **Standardization**: upstream parsers all output a normalized SectionText schema (text, bbox/page span, metadata) so the chunk engine sees a consistent interface, regardless of source format.

### 3. Section Segmentation
- Group nodes into `SectionText` units according to policy-defined headings (Diagnostic Criteria, Specifiers, Short-Term Objectives, etc.).
- Each section stores name, combined text, page span, and extra metadata (ICD code, diagnosis group, goal type).

### 4. Chunk Generation Engine
- `chunking.engine.generate_chunks` applies policy chunk modes per section:
  - `semantic`: raw nodes within min/max thresholds.
  - `window`: sliding windows with configurable size/overlap.
  - `criteria`: regex/heading splits (Criterion A/B/C, numbered lists).
  - `table`: preserve tables as single chunks.
  - `highlight`: short callouts for warnings or suicide risk.
  - `page`: optional page-level preservation for layout-critical PDFs.
- Every chunk receives metadata: document ID, section slug, start/end page, bbox pointer, chunk tier/index, token count, plus policy-required fields (diagnosis group, specifier flags, severity, etc.).
- Debug artifacts store both the chunk list and the resolved policy JSON for traceability/debugging.
- All chunk text is normalized to Markdown before embeddings (headings, bullet lists, tables) so retrieval UI and generation prompts see consistent formatting regardless of original file type.

### 5. LLM Metadata Enrichment (Optional)
- When `llm_metadata.enabled`, batches of chunks run through the specified model to extract:
  - `summary`: concise paraphrase.
  - `key_terms`: search aliases/phrases.
  - `requires_previous` + `prev_chunk_id`: dependency hints for cross-chunk context.
  - `confidence_note`: labels like “criteria-heavy”, “narrative background”.
  - Additional fields defined in the schema (e.g., ICD mentions, insights).
- Enriched metadata is stored per chunk and logged for downstream systems (search alias index, prompt templates).

### 6. Per-Document Term Index
- Each chunk’s `key_terms` populate `term_index[document_id][chunk_id] = {"aliases": [...], "score": baseline}`.
- This index powers pre-vector-search query expansion and can be modified by user feedback (thumbs up/down adjust alias scores and optionally add/remove phrases).

### 7. Embedding & Validation
- Chunks are embedded using the configured model (default `text-embedding-3-large`).
- Before uploading, ensure vector dimension matches collection requirements (prevents schema mismatch errors).
- Embedding results are cached in `artifacts/<slug>/embeddings.json`; reruns only compute missing entries unless forced.

### 8. Cached Qdrant Upsert
Track uploaded chunk IDs in `artifacts/<slug>/vectorstore_uploaded.json`.
On rerun, only upsert new/changed chunks; metadata is serialized with booleans/JSON strings per Qdrant requirements.

### 9. Feedback Loop Hooks
- Each chunk’s metadata includes `chunk_id` (+ optional `prev_chunk_id`) so front-end can auto-expand context.
- Thumbs up/down feedback logs `(query, chunk_ids, feedback)`; a follow-up job updates term index scores and can trigger re-enrichment if certain chunks consistently underperform.

### 10. Search & Retrieval Flow
- Query pre-processing consults the term index for high-score aliases, expanding or rewriting user input before vector search.
- **Vector store**: move toward Qdrant (local or managed) for production. Store embeddings + metadata payloads; leverage filters to target chunk tiers, diagnoses, ICD codes, etc.
- Retrieval pipeline:
  1. Use Qdrant’s payload filtering to restrict candidate chunks (e.g., `diagnosis_group="trauma_related"`, `chunk_tier in ["criteria","semantic"]`).
  2. Run vector search with `k` tuned per question type; optionally follow with re-ranking (e.g., Cohere ReRank or Qdrant sparse hybrid search).
  3. Optionally fetch `prev_chunk_id` when `requires_previous=true` to keep context intact.
- Retrieved chunks provide citations via metadata (section name, page span, ICD code, LLM summary).

### 11. LLM Generation
- Prompt templates incorporate retrieved chunks, referencing metadata for citations and mentions of page numbers or diagnostic sections.
- If `requires_previous` is true, the orchestrator auto-loads the upstream chunk to maintain continuity.
- Responses carry source IDs so the feedback loop knows which chunks to reward/punish.
- Generation spec (per research best practices):
  - Include chunk summaries + raw text in the prompt so the LLM sees both high-level context and verbatim references.
  - Ask the LLM to cite chunk/page identifiers (Qdrant payload fields) and to note when context is insufficient rather than hallucinate.
  - When feedback is negative, log the involved chunk IDs for re-embedding/re-chunking if patterns emerge.

### 12. Continuous Improvement
- Scenario tests (`scripts/chroma_showcase.py`) run representative queries to monitor retrieval precision per diagnosis/class.
- Schema updates (chunk sizes, section rules) are applied via YAML changes; cached artifacts allow targeted reprocessing.
- Research-driven features (page-level chunking, context summaries, SPLICE-style hybrid splitting) are encoded in the schema engine, ensuring future corpora benefit without code rewrites.

### 13. Orchestration & Batch Interfaces
- The ingestion pipeline exposes adapters so external schedulers/queues can orchestrate work:
  - **Source adapters** accept either file paths, URLs, or already-fetched blobs (e.g., Firecrawl output), so workers simply hand off the pointer and metadata.
  - **Status hooks** emit events at each phase (parsed → chunked → enriched → embedded → uploaded), allowing queue workers to log progress or retry failed steps without guessing internal state.
  - **Configurable artifact destinations** (local paths, S3 prefixes) ensure each job can isolate its outputs.
  - Idempotent caching (semantic chunks, embeddings, uploaded IDs) makes retries safe, which is critical for batch job queues that may re-run tasks.
- Queue/cron logic stays outside the pipeline; the pipeline just provides clean entrypoints for asynchronous orchestrators to invoke.

> **Future Idea:** Log chunk relationships (e.g., DSM criteria ↔ treatments ↔ patient notes) and periodically sync them into a lightweight graph store (Neo4j/Memgraph). This enables relationship-aware search and visual analytics later without changing the core pipeline.
- **Supported formats**: PDF (digital + scanned), DOCX, PPTX, XLSX, HTML/Markdown, JSON exports, and raw webpages. Unknown formats are routed through `unstructured.partition.auto` first; if parsing fails we log and skip.
- **Direct URL ingestion**: for non-file content, Firecrawl fetches the page (or sitemap) and emits cleaned Markdown/HTML. That output is normalized into SectionText before chunking so the pipeline treats crawled URLs identically to uploaded documents.
