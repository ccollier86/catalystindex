## Generation Specification

### 1. Goals
- Keep the core platform focused on ingestion + retrieval; let downstream apps run their own prompts.
- Provide a single lightweight generation endpoint for use cases that need a quick, standardized answer (e.g., mobile, public endpoint, internal admin console).
- Supply SDK helpers (Python / TypeScript) so product teams can build custom generation flows without re-implementing plumbing.

### 2. Simple Generation Endpoint (`POST /generate/summary`)

**Use cases**
- Quick “on-the-go” summaries (mobile app, clinician phone call).
- Public/limited-access knowledge sharing (optional rate-limited endpoint).
- Internal tooling where the client doesn’t want to manage prompt templates.

**Inputs**
```json
{
  "query": "What are the short-term objectives for PTSD exposure therapy?",
  "tenant_id": "org_123",
  "retrieval_options": {
    "max_chunks": 6,
    "chunk_tiers": ["criteria", "semantic"],
    "include_vision": false
  }
}
```

**Process**
1. Reuse the primary retrieval pipeline (hybrid search + rerank) with a smaller `k` to keep latency low.
2. Assemble a templated prompt:
   ```
   You are a clinical reference assistant. Using only the provided context, answer the question.
   Cite chunks using [chunk_id] and include DSM/ICD references when available.
   Context:
   - Chunk chunk_1: <summary + snippet + citation>
   - ...
   Question: <query>
   ```
3. LLM selection (configurable, default `gpt-4o-mini`). Enforce max tokens and cost guardrails.
4. Return JSON with `answer`, `citations` (chunk IDs + page info), `confidence`, `source_chunks`.

**Outputs**
```json
{
  "answer": "Short-term objectives include ...",
  "citations": [
    { "chunk_id": "ptsd|criteria|2", "pages": "p.3-4" }
  ],
  "sources": [
    { "chunk_id": "...", "document_title": "...", "section_slug": "...", "snippet": "..." }
  ],
  "confidence": 0.82
}
```

**Limitations**
- Not multi-turn conversation; single question per call.
- Rate-limited per tenant; JWT required.
- Returns general guidance only (no patient-specific recommendations).

### 3. SDK Strategy
- Publish **Python** and **TypeScript** SDKs that expose:
  - Retrieval client (wrapping `/search` with hybrid/rerank defaults).
  - Prompt helpers (functions to build DSM summaries, treatment outlines, etc.).
  - Optional wrapper for `/generate/summary`.
  - Utilities for logging feedback back to the platform.
- The SDK spec will detail authentication (JWT), pagination, streaming support, and prompt helper interfaces in a separate document.

### 4. Custom Generation Guidance
- Apps can call the retrieval API directly and apply their own prompt templates.
- Recommend including:
  - Retrieved chunk summaries + raw text (for citations).
  - Page numbers and ICD codes in the prompt instructions.
  - Guardrails (“If context is insufficient, respond with `INSUFFICIENT_CONTEXT`.”).
- Encourage logging user feedback and chunk IDs to feed the EVOLVE loop.

### 5. Economy Considerations
- Economy tier may disable `/generate/summary` entirely or use a smaller LLM (e.g., `gpt-4o-mini` vs `gpt-4o`).
- SDK should allow users to plug in their own LLM endpoint (OpenAI, Anthropic, self-hosted) while still leveraging the same retrieval plumbing.

### 6. Security & Compliance
- Require JWT with tenant info for every generation call.
- Enforce content policies (e.g., strip PHI from prompts unless tenant is authorized).
- Log prompt+response metadata for auditing (hashed/anonymized as needed).
