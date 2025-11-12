# LLM-Assisted Policy & Metadata Plan

## 1. Policy Advisor Service
- Sample document (first N chars + metadata)
- Call OpenAI (e.g., `gpt-4o-mini`) asking for the best ingestion recipe.
- Return: `policy_name`, overrides (chunk modes/window), confidence, suggested tags.

## 2. Ingestion Flow Integration
- Use manual schema when provided; otherwise call Policy Advisor.
- Attach advisor decision to `document_metadata`.
- Apply overrides when the advisor suggests custom chunk settings.
- Log decisions in audit telemetry.

## 3. LLM Metadata Enrichment
- Optional `LLMMetadataService` for selected chunk tiers.
- Generate refined summaries and classification tags (doc type, severity, etc.).
- Respect policy-level flags (`llm_metadata.enabled`).

## 4. Configuration
- `CATALYST_POLICY_ADVISOR__provider`, `__model`, `__api_key`, `__sample_chars`.
- `CATALYST_METADATA_LLM__enabled`, `__model`, `__api_key` (defaults to OpenAI key if empty).

## 5. Testing & Validation
- Unit tests with mocked LLM responses.
- Smoke test ingesting a doc without schema and verifying the advisor result.
- Document how to enable/disable and audit the advisor.

## 6. Future Enhancements
- Cache LLM decisions per document hash to avoid requerying.
- Few-shot prompts for better classification accuracy.
- Telemetry metrics for advisor confidence, fallback rates, and metadata coverage.
