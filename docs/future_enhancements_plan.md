# Future Enhancements Plan

## 1. Feedback-Driven Auto-Tuning
- Use aggregated feedback to adjust chunk tier weights, reranker weights, and alias prominence per tenant.
- Provide a dashboard showing how feedback influenced ranking decisions.

## 2. Data Retention & Compliance Controls
- Add per-policy retention windows, masking rules, and a “forget this document” workflow.
- Audit logs should note when content is purged or masked.

## 3. Answerability & Guardrails
- Expose an “answerability score” from the generation service; allow clients to stop when confidence is low.
- Provide configurable thresholds per tenant.

## 4. Ops Console
- Build a web dashboard to visualize ingestion jobs, crawls, streaming feeds, and telemetry snapshots.
- Include drill-down into failures, retries, and feedback analytics.

## 5. Tenant-Specific Embeddings/Rerankers
- Let premium tenants choose their provider (Azure OpenAI, local model, etc.).
- Support mixing providers (e.g., OpenAI embeddings + Cohere reranker) with per-tenant config.

## 6. Feedback-to-Training Export
- Produce a structured dataset of approved chunks and rejections for offline fine-tuning or prompt updates.
- Integrate with model retraining pipelines when available.
