# Feature Backlog (High-Value Enhancements)

## Sliding Context / Auto-Expansion
- Store chunk adjacency (prev/next IDs) and expose API to pull neighboring chunks.
- Allow generation service to auto-expand context when confidence is low.
- Add UI control (“Expand context”) so users can explicitly pull surrounding sections.

## Visual Reconstruction & Browsing
- Render original artifacts (HTML/PDF) with chunk highlights.
- Provide a “Browse Document” mode with citations linking back to the highlighted source.

## Live RAG-as-a-Service Endpoint
- Single endpoint that accepts raw text or URL, ingests on the fly, retrieves, and responds immediately.
- Ideal for “upload and ask” demos and testing.

## Workflow Templates (“Ingestion Recipes”)
- Let users define multi-step pipelines (fetch, OCR, chunk, tag, embed) as JSON or YAML.
- UI/CLI to select and run templates for specific document types.

## Feedback-Triggered Retraining
- Monitor positive feedback counts per chunk.
- When thresholds are exceeded, export those chunks for fine-tuning or prompt updates.

## Knowledge Diff / Release Notes
- After crawls or bulk ingest runs, summarize what changed (new sections, removed content).
- Deliver release notes via email/Slack/webhook.

## Tenant-Specific Search Personalization
- Allow per-tenant ranking rules (boost/deboost policies, recency bias).
- Optionally plug in tenant-provided rerank APIs.

## Agent Hooks / Automation
- Expose an API to queue tasks (“ingest this URL”, “monitor this site”) so external agents can drive the pipeline.
- Provide status callbacks for those agents.

## Contextual Alerts
- Let users define alert rules (e.g., “if chunk contains ‘FDA recall’, send Slack alert”).
- Integrate with Slack/Teams/webhooks.

## On-Device / Edge Mode
- Provide a configuration profile that uses quantized embeddings, smaller chunk sizes, and optional local vector store.
- Enables air-gapped or on-prem deployments.
