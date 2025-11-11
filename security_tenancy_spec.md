## Security & Multitenancy Specification

### 1. Identity & Authentication
- **JWT-based auth** with HS256/RS256 signatures (per deployment). Claims include:
  - `org_id`, `workspace_id`, `user_id`, `roles`, `scopes`, `exp`, `iat`.
  - Optional `feature_flags` (economy vs premium).
- **API keys** only for service-to-service interactions; must exchange for JWT via `/auth/token`.
- Rotate signing keys quarterly; store in secure vault (AWS KMS, GCP Secret Manager).

### 2. Authorization & Isolation
- Every request must include a JWT; services enforce:
  - **Row-level filtering**: Qdrant filters `org_id`/`workspace_id`/`user_id`.
  - **Policy enforcement**: Access to ingestion/search/generation endpoints gated by role scopes (`ingest:write`, `search:read`, `generate:write`, etc.).
- **Cross-tenant isolation**:
  - No shared collections unless payload filters apply per query.
  - Optional dedicated Qdrant collections per tenant for high-security clients.
- **Adapter boundaries**: SDKs automatically attach JWT; internal services pass JWT context downstream.

### 3. Data Protection
- **At rest**: Encrypt S3 artifacts, Qdrant payload storage, Postgres feedback logs. Use KMS-managed keys.
- **In transit**: Enforce TLS 1.2+ for all endpoints. Mutual TLS between internal services (optional but recommended).
- **PHI/PII handling**: Redact sensitive fields in logs; store minimal patient identifiers unless tenant explicitly enables PHI storage.

### 4. Audit & Compliance
- Audit events (ingest/search/generation/policy changes) written to append-only store (see logs spec).
- Provide tenant-level export for compliance audits (JSON/CSV).
- Support GDPR/CCPA data deletion: per tenant/user, purge artifacts + vector entries via job or TTL policies.

### 5. Rate Limiting & Abuse Prevention
- Tiered rate limits per tenant and per user (e.g., 100 search requests/minute, 10 generation requests/minute).
- Burst allowances for ingestion jobs with background queue monitoring.
- Automatic block after repeated invalid JWTs or suspected scraping.

### 6. Secrets & Config
- Use centralized secret manager; environments load via DI (no secrets in repo).
- SDKs encourage environment variable config but never log secrets.

### 7. Vision/LLM Considerations
- Vision pipeline (ColQwen/Nougat) runs in isolated containers with no external network access (only object storage).
- LLM calls (OpenAI/Meta) proxied through secure gateway; per-tenant usage limits tracked for billing.

### 8. Incident Response
- On detection of cross-tenant data exposure:
  1. Quarantine affected services.
  2. Rotate JWT signing keys.
  3. Audit Qdrant payload filters for misconfiguration.
  4. Notify affected tenants per SLA.
- Maintain runbooks for token revocation, Qdrant ACL updates, and emergency shutdown of generation endpoints if abuse detected.

### 9. Deployment Considerations
- Separate environments (dev/stage/prod) with distinct tenants; never mix customer data in dev.
- Infrastructure-as-code (Terraform) with security scanning (Checkov, tfsec).
- Regular penetration tests; automated dependency scanning (Snyk, Dependabot).
