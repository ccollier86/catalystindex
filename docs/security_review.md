# Security Review – Multitenancy & Access Controls

This review evaluates the current implementation against the commitments in `security_tenancy_spec.md` with focus on JWT propagation, tenant isolation, and artifact access management.

## JWT Propagation & Validation
- `decode_jwt` verifies HMAC signatures using the configured secret before any claims are consumed and raises `JWTAuthError` when the format or payload is invalid. 【F:src/catalystindex/auth/jwt.py†L18-L41】
- API dependencies require an `Authorization: Bearer` header, decode the token, and extract tenant identity (`org_id`, `workspace_id`, `user_id`). Downstream dependencies receive both the raw claims and the typed `Tenant` object, ensuring every request carries the caller context. 【F:src/catalystindex/api/dependencies.py†L115-L163】
- Scope checks are implemented via `ensure_scopes`, which runs inside dependency functions like `require_scopes`, keeping authorization decisions centralized and composable per endpoint. 【F:src/catalystindex/auth/jwt.py†L44-L52】【F:src/catalystindex/api/dependencies.py†L165-L176】

## Tenant Isolation
- Tenant identity is injected into service calls through dependency wiring. Ingestion and search services both require a `Tenant`, guaranteeing that orchestration work (chunk enrichment, metrics, audit logging) stays tenant-aware. 【F:src/catalystindex/services/ingestion.py†L43-L88】【F:src/catalystindex/services/search.py†L69-L152】
- Vector store adapters namespace both data writes and queries. The in-memory store keys documents by `org_id:workspace_id` while the Qdrant adapter prefixes collection IDs and filters every search with the tenant org/workspace pair plus the requested track. 【F:src/catalystindex/storage/vector_store.py†L33-L76】【F:src/catalystindex/storage/vector_store.py†L205-L271】
- Term index and metrics wiring are resolved per request using dependency injection, preventing cross-tenant leakage through shared caches or instrumentation state. 【F:src/catalystindex/api/dependencies.py†L20-L84】

## Artifact Access Controls
- All artifact stores scope storage paths by tenant identifiers. The in-memory store keys URIs as `memory://{org}/{workspace}/{job}/{document}`, while the local filesystem backend creates nested directories per tenant and job before writing content. 【F:src/catalystindex/artifacts/store.py†L26-L60】【F:src/catalystindex/artifacts/store.py†L62-L113】
- The S3-backed implementation constructs object keys that include organization, workspace, and job segments, and attaches metadata containing payload size and timestamps for auditing and size enforcement. 【F:src/catalystindex/artifacts/store.py†L115-L186】
- Document metadata is merged into artifact payloads (including policy, source labels, and size), ensuring downstream access control layers can inspect or redact sensitive attributes consistently. 【F:src/catalystindex/services/ingestion.py†L93-L138】

## Summary
The current implementation satisfies the spec’s requirements for authenticated, per-tenant operations. JWT claims are validated and propagated through FastAPI dependencies, storage adapters enforce tenant-specific namespaces, and artifact stores persist objects under tenant-scoped keys with accompanying metadata for auditability. Future hardening should add automated rotation checks for signing keys and extend audit logging to capture failed authorization attempts.
