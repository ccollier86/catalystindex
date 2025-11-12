# Catalyst Index RAG Platform

This repository contains a reference implementation of the Catalyst Index retrieval-augmented generation platform as described in the accompanying specifications. The service is designed with FastAPI and emphasises clean architecture boundaries (parsers, chunking, storage, search, generation) so additional providers can be swapped in without cross-module coupling.

## Features

- Adaptive ingestion pipeline driven by chunking policies
- Parser abstraction with registry (includes plain text parser; extend for OCR, HTML, etc.)
- Deterministic embedding provider for local development (replace with hosted models in production)
- Tenant-scoped in-memory vector store abstraction
- Hybrid search and lightweight generation facade
- JWT protected REST API with ingestion, search, and generation endpoints
- Structured telemetry hooks for metrics and audit logging
- Python SDK skeleton for typed client integrations

## Running locally

Install dependencies and start the FastAPI application using [uv](https://github.com/astral-sh/uv):

```bash
uv pip install -e .[dev]
uv run uvicorn catalystindex.main:app --reload
```

For end-to-end testing with the asynchronous ingestion queue, Redis/Postgres, and Qdrant, see [`docs/runtime_profiles.md`](docs/runtime_profiles.md). The repo ships with `infrastructure/docker-compose.dev.yml` to boot the supporting services and a helper worker runner under `scripts/run_worker.py`.

Runtime settings can be overridden via environment variables using the pattern `CATALYST_<section>__<field>`. For example:

```bash
export CATALYST_JOBS__store__postgres_dsn=postgresql://catalyst:catalyst@localhost:5432/catalystjobs
export CATALYST_STORAGE__vector_backend=qdrant
```

Operational dashboards and CLI tooling are described in [`docs/observability.md`](docs/observability.md). Use `scripts/catalystctl.py` for quick telemetry snapshots or `scripts/ingest_and_search_demo.py` for manual smoke tests.

## Testing

```bash
uv pip install -e .[dev]
uv run pytest
```

- Smoke tests (FastAPI test client):
  ```bash
  uv run pytest tests/smoke/test_ingest_search_flow.py
  ```
- Integration (requires Qdrant running via docker compose):
  ```bash
  TEST_QDRANT_HOST=localhost TEST_QDRANT_PORT=6333 uv run pytest tests/integration/test_qdrant_vector_store.py
  ```

See [`docs/testing.md`](docs/testing.md) for additional scenarios (perf scripts, markers, CI profile).

## SDK

A lightweight Python SDK is available under `sdk/python/catalyst_index_sdk`. It uses `httpx` and mirrors the ingestion/search/generation APIs so product teams can integrate without re-implementing HTTP calls.

## Extending

- Add new parser adapters under `src/catalystindex/parsers` and register them in `ParserRegistryBuilder`.
- Implement new vector store clients by inheriting from `VectorStoreClient`.
- Provide production embedding providers implementing `EmbeddingProvider`.
- Update policy definitions in `policies/resolver.py` for new document archetypes.

Refer to the specification documents in the repository root for detailed behaviour expectations.
