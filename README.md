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

Install dependencies and start the FastAPI application:

```bash
pip install -e .[dev]
uvicorn catalystindex.main:app --reload
```

## Testing

```bash
pytest
```

## SDK

A lightweight Python SDK is available under `sdk/python/catalyst_index_sdk`. It uses `httpx` and mirrors the ingestion/search/generation APIs so product teams can integrate without re-implementing HTTP calls.

## Extending

- Add new parser adapters under `src/catalystindex/parsers` and register them in `ParserRegistryBuilder`.
- Implement new vector store clients by inheriting from `VectorStoreClient`.
- Provide production embedding providers implementing `EmbeddingProvider`.
- Update policy definitions in `policies/resolver.py` for new document archetypes.

Refer to the specification documents in the repository root for detailed behaviour expectations.
