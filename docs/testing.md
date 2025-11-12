# Testing Guide

## Install dependencies

```bash
uv pip install -e .[dev]
```

## Suites

- **Full suite**: `uv run pytest`
- **Smoke only**: `uv run pytest tests/smoke/test_ingest_search_flow.py`
- **Integration (requires Qdrant from `docker-compose`):**
  ```bash
  docker compose -f infrastructure/docker-compose.dev.yml up -d qdrant
  TEST_QDRANT_HOST=localhost TEST_QDRANT_PORT=6333 uv run pytest tests/integration/test_qdrant_vector_store.py
  ```
- **Performance scripts** (`tests/perf`): run `python tests/perf/ingestion_load.py --iterations 5` (after `uv pip install -e .[dev]`).

## Pytest markers

`pyproject.toml` defines an `integration` marker. Skip integration tests by default:

```bash
uv run pytest -m "not integration"
```
