# Performance & Scenario Smoke Tests

The scripts in this directory provide lightweight load exercises for ingestion and search. They reuse the in-memory service stack so they run quickly in CI while still validating orchestration, tenant propagation, and vector store wiring.

## Prerequisites

```bash
pip install -e .[dev]
```

## Ingestion Load Script

Run a configurable number of ingestion iterations. Each run ingests a curated clinical document and prints latency statistics.

```bash
python tests/perf/ingestion_load.py --iterations 10
```

## Search Load Script

Prime the vector store with sample documents and exercise both economy and premium retrieval modes.

```bash
python tests/perf/search_load.py --iterations 12
```

Both scripts exit with a non-zero code if the pipeline fails to produce chunks or search results, making them suitable for CI gating.
