# Runtime Profiles

This guide describes two recommended profiles for running Catalyst Index end-to-end with the asynchronous ingestion queue and Qdrant vector store.

## 1. Async Ingestion Profile

1. **Install dependencies** (includes worker extras):
   ```bash
   pip install -e .[dev,workers,redis]
   ```
2. **Start infra services** (Postgres + Redis) in a separate terminal:
   ```bash
   docker compose -f infrastructure/docker-compose.dev.yml up -d postgres redis
   ```
   The compose file provisions Postgres on `5432` and Redis on `6379`.
3. **Set runtime settings** (via env vars or `.env`):
   ```bash
   export CATALYST_JOBS__store__postgres_dsn=postgresql://catalyst:catalyst@localhost:5432/catalystjobs
   export CATALYST_JOBS__store__redis_url=redis://localhost:6379/0
   export CATALYST_JOBS__worker__enabled=true
   ```
4. **Run API** as usual (`uvicorn catalystindex.main:app --reload`).
5. **Run the ingestion worker**:
   ```bash
   python scripts/run_worker.py
   ```
   The worker listens to the queue defined in `settings.jobs.worker.queue_name` (default `ingestion`).

6. **Smoke test the pipeline** using the helper script (requires a valid JWT):
   ```bash
   python scripts/ingest_and_search_demo.py --base-url http://localhost:8000 --token <JWT>
   ```

7. **Automated smoke test** (optional):
   ```bash
   pytest tests/smoke/test_ingest_search_flow.py
   ```
   This uses FastAPI's in-memory client to ensure ingestion + search work even without the external services.

## 2. Qdrant + Premium Search Profile

1. **Install vector extras**:
   ```bash
   pip install -e .[qdrant]
   ```
2. **Start Qdrant** (and optional Postgres/Redis) via compose:
   ```bash
   docker compose -f infrastructure/docker-compose.dev.yml up -d qdrant
   ```
3. **Update settings** to point at Qdrant:
   ```bash
   export CATALYST_STORAGE__vector_backend=qdrant
   export CATALYST_STORAGE__qdrant__host=localhost
   export CATALYST_STORAGE__qdrant__port=6333
   export CATALYST_STORAGE__qdrant__sparse_vectors=true
   ```
4. **Enable premium reranking (optional)** by configuring the reranker provider:
   ```bash
   export CATALYST_RERANKER__enabled=true
   export CATALYST_RERANKER__provider=cohere
   export CATALYST_RERANKER__api_key=... # Cohere or OpenAI key
   ```
5. Restart the API to pick up settings. Searches will now hit Qdrant collections and run sparse+dense fusion with premium reranking when requested.

> **Note**: Environment variables use the `CATALYST_<section>__<field>` naming convention and are automatically merged into the configuration on startup.
