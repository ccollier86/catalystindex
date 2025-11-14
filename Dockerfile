FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir uv && \
    rm -rf fastapi pydantic && \
    uv pip install --system --no-cache -e .[qdrant,openai,workers,redis,ingestion] fastapi uvicorn

EXPOSE 8000

CMD ["uvicorn", "catalystindex.main:app", "--host", "0.0.0.0", "--port", "8000"]
