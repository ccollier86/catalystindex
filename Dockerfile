FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        libmagic1 \
        poppler-utils \
        libxml2 \
        libxslt1.1 \
        libjpeg62-turbo \
        zlib1g \
        libpng16-16 \
        libopenjp2-7 \
        tesseract-ocr \
        ffmpeg \
        libcairo2 \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libheif1 \
        ghostscript && \
    rm -rf /var/lib/apt/lists/*

COPY . .

RUN rm -rf pydantic && \
    pip install --no-cache-dir pydantic==1.10.14 qdrant-client==1.15.1 && \
    pip install --no-cache-dir cohere && \
    pip install --no-cache-dir -e .[qdrant,openai,workers,redis,ingestion] uvicorn

EXPOSE 8000

CMD ["uvicorn", "catalystindex.main:app", "--host", "0.0.0.0", "--port", "8000"]
