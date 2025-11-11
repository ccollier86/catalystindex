from catalystindex.embeddings.hash import HashEmbeddingProvider
from catalystindex.models.common import ChunkRecord, Tenant
from catalystindex.services.search import SearchOptions, SearchService
from catalystindex.storage.vector_store import InMemoryVectorStore, VectorDocument
from catalystindex.telemetry.logger import AuditLogger, MetricsRecorder


def build_chunk(chunk_id: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        section_slug="body",
        text=text,
        chunk_tier="semantic",
        start_page=1,
        end_page=1,
        bbox_pointer=None,
        summary=None,
        key_terms=[],
        requires_previous=False,
        prev_chunk_id=None,
        confidence_note=None,
        metadata={"policy": "default"},
    )


def test_search_returns_results_sorted_by_score():
    embedding_provider = HashEmbeddingProvider(dimension=32)
    vector_store = InMemoryVectorStore()
    service = SearchService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        audit_logger=AuditLogger(),
        metrics=MetricsRecorder(),
    )
    tenant = Tenant(org_id="org", workspace_id="ws", user_id="user")
    chunks = [
        build_chunk("doc|body|1", "trauma exposure treatment"),
        build_chunk("doc|body|2", "diagnostic criteria ptsd"),
    ]
    embeddings = list(embedding_provider.embed([chunk.text for chunk in chunks]))
    vector_store.upsert(
        tenant,
        [VectorDocument(chunk=chunk, vector=vec, track="text") for chunk, vec in zip(chunks, embeddings)],
    )

    results = service.retrieve(tenant, query="ptsd trauma", options=SearchOptions(limit=2))
    assert len(results) == 2
    assert results[0].score >= results[1].score
