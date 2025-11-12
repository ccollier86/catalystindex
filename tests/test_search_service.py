from catalystindex.embeddings.hash import HashEmbeddingProvider
from catalystindex.models.common import ChunkRecord, Tenant
from catalystindex.services.search import EmbeddingReranker, SearchOptions, SearchService
from catalystindex.storage.term_index import InMemoryTermIndex
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
    term_index = InMemoryTermIndex()
    service = SearchService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        term_index=term_index,
        audit_logger=AuditLogger(),
        metrics=MetricsRecorder(),
        reranker=EmbeddingReranker(embedding_provider),
    )
    tenant = Tenant(org_id="org", workspace_id="ws", user_id="user")
    chunks = [
        build_chunk("doc|body|1", "trauma exposure treatment"),
        build_chunk("doc|body|2", "diagnostic criteria ptsd"),
    ]
    for chunk in chunks:
        chunk.key_terms = ["post-traumatic stress"]
        term_index.update(tenant, "doc", chunk.chunk_id, chunk.key_terms)
    embeddings = list(embedding_provider.embed([chunk.text for chunk in chunks]))
    vector_store.upsert(
        tenant,
        [VectorDocument(chunk=chunk, vector=vec, track="text") for chunk, vec in zip(chunks, embeddings)],
    )

    execution = service.retrieve(tenant, query="ptsd trauma", options=SearchOptions(limit=2))
    assert len(execution.results) == 2
    assert execution.results[0].score >= execution.results[1].score
    assert execution.explanations


def test_search_debug_information_includes_aliases_and_tracks():
    embedding_provider = HashEmbeddingProvider(dimension=32)
    vector_store = InMemoryVectorStore()
    term_index = InMemoryTermIndex()
    service = SearchService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        term_index=term_index,
        audit_logger=AuditLogger(),
        metrics=MetricsRecorder(),
    )
    tenant = Tenant(org_id="org", workspace_id="ws", user_id="user")
    chunk = build_chunk("doc|body|1", "behavioral activation helps")
    chunk.key_terms = ["behavioral activation", "pleasant activities scheduling"]
    term_index.update(tenant, "doc", chunk.chunk_id, chunk.key_terms)
    embedding = next(iter(embedding_provider.embed([chunk.text])))
    vector_store.upsert(tenant, [VectorDocument(chunk=chunk, vector=embedding, track="text")])

    execution = service.retrieve(
        tenant,
        query="behavioral activation",
        options=SearchOptions(limit=1, debug=True, alias_limit=5),
    )
    assert execution.debug is not None
    assert "behavioral" in execution.debug.expanded_query
    assert execution.debug.alias_terms
    assert execution.debug.tracks


def test_premium_mode_uses_custom_reranker():
    class ReverseReranker:
        def rerank(self, query, results, *, limit):
            return list(reversed(results))[:limit]

    embedding_provider = HashEmbeddingProvider(dimension=32)
    vector_store = InMemoryVectorStore()
    tenant = Tenant(org_id="org", workspace_id="ws", user_id="user")
    chunks = [
        build_chunk("doc|body|1", "first chunk"),
        build_chunk("doc|body|2", "second chunk"),
    ]
    embeddings = list(embedding_provider.embed([chunk.text for chunk in chunks]))
    vector_store.upsert(
        tenant,
        [VectorDocument(chunk=chunk, vector=vec, track="text") for chunk, vec in zip(chunks, embeddings)],
    )

    baseline_service = SearchService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        term_index=None,
        audit_logger=AuditLogger(),
        metrics=MetricsRecorder(),
    )
    baseline = baseline_service.retrieve(
        tenant,
        query="query",
        options=SearchOptions(limit=2, mode="premium"),
    )
    baseline_order = [result.chunk.chunk_id for result in baseline.results]

    service = SearchService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        term_index=None,
        audit_logger=AuditLogger(),
        metrics=MetricsRecorder(),
        reranker=ReverseReranker(),
    )

    execution = service.retrieve(
        tenant,
        query="query",
        options=SearchOptions(limit=2, mode="premium"),
    )
    assert [result.chunk.chunk_id for result in execution.results] == list(reversed(baseline_order))
