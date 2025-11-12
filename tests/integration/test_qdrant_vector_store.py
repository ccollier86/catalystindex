from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

import pytest

from catalystindex.models.common import ChunkRecord, Tenant
from catalystindex.storage.vector_store import QdrantVectorStore, VectorDocument

pytestmark = pytest.mark.integration


@dataclass
class _TrackConfig:
    name: str = "text"


@pytest.fixture(scope="module")
def qdrant_connection():
    qdrant = pytest.importorskip("qdrant_client")
    host = os.getenv("TEST_QDRANT_HOST", "localhost")
    port = int(os.getenv("TEST_QDRANT_PORT", "6333"))
    client = qdrant.QdrantClient(host=host, port=port)
    try:
        client.get_collections()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Qdrant not available: {exc}")
    return client


@pytest.fixture()
def qdrant_store(qdrant_connection):
    client = qdrant_connection
    prefix = f"ci_test_{uuid.uuid4().hex[:8]}"
    store = QdrantVectorStore(client, collection_prefix=prefix, vector_size=4, metadata_fields={"environment": "test"})
    yield store
    for suffix in ("text", "vision"):
        collection = f"{prefix}_{suffix}"
        try:
            client.delete_collection(collection_name=collection)
        except Exception:
            continue


def _chunk(chunk_id: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        section_slug="body",
        text="sample criteria text",
        chunk_tier="criteria",
        start_page=1,
        end_page=1,
        bbox_pointer=None,
        summary="criteria",
        key_terms=["criteria"],
        requires_previous=False,
        prev_chunk_id=None,
        confidence_note=None,
        metadata={"policy": "dsm5"},
    )


def test_qdrant_vector_store_round_trip(qdrant_store):
    tenant = Tenant(org_id="org", workspace_id="ws", user_id="tester")
    doc = VectorDocument(chunk=_chunk("chunk-1"), vector=[0.1, 0.3, 0.4, 0.2], track="text")
    qdrant_store.upsert(tenant, [doc])
    results = qdrant_store.query(
        tenant,
        [0.1, 0.2, 0.3, 0.4],
        track="text",
        limit=5,
        filters={"policy": "dsm5", "chunk_tier": "criteria"},
    )
    assert results, "Expected at least one retrieval"
    assert results[0].chunk.chunk_id == "chunk-1"
