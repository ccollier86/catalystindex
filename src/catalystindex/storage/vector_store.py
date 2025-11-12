from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

from ..models.common import ChunkRecord, RetrievalResult, Tenant


@dataclass(slots=True)
class VectorDocument:
    chunk: ChunkRecord
    vector: Sequence[float]
    track: str
    sparse_vector: Mapping[int, float] | None = None


class VectorStoreClient:
    """Interface for vector store operations."""

    def upsert(self, tenant: Tenant, documents: Iterable[VectorDocument]) -> None:
        raise NotImplementedError

    def query(
        self,
        tenant: Tenant,
        vector: Sequence[float],
        *,
        track: str,
        limit: int,
        filters: Dict[str, object] | None = None,
        sparse_vector: Mapping[int, float] | None = None,
    ) -> List[RetrievalResult]:
        raise NotImplementedError

    def apply_feedback(
        self,
        tenant: Tenant,
        chunk_ids: Iterable[str],
        *,
        positive: bool,
    ) -> None:
        """Adjust payload metadata based on user feedback."""

        return None


class InMemoryVectorStore(VectorStoreClient):
    """Tenant-aware in-memory vector store supporting cosine similarity."""

    def __init__(self) -> None:
        self._store: Dict[str, List[VectorDocument]] = {}

    def _key(self, tenant: Tenant) -> str:
        return f"{tenant.org_id}:{tenant.workspace_id}"

    def upsert(self, tenant: Tenant, documents: Iterable[VectorDocument]) -> None:
        docs = self._store.setdefault(self._key(tenant), [])
        docs.extend(documents)

    def query(
        self,
        tenant: Tenant,
        vector: Sequence[float],
        *,
        track: str,
        limit: int,
        filters: Dict[str, object] | None = None,
        sparse_vector: Mapping[int, float] | None = None,
    ) -> List[RetrievalResult]:
        docs = self._store.get(self._key(tenant), [])
        if not docs:
            return []
        results: List[RetrievalResult] = []
        for doc in docs:
            if doc.track != track:
                continue
            if filters and not _matches_filters(doc.chunk, filters):
                continue
            denom = float(_norm(doc.vector) * _norm(vector))
            score = _dot(doc.vector, vector) / denom if not math.isclose(denom, 0.0) else 0.0
            results.append(
                RetrievalResult(chunk=doc.chunk, score=score, track=doc.track, vision_context=doc.chunk.metadata.get("vision"))
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    def apply_feedback(
        self,
        tenant: Tenant,
        chunk_ids: Iterable[str],
        *,
        positive: bool,
    ) -> None:
        docs = self._store.get(self._key(tenant), [])
        if not docs:
            return
        adjustment = 1 if positive else -1
        targets = set(chunk_ids)
        if not targets:
            return
        for doc in docs:
            if doc.chunk.chunk_id not in targets:
                continue
            metadata = doc.chunk.metadata
            key = "feedback_positive" if positive else "feedback_negative"
            metadata[key] = int(metadata.get(key, 0)) + 1
            metadata["feedback_score"] = float(metadata.get("feedback_score", 0.0)) + adjustment


class QdrantVectorStore(VectorStoreClient):
    """Vector store implementation backed by Qdrant collections."""

    def __init__(
        self,
        client: object,
        *,
        collection_prefix: str,
        vector_size: int,
        sparse_enabled: bool = False,
        metadata_fields: MutableMapping[str, object] | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient  # type: ignore  # pragma: no cover
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "QdrantVectorStore requires the 'qdrant-client' package. Install optional dependencies to enable it."
            ) from exc

        if not isinstance(client, QdrantClient):  # pragma: no cover - type guard
            raise TypeError("client must be an instance of qdrant_client.QdrantClient")
        from qdrant_client.http import models as rest

        self._client: QdrantClient = client
        self._rest = rest
        self._collection_prefix = collection_prefix.rstrip(":")
        self._vector_size = vector_size
        self._sparse_enabled = sparse_enabled
        self._metadata_fields = metadata_fields or {}
        self._known_tracks: set[str] = set()

    def upsert(self, tenant: Tenant, documents: Iterable[VectorDocument]) -> None:
        grouped: Dict[str, List[VectorDocument]] = defaultdict(list)
        for document in documents:
            grouped[document.track].append(document)
        for track, docs in grouped.items():
            collection = self._collection_name(track)
            self._ensure_collection(collection)
            points = [self._build_point(tenant, doc) for doc in docs]
            self._client.upsert(collection_name=collection, points=points)
            self._known_tracks.add(track)

    def query(
        self,
        tenant: Tenant,
        vector: Sequence[float],
        *,
        track: str,
        limit: int,
        filters: Dict[str, object] | None = None,
        sparse_vector: Mapping[int, float] | None = None,
    ) -> List[RetrievalResult]:
        collection = self._collection_name(track)
        if not self._collection_exists(collection):
            return []
        qdrant_filter = self._build_filter(tenant, track, filters)
        search_params = self._rest.SearchParams(exact=False)
        search_kwargs = {
            "collection_name": collection,
            "query_vector": vector,
            "query_filter": qdrant_filter,
            "limit": limit,
            "search_params": search_params,
        }
        if sparse_vector and self._sparse_enabled:
            sparse_class = getattr(self._rest, "SparseVector", None)
            if sparse_class is None:
                raise RuntimeError("Sparse vector queries require qdrant-client with sparse vector support enabled.")
            search_kwargs["query_sparse_vector"] = sparse_class(
                indices=list(sparse_vector.keys()),
                values=list(sparse_vector.values()),
            )
        results = self._client.search(**search_kwargs)
        retrievals: List[RetrievalResult] = []
        for point in results:
            payload = point.payload or {}
            chunk_payload = payload.get("chunk")
            if not chunk_payload:
                continue
            chunk = ChunkRecord(**chunk_payload)
            retrievals.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(point.score or 0.0),
                    track=payload.get("track", track),
                    vision_context=payload.get("vision_context"),
                )
            )
        retrievals.sort(key=lambda item: item.score, reverse=True)
        return retrievals

    def apply_feedback(
        self,
        tenant: Tenant,
        chunk_ids: Iterable[str],
        *,
        positive: bool,
    ) -> None:
        identifiers = list(chunk_ids)
        if not identifiers:
            return
        point_ids = [self._point_id(tenant, chunk_id) for chunk_id in identifiers]
        adjustment = 1 if positive else -1
        for track in list(self._known_tracks):
            collection = self._collection_name(track)
            if not self._collection_exists(collection):
                continue
            try:
                points = self._client.retrieve(
                    collection_name=collection,
                    ids=point_ids,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception:  # pragma: no cover - passthrough to qdrant
                continue
            if not points:
                continue
            for point in points:
                payload = point.payload or {}
                positive_count = int(payload.get("feedback_positive", 0))
                negative_count = int(payload.get("feedback_negative", 0))
                score = float(payload.get("feedback_score", 0.0))
                if positive:
                    positive_count += 1
                else:
                    negative_count += 1
                score += adjustment
                identifier = getattr(point, "id", None)
                if identifier is None and isinstance(payload, dict):
                    identifier = payload.get("id")
                if identifier is None:
                    continue
                self._client.set_payload(  # type: ignore[call-arg]
                    collection_name=collection,
                    payload={
                        "feedback_positive": positive_count,
                        "feedback_negative": negative_count,
                        "feedback_score": score,
                    },
                    point_ids=[identifier],
                )

    # Internal helpers -------------------------------------------------
    def _collection_name(self, track: str) -> str:
        return f"{self._collection_prefix}_{track}"

    def _point_id(self, tenant: Tenant, chunk_id: str) -> str:
        return f"{tenant.org_id}:{tenant.workspace_id}:{chunk_id}"

    def _ensure_collection(self, collection: str) -> None:
        if self._collection_exists(collection):
            return
        vectors_config = self._rest.VectorParams(size=self._vector_size, distance=self._rest.Distance.COSINE)
        sparse_config = None
        if self._sparse_enabled:
            sparse_params = getattr(self._rest, "SparseVectorParams", None)
            if sparse_params is None:
                raise RuntimeError("Sparse vectors requested but qdrant-client does not support them.")
            index_params = getattr(self._rest, "SparseIndexParams", None)
            sparse_config = (
                sparse_params(index=index_params(on_disk=False)) if index_params else sparse_params()
            )
        self._client.create_collection(
            collection_name=collection,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config,
        )

    def _collection_exists(self, collection: str) -> bool:
        try:
            self._client.get_collection(collection)
            return True
        except Exception:  # pragma: no cover - passthrough to qdrant
            return False

    def _build_point(self, tenant: Tenant, document: VectorDocument):
        from qdrant_client.http import models as rest

        payload = {
            "tenant_org": tenant.org_id,
            "tenant_workspace": tenant.workspace_id,
            "track": document.track,
            "chunk": _chunk_to_payload(document.chunk),
            "chunk_tier": document.chunk.chunk_tier,
            "section_slug": document.chunk.section_slug,
            "vision_context": document.chunk.metadata.get("vision"),
        }
        policy = document.chunk.metadata.get("policy")
        if policy:
            payload["policy"] = policy
        payload.update(self._metadata_fields)
        point_id = f"{tenant.org_id}:{tenant.workspace_id}:{document.chunk.chunk_id}"
        point_payload = rest.PointStruct(  # type: ignore[arg-type]
            id=point_id,
            vector=document.vector,
            payload=payload,
        )
        if document.sparse_vector and self._sparse_enabled:
            sparse_class = getattr(rest, "SparseVector", None)
            if sparse_class is None:
                raise RuntimeError("Sparse vectors requested but qdrant-client does not support them.")
            point_payload.sparse_vector = sparse_class(
                indices=list(document.sparse_vector.keys()),
                values=list(document.sparse_vector.values()),
            )
        return point_payload

    def _build_filter(
        self,
        tenant: Tenant,
        track: str,
        filters: Dict[str, object] | None,
    ):
        from qdrant_client.http import models as rest

        conditions = [
            rest.FieldCondition(key="tenant_org", match=rest.MatchValue(value=tenant.org_id)),
            rest.FieldCondition(key="tenant_workspace", match=rest.MatchValue(value=tenant.workspace_id)),
            rest.FieldCondition(key="track", match=rest.MatchValue(value=track)),
        ]
        for key, value in (filters or {}).items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                conditions.append(rest.FieldCondition(key=key, match=rest.MatchAny(any=list(value))))
            else:
                conditions.append(rest.FieldCondition(key=key, match=rest.MatchValue(value=value)))
        return rest.Filter(must=conditions)


def _chunk_to_payload(chunk: ChunkRecord) -> Dict[str, object]:
    from dataclasses import asdict

    payload = asdict(chunk)
    payload["metadata"] = dict(chunk.metadata)
    payload["key_terms"] = list(chunk.key_terms)
    return payload


def _matches_filters(chunk: ChunkRecord, filters: Dict[str, object]) -> bool:
    for key, value in filters.items():
        if key == "chunk_tier":
            candidate = chunk.chunk_tier
        elif key == "policy":
            candidate = chunk.metadata.get("policy")
        else:
            candidate = chunk.metadata.get(key)
        if isinstance(value, (list, tuple, set)):
            if candidate not in value:
                return False
        elif str(candidate) != str(value):
            return False
    return True


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(l * r for l, r in zip(left, right)))


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


__all__ = ["VectorDocument", "VectorStoreClient", "InMemoryVectorStore", "QdrantVectorStore"]
