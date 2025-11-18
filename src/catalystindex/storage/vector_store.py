from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence
import uuid
import requests

from ..models.common import ChunkRecord, RetrievalResult, Tenant


@dataclass(slots=True)
class VectorDocument:
    chunk: ChunkRecord
    vector: Sequence[float]
    track: str
    knowledge_base_id: str
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
        knowledge_base_ids: Sequence[str] | None = None,
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
        rest_base_url: str | None = None,
        api_key: str | None = None,
        hnsw_m: int | None = None,
        hnsw_m_final: int | None = None,
        indexing_threshold_kb: int | None = None,
        shard_number: int | None = None,
        on_disk_vectors: bool | None = None,
        hnsw_on_disk: bool | None = None,
        defer_indexing: bool = False,
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
        self._known_collections: set[tuple[str, str]] = set()
        self._supports_sparse_vectors = "sparse_vectors" in getattr(self._rest.PointStruct, "model_fields", {})  # type: ignore[attr-defined]
        self._rest_base_url = rest_base_url.rstrip("/") if rest_base_url else None
        self._api_key = api_key
        self._hnsw_m = hnsw_m
        self._hnsw_m_final = hnsw_m_final
        self._indexing_threshold_kb = indexing_threshold_kb
        self._shard_number = shard_number
        self._on_disk_vectors = on_disk_vectors
        self._hnsw_on_disk = hnsw_on_disk
        self._defer_indexing = defer_indexing

    def upsert(self, tenant: Tenant, documents: Iterable[VectorDocument]) -> None:
        grouped: Dict[tuple[str, str], List[VectorDocument]] = defaultdict(list)
        for document in documents:
            if not document.knowledge_base_id:
                raise ValueError("knowledge_base_id is required for Qdrant upsert")
            grouped[(document.track, document.knowledge_base_id)].append(document)
        for (track, kb_id), docs in grouped.items():
            collection = self._collection_name(track, kb_id)
            self._ensure_collection(collection)
            batch_size = 16
            use_client_sparse = self._sparse_enabled and self._supports_sparse_vectors
            for start in range(0, len(docs), batch_size):
                batch_docs = docs[start : start + batch_size]
                retries = 3
                last_error: Exception | None = None
                for attempt in range(1, retries + 1):
                    try:
                        if use_client_sparse:
                            points = [self._build_point(tenant, doc) for doc in batch_docs]
                            self._client.upsert(collection_name=collection, points=points, wait=True)
                        else:
                            self._upsert_via_http(collection, tenant, batch_docs)
                        last_error = None
                        break
                    except Exception as exc:  # pragma: no cover - passthrough to qdrant
                        last_error = exc
                        if attempt < retries:
                            continue
                        raise
            self._known_tracks.add(track)
            self._known_collections.add((track, kb_id))
            # If indexing was deferred, re-enable after ingest for this collection.
            if self._defer_indexing:
                self.finalize_index(collection)

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
        filters = dict(filters or {})
        kb_filters = self._extract_kb_filters(filters)
        aggregated: List[RetrievalResult] = []
        dense_results: List[RetrievalResult] = []
        sparse_results: List[RetrievalResult] = []
        for kb_id in kb_filters:
            collection = self._collection_name(track, kb_id)
            if not self._collection_exists(collection):
                continue
            qdrant_filter = self._build_filter(tenant, track, filters, kb_id)
            search_params = self._rest.SearchParams(exact=False)
            if self._sparse_enabled:
                dense_query = self._rest.NamedVector(name="text_dense", vector=vector)
            else:
                dense_query = vector
            dense_kwargs = {
                "collection_name": collection,
                "query_vector": dense_query,
                "query_filter": qdrant_filter,
                "limit": limit,
                "search_params": search_params,
            }
            dense_results.extend(self._client.search(**dense_kwargs))

            if sparse_vector and self._sparse_enabled:
                sparse_class = getattr(self._rest, "SparseVector", None)
                named_sparse_class = getattr(self._rest, "NamedSparseVector", None)
                if sparse_class is None or named_sparse_class is None:
                    raise RuntimeError("Sparse vector queries require qdrant-client with sparse vector support enabled.")
                sparse = named_sparse_class(
                    name="text_sparse",
                    vector=sparse_class(indices=list(sparse_vector.keys()), values=list(sparse_vector.values())),
                )
                sparse_kwargs = {
                    "collection_name": collection,
                    "query_vector": sparse,
                    "query_filter": qdrant_filter,
                    "limit": limit,
                    "search_params": search_params,
                }
                sparse_results.extend(self._client.search(**sparse_kwargs))

        retrievals = self._build_retrievals(dense_results, track) if dense_results else []
        if sparse_results:
            sparse_retrievals = self._build_retrievals(sparse_results, track)
            retrievals = self._rrf_merge(retrievals, sparse_retrievals, limit)
        retrievals.sort(key=lambda item: item.score, reverse=True)
        return retrievals[:limit]

    def _build_retrievals(self, results: Sequence[object], track: str) -> List[RetrievalResult]:
        retrievals: List[RetrievalResult] = []
        for point in results:
            payload = getattr(point, "payload", None) or {}
            chunk_payload = payload.get("chunk")
            if not chunk_payload:
                continue
            chunk = ChunkRecord(**chunk_payload)
            feedback_positive = payload.get("feedback_positive")
            feedback_negative = payload.get("feedback_negative")
            feedback_score = payload.get("feedback_score")
            if any(value is not None for value in (feedback_positive, feedback_negative, feedback_score)):
                meta = dict(chunk.metadata)
                if feedback_positive is not None:
                    meta["feedback_positive"] = feedback_positive
                if feedback_negative is not None:
                    meta["feedback_negative"] = feedback_negative
                if feedback_score is not None:
                    meta["feedback_score"] = feedback_score
                chunk = ChunkRecord(
                    chunk_id=chunk.chunk_id,
                    section_slug=chunk.section_slug,
                    text=chunk.text,
                    chunk_tier=chunk.chunk_tier,
                    start_page=chunk.start_page,
                    end_page=chunk.end_page,
                    bbox_pointer=chunk.bbox_pointer,
                    summary=chunk.summary,
                    key_terms=chunk.key_terms,
                    requires_previous=chunk.requires_previous,
                    prev_chunk_id=chunk.prev_chunk_id,
                    confidence_note=chunk.confidence_note,
                    metadata=meta,
                )
            retrievals.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(getattr(point, "score", 0.0) or 0.0),
                    track=payload.get("track", track),
                    vision_context=payload.get("vision_context"),
                )
            )
        return retrievals

    def _rrf_merge(self, dense: Sequence[RetrievalResult], sparse: Sequence[RetrievalResult], limit: int) -> List[RetrievalResult]:
        rrf_constant = 60
        ranked: Dict[str, tuple[float, RetrievalResult]] = {}
        for records in (dense, sparse):
            for rank, result in enumerate(records, start=1):
                key = result.chunk.chunk_id
                score = 1.0 / (rrf_constant + rank)
                combined = float(result.score) + score
                current = ranked.get(key)
                if not current or combined > current[0]:
                    ranked[key] = (combined, result)
        merged = [item[1] for item in sorted(ranked.values(), key=lambda pair: pair[0], reverse=True)]
        return merged[:limit]

    def apply_feedback(
        self,
        tenant: Tenant,
        chunk_ids: Iterable[str],
        *,
        positive: bool,
        knowledge_base_ids: Sequence[str] | None = None,
    ) -> None:
        identifiers = list(chunk_ids)
        if not identifiers:
            return
        point_ids = [self._point_id(tenant, chunk_id) for chunk_id in identifiers]
        adjustment = 1 if positive else -1
        for track, kb_id in list(self._known_collections):
            if knowledge_base_ids and kb_id not in knowledge_base_ids:
                continue
            collection = self._collection_name(track, kb_id)
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
    def _collection_name(self, track: str, knowledge_base_id: str) -> str:
        safe_kb = knowledge_base_id.replace(":", "-")
        return f"{self._collection_prefix}_{track}_{safe_kb}"

    def _point_id(self, tenant: Tenant, chunk_id: str) -> str:
        base = f"{tenant.org_id}:{tenant.workspace_id}:{chunk_id}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, base))

    def _upsert_via_http(self, collection: str, tenant: Tenant, docs: Sequence[VectorDocument]) -> None:
        if not self._rest_base_url:
            raise RuntimeError("Sparse vectors requested but no rest_base_url configured for HTTP upsert.")
        points: List[dict] = []
        for doc in docs:
            vector_payload = {"text_dense": doc.vector} if self._sparse_enabled else doc.vector
            point = {
                "id": self._point_id(tenant, doc.chunk.chunk_id),
                "vector": vector_payload,
                "payload": {
                    "tenant_org": tenant.org_id,
                    "tenant_workspace": tenant.workspace_id,
                    "track": doc.track,
                    "chunk": _chunk_to_payload(doc.chunk),
                    "chunk_tier": doc.chunk.chunk_tier,
                    "section_slug": doc.chunk.section_slug,
                    "vision_context": doc.chunk.metadata.get("vision"),
                    "knowledge_base_id": doc.knowledge_base_id,
                },
            }
            if doc.chunk.metadata.get("policy"):
                point["payload"]["policy"] = doc.chunk.metadata["policy"]
            point["payload"].update(self._metadata_fields)
            if doc.sparse_vector:
                point.setdefault("sparse_vectors", {})["text_sparse"] = {
                    "indices": list(doc.sparse_vector.keys()),
                    "values": list(doc.sparse_vector.values()),
                }
            points.append(point)
        url = f"{self._rest_base_url}/collections/{collection}/points?wait=true"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["api-key"] = self._api_key
        response = requests.put(url, json={"points": points}, headers=headers, timeout=300)
        if not response.ok:
            raise RuntimeError(f"HTTP upsert failed: {response.status_code} {response.text}")

    def _ensure_collection(self, collection: str) -> None:
        if self._collection_exists(collection):
            try:
                info = self._client.get_collection(collection)
                params = getattr(info.config, "params", None)
                vectors_cfg = getattr(params, "vectors", None)
                named_cfg = getattr(params, "vectors_configs", None)
                sparse_cfg = getattr(params, "sparse_vectors_config", None)
                # If tuning params mismatch (e.g., HNSW m or on_disk), allow recreate via drop.
                collection_mismatch = False
                if self._hnsw_m is not None:
                    current_hnsw = getattr(params, "hnsw_config", None)
                    current_m = getattr(current_hnsw, "m", None) if current_hnsw else None
                    if current_m is not None and current_m != self._hnsw_m:
                        collection_mismatch = True
                if self._sparse_enabled:
                    # require named dense and named sparse
                    has_dense = bool(named_cfg and "text_dense" in named_cfg)
                    has_sparse = bool(sparse_cfg and "text_sparse" in sparse_cfg)
                    if has_dense and has_sparse and not collection_mismatch:
                        return
                else:
                    # dense-only collection
                    if isinstance(vectors_cfg, self._rest.VectorParams) and not collection_mismatch:
                        return
            except Exception:
                pass
            # Drop and recreate with correct schema
            self._client.delete_collection(collection)
        vector_params_kwargs = {"size": self._vector_size, "distance": self._rest.Distance.COSINE}
        if self._on_disk_vectors is not None:
            vector_params_kwargs["on_disk"] = self._on_disk_vectors
        indexing_threshold = self._indexing_threshold_kb
        hnsw_m = self._hnsw_m
        if self._defer_indexing:
            hnsw_m = 0
            indexing_threshold = 0
        if self._sparse_enabled:
            vectors_config = {"text_dense": self._rest.VectorParams(**vector_params_kwargs)}
            sparse_params = getattr(self._rest, "SparseVectorParams", None)
            if sparse_params is None:
                raise RuntimeError("Sparse vectors requested but qdrant-client does not support them.")
            index_params = getattr(self._rest, "SparseIndexParams", None)
            params = sparse_params(index=index_params(on_disk=False)) if index_params else sparse_params()
            sparse_config = {"text_sparse": params}
        else:
            vectors_config = self._rest.VectorParams(**vector_params_kwargs)
            sparse_config = None
        create_kwargs = {
            "collection_name": collection,
            "vectors_config": vectors_config,
            "sparse_vectors_config": sparse_config,
        }
        if hnsw_m is not None:
            create_kwargs["hnsw_config"] = self._rest.HnswConfigDiff(m=hnsw_m, on_disk=self._hnsw_on_disk)
        if indexing_threshold is not None:
            create_kwargs["optimizers_config"] = self._rest.OptimizersConfigDiff(indexing_threshold_kb=indexing_threshold)
        if self._shard_number is not None:
            create_kwargs["shard_number"] = self._shard_number
        self._client.create_collection(**create_kwargs)

    def finalize_index(self, collection: str) -> None:
        """Re-enable indexing after deferred ingest."""
        if not self._defer_indexing:
            return
        m_final = self._hnsw_m_final or 16
        payload = {"hnsw_config": {"m": m_final}}
        if self._indexing_threshold_kb:
            payload["optimizers_config"] = {"indexing_threshold_kb": self._indexing_threshold_kb}
        self._client.update_collection(collection_name=collection, **payload)

    def _collection_exists(self, collection: str) -> bool:
        try:
            self._client.get_collection(collection)
            return True
        except Exception:  # pragma: no cover - passthrough to qdrant
            return False

    def _extract_kb_filters(self, filters: Dict[str, object]) -> List[str]:
        value = filters.pop("knowledge_base_id", None)
        if value is None:
            raise ValueError("knowledge_base_id filter is required for Qdrant queries")
        if isinstance(value, str):
            return [value]
        if isinstance(value, Iterable):
            result = []
            for item in value:
                if item:
                    result.append(str(item))
            return result
        return []

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
        payload["knowledge_base_id"] = document.knowledge_base_id
        policy = document.chunk.metadata.get("policy")
        if policy:
            payload["policy"] = policy
        payload.update(self._metadata_fields)
        point_id = self._point_id(tenant, document.chunk.chunk_id)
        vector_payload = {"text_dense": document.vector} if self._sparse_enabled else document.vector
        point_kwargs = {
            "id": point_id,
            "vector": vector_payload,
            "payload": payload,
        }
        if document.sparse_vector and self._sparse_enabled and self._supports_sparse_vectors:
            sparse_class = getattr(rest, "SparseVector", None)
            if sparse_class is None:
                raise RuntimeError("Sparse vectors requested but qdrant-client does not support them.")
            sparse_payload = sparse_class(
                indices=list(document.sparse_vector.keys()),
                values=list(document.sparse_vector.values()),
            )
            point_kwargs["sparse_vectors"] = {"text_sparse": sparse_payload}
        point_payload = rest.PointStruct(**point_kwargs)  # type: ignore[arg-type]
        return point_payload

    def _build_filter(
        self,
        tenant: Tenant,
        track: str,
        filters: Dict[str, object] | None,
        knowledge_base_id: str,
    ):
        from qdrant_client.http import models as rest

        conditions = [
            rest.FieldCondition(key="tenant_org", match=rest.MatchValue(value=tenant.org_id)),
            rest.FieldCondition(key="tenant_workspace", match=rest.MatchValue(value=tenant.workspace_id)),
            rest.FieldCondition(key="track", match=rest.MatchValue(value=track)),
            rest.FieldCondition(key="knowledge_base_id", match=rest.MatchValue(value=knowledge_base_id)),
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
        elif key == "knowledge_base_id":
            candidate = chunk.metadata.get("knowledge_base_id")
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
