from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple

from ..models.common import Tenant


class TermIndex(ABC):
    """Interface for alias persistence and expansion."""

    @abstractmethod
    def update(self, tenant: Tenant, document_id: str, chunk_id: str, aliases: Iterable[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def expand_query(self, tenant: Tenant, query: str, *, limit: int = 5) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def record_feedback(
        self, tenant: Tenant, query: str, chunk_ids: Iterable[str], *, positive: bool
    ) -> None:
        raise NotImplementedError


class InMemoryTermIndex(TermIndex):
    """Thread-safe in-memory implementation suitable for development and testing."""

    def __init__(self) -> None:
        self._aliases: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._document_terms: Dict[str, Dict[str, Dict[str, List[str]]]] = {}

    def update(self, tenant: Tenant, document_id: str, chunk_id: str, aliases: Iterable[str]) -> None:
        tenant_key = _tenant_key(tenant)
        alias_store = self._aliases.setdefault(tenant_key, {})
        document_terms = self._document_terms.setdefault(tenant_key, {})
        chunk_terms = document_terms.setdefault(document_id, {}).setdefault(chunk_id, [])
        chunk_key = _chunk_key(document_id, chunk_id)
        for rank, alias in enumerate(_normalize_aliases(aliases)):
            weight = max(0.1, 1.0 - rank * 0.1)
            mapping = alias_store.setdefault(alias, {})
            existing = mapping.get(chunk_key, 0.0)
            if weight > existing:
                mapping[chunk_key] = weight
            if alias not in chunk_terms:
                chunk_terms.append(alias)

    def expand_query(self, tenant: Tenant, query: str, *, limit: int = 5) -> List[str]:
        tenant_key = _tenant_key(tenant)
        alias_store = self._aliases.get(tenant_key, {})
        if not alias_store or not query.strip():
            return []
        tokens = {token for token in _tokenize(query) if token}
        if not tokens:
            return []
        scored: List[Tuple[str, float]] = []
        for alias, mapping in alias_store.items():
            if alias in tokens:
                continue
            if not any(token in alias for token in tokens):
                continue
            total_score = sum(mapping.values())
            scored.append((alias, total_score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [alias for alias, _ in scored[:limit]]

    def record_feedback(self, tenant: Tenant, query: str, chunk_ids: Iterable[str], *, positive: bool) -> None:
        tenant_key = _tenant_key(tenant)
        alias_store = self._aliases.get(tenant_key, {})
        if not alias_store:
            return
        adjustment = 0.2 if positive else -0.3
        tokens = {token for token in _tokenize(query)}
        chunk_suffixes = {f":{chunk_id}" for chunk_id in chunk_ids}
        for alias, mapping in alias_store.items():
            if not tokens.intersection(alias.split()):
                continue
            for chunk_key in list(mapping):
                if any(chunk_key.endswith(suffix) for suffix in chunk_suffixes):
                    mapping[chunk_key] = max(0.05, mapping[chunk_key] + adjustment)


@dataclass(slots=True)
class RedisClientAdapter:
    """Minimal interface required from redis clients used by :class:`RedisTermIndex`."""

    client: object

    def pipeline(self):  # pragma: no cover - passthrough to redis client
        return getattr(self.client, "pipeline")()

    def scan_iter(self, match: str) -> Iterator[str]:  # pragma: no cover - passthrough
        for value in getattr(self.client, "scan_iter")(match):
            yield value.decode() if isinstance(value, bytes) else value

    def hgetall(self, key: str) -> Dict[str, float]:
        raw = getattr(self.client, "hgetall")(key)
        result: Dict[str, float] = {}
        for chunk_key, score in raw.items():
            if isinstance(chunk_key, bytes):
                chunk_key = chunk_key.decode()
            if isinstance(score, bytes):
                score = score.decode()
            result[chunk_key] = float(score)
        return result

    def smembers(self, key: str) -> List[str]:
        raw = getattr(self.client, "smembers")(key)
        members: List[str] = []
        for value in raw:
            members.append(value.decode() if isinstance(value, bytes) else value)
        return members


class RedisTermIndex(TermIndex):
    """Redis-backed alias index with TTL support and tenant isolation."""

    def __init__(self, client: object, *, ttl_seconds: int = 60 * 60 * 24 * 7) -> None:
        try:
            import redis  # type: ignore  # pragma: no cover
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "RedisTermIndex requires the 'redis' package. Install optional dependencies to enable it."
            ) from exc

        if not hasattr(client, "pipeline"):
            raise TypeError("RedisTermIndex client must provide pipeline support")
        self._client = RedisClientAdapter(client)
        self._ttl = ttl_seconds

    def update(self, tenant: Tenant, document_id: str, chunk_id: str, aliases: Iterable[str]) -> None:
        chunk_key = self._chunk_key(tenant, document_id, chunk_id)
        alias_prefix = self._alias_prefix(tenant)
        doc_terms_key = self._document_terms_key(tenant, chunk_id)
        pipe = self._client.pipeline()
        normalized_aliases = list(_normalize_aliases(aliases))
        for rank, alias in enumerate(normalized_aliases):
            weight = max(0.1, 1.0 - rank * 0.1)
            alias_key = f"{alias_prefix}{alias}"
            pipe.hset(alias_key, chunk_key, weight)
            pipe.expire(alias_key, self._ttl)
        if normalized_aliases:
            pipe.sadd(doc_terms_key, *normalized_aliases)
            pipe.expire(doc_terms_key, self._ttl)
        pipe.execute()

    def expand_query(self, tenant: Tenant, query: str, *, limit: int = 5) -> List[str]:
        if not query.strip():
            return []
        tokens = {token for token in _tokenize(query) if token}
        if not tokens:
            return []
        alias_prefix = self._alias_prefix(tenant)
        scored: List[Tuple[str, float]] = []
        for alias_key in self._client.scan_iter(f"{alias_prefix}*"):
            alias = alias_key.split(":")[-1]
            if alias in tokens:
                continue
            if not any(token in alias for token in tokens):
                continue
            mapping = self._client.hgetall(alias_key)
            total_score = sum(mapping.values())
            if total_score:
                scored.append((alias, total_score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [alias for alias, _ in scored[:limit]]

    def record_feedback(self, tenant: Tenant, query: str, chunk_ids: Iterable[str], *, positive: bool) -> None:
        tokens = {token for token in _tokenize(query)}
        if not tokens:
            return
        doc_terms_keys = [self._document_terms_key(tenant, chunk_id) for chunk_id in chunk_ids]
        alias_prefix = self._alias_prefix(tenant)
        pipe = self._client.pipeline()
        adjustment = 0.2 if positive else -0.3
        for doc_terms_key in doc_terms_keys:
            for alias in self._client.smembers(doc_terms_key):
                if not tokens.intersection(alias.split()):
                    continue
                alias_key = f"{alias_prefix}{alias}"
                for chunk_id in chunk_ids:
                    chunk_key = self._chunk_key_from_identifier(tenant, chunk_id)
                    pipe.hincrbyfloat(alias_key, chunk_key, adjustment)
                    pipe.expire(alias_key, self._ttl)
        pipe.execute()

    def _alias_prefix(self, tenant: Tenant) -> str:
        return f"termindex:{tenant.org_id}:{tenant.workspace_id}:alias:"

    def _chunk_key(self, tenant: Tenant, document_id: str, chunk_id: str) -> str:
        return f"{tenant.org_id}:{tenant.workspace_id}:{document_id}:{chunk_id}"

    def _chunk_key_from_identifier(self, tenant: Tenant, identifier: str) -> str:
        parts = identifier.split(":")
        if len(parts) >= 4:
            return identifier
        if len(parts) == 2:
            document_id, chunk_id = parts
            return self._chunk_key(tenant, document_id, chunk_id)
        chunk_id = parts[-1]
        return self._chunk_key(tenant, document_id="", chunk_id=chunk_id)

    def _document_terms_key(self, tenant: Tenant, chunk_id: str) -> str:
        return f"termindex:{tenant.org_id}:{tenant.workspace_id}:chunk:{chunk_id}"


def _normalize_aliases(aliases: Iterable[str]) -> List[str]:
    return [alias.strip().lower() for alias in aliases if alias and alias.strip()]


def _tenant_key(tenant: Tenant) -> str:
    return f"{tenant.org_id}:{tenant.workspace_id}"


def _chunk_key(document_id: str, chunk_id: str) -> str:
    return f"{document_id}:{chunk_id}"


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]{3,}", text.lower())


__all__ = ["TermIndex", "InMemoryTermIndex", "RedisTermIndex"]

