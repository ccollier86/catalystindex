from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

from ..models.common import Tenant


class TermIndex(ABC):
    """Interface for alias persistence and expansion."""

    @abstractmethod
    def update(
        self,
        tenant: Tenant,
        knowledge_base_id: str,
        document_id: str,
        chunk_id: str,
        aliases: Iterable[str],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def expand_query(
        self,
        tenant: Tenant,
        query: str,
        *,
        knowledge_base_ids: Sequence[str] | None = None,
        limit: int = 5,
    ) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def record_feedback(
        self,
        tenant: Tenant,
        query: str,
        chunk_ids: Iterable[str],
        *,
        knowledge_base_ids: Sequence[str] | None = None,
        positive: bool,
    ) -> None:
        raise NotImplementedError


class InMemoryTermIndex(TermIndex):
    """Thread-safe in-memory implementation suitable for development and testing."""

    def __init__(self) -> None:
        self._aliases: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._document_terms: Dict[str, Dict[str, List[str]]] = {}

    def update(
        self,
        tenant: Tenant,
        knowledge_base_id: str,
        document_id: str,
        chunk_id: str,
        aliases: Iterable[str],
    ) -> None:
        kb_key = _kb_key(tenant, knowledge_base_id)
        alias_store = self._aliases.setdefault(kb_key, {})
        document_terms = self._document_terms.setdefault(kb_key, {})
        chunk_terms = document_terms.setdefault(chunk_id, [])
        chunk_key = _chunk_key(document_id, chunk_id)
        for rank, alias in enumerate(_normalize_aliases(aliases)):
            weight = max(0.1, 1.0 - rank * 0.1)
            mapping = alias_store.setdefault(alias, {})
            existing = mapping.get(chunk_key, 0.0)
            if weight > existing:
                mapping[chunk_key] = weight
            if alias not in chunk_terms:
                chunk_terms.append(alias)

    def expand_query(
        self,
        tenant: Tenant,
        query: str,
        *,
        knowledge_base_ids: Sequence[str] | None = None,
        limit: int = 5,
    ) -> List[str]:
        if not query.strip():
            return []
        tokens = {token for token in _tokenize(query) if token}
        if not tokens:
            return []
        scored: List[Tuple[str, float]] = []
        for alias_store in self._alias_sources(tenant, knowledge_base_ids):
            for alias, mapping in alias_store.items():
                if alias in tokens:
                    continue
                if not any(token in alias for token in tokens):
                    continue
                total_score = sum(mapping.values())
                scored.append((alias, total_score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [alias for alias, _ in scored[:limit]]

    def record_feedback(
        self,
        tenant: Tenant,
        query: str,
        chunk_ids: Iterable[str],
        *,
        knowledge_base_ids: Sequence[str] | None = None,
        positive: bool,
    ) -> None:
        alias_sources = self._alias_sources(tenant, knowledge_base_ids, include_empty=False)
        if not alias_sources:
            return
        kb_keys = self._kb_keys_for_feedback(tenant, knowledge_base_ids)
        if not kb_keys:
            return
        adjustment = 0.2 if positive else -0.3
        tokens = {token for token in _tokenize(query)}
        chunk_suffixes = {f":{chunk_id}" for chunk_id in chunk_ids}
        for kb_key in kb_keys:
            alias_store = self._aliases.get(kb_key, {})
            if not alias_store:
                continue
            chunk_terms = self._document_terms.get(kb_key, {})
            for alias, mapping in alias_store.items():
                if not tokens.intersection(alias.split()):
                    continue
                for chunk_key in list(mapping):
                    if any(chunk_key.endswith(suffix) for suffix in chunk_suffixes):
                        mapping[chunk_key] = max(0.05, mapping[chunk_key] + adjustment)

    def _alias_sources(
        self,
        tenant: Tenant,
        knowledge_base_ids: Sequence[str] | None,
        include_empty: bool = True,
    ) -> List[Dict[str, Dict[str, float]]]:
        sources: List[Dict[str, Dict[str, float]]] = []
        if knowledge_base_ids:
            keys = [_kb_key(tenant, kb_id) for kb_id in knowledge_base_ids]
        else:
            prefix = _tenant_prefix(tenant)
            keys = [key for key in self._aliases if key.startswith(prefix)]
        for key in keys:
            store = self._aliases.get(key)
            if store or include_empty:
                sources.append(store or {})
        return sources

    def _kb_keys_for_feedback(self, tenant: Tenant, knowledge_base_ids: Sequence[str] | None) -> List[str]:
        if knowledge_base_ids:
            return [_kb_key(tenant, kb_id) for kb_id in knowledge_base_ids]
        prefix = _tenant_prefix(tenant)
        return [key for key in self._document_terms if key.startswith(prefix)]


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

    def update(
        self,
        tenant: Tenant,
        knowledge_base_id: str,
        document_id: str,
        chunk_id: str,
        aliases: Iterable[str],
    ) -> None:
        chunk_key = self._chunk_key(tenant, document_id, chunk_id)
        alias_prefix = self._alias_prefix(tenant, knowledge_base_id)
        doc_terms_key = self._document_terms_key(tenant, knowledge_base_id, chunk_id)
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

    def expand_query(
        self,
        tenant: Tenant,
        query: str,
        *,
        knowledge_base_ids: Sequence[str] | None = None,
        limit: int = 5,
    ) -> List[str]:
        if not query.strip():
            return []
        tokens = {token for token in _tokenize(query) if token}
        if not tokens:
            return []
        scored: List[Tuple[str, float]] = []
        patterns: List[str] = []
        if knowledge_base_ids:
            patterns = [f"{self._alias_prefix(tenant, kb_id)}*" for kb_id in knowledge_base_ids]
        else:
            patterns = [f"{self._alias_glob(tenant)}*"]
        for pattern in patterns:
            for alias_key in self._client.scan_iter(pattern):
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

    def record_feedback(
        self,
        tenant: Tenant,
        query: str,
        chunk_ids: Iterable[str],
        *,
        knowledge_base_ids: Sequence[str] | None = None,
        positive: bool,
    ) -> None:
        tokens = {token for token in _tokenize(query)}
        if not tokens:
            return
        doc_terms_entries: List[Tuple[str, str]] = []
        if knowledge_base_ids:
            for kb_id in knowledge_base_ids:
                for chunk_id in chunk_ids:
                    doc_terms_entries.append((self._document_terms_key(tenant, kb_id, chunk_id), kb_id))
        else:
            for chunk_id in chunk_ids:
                pattern = self._document_terms_glob(tenant, chunk_id)
                for key in self._client.scan_iter(pattern):
                    kb_id = self._extract_kb_from_terms_key(tenant, key)
                    if kb_id:
                        doc_terms_entries.append((key, kb_id))
        if not doc_terms_entries:
            return
        pipe = self._client.pipeline()
        adjustment = 0.2 if positive else -0.3
        chunk_suffixes = tuple(chunk_ids)
        for doc_terms_key, kb_id in doc_terms_entries:
            aliases = self._client.smembers(doc_terms_key)
            if not aliases:
                continue
            alias_prefix = self._alias_prefix(tenant, kb_id)
            for alias in aliases:
                if not tokens.intersection(alias.split()):
                    continue
                alias_key = f"{alias_prefix}{alias}"
                for chunk_id in chunk_suffixes:
                    chunk_key = self._chunk_key_from_identifier(tenant, chunk_id)
                    pipe.hincrbyfloat(alias_key, chunk_key, adjustment)
                    pipe.expire(alias_key, self._ttl)
            pipe.expire(doc_terms_key, self._ttl)
        pipe.execute()

    def _alias_prefix(self, tenant: Tenant, knowledge_base_id: str) -> str:
        return f"termindex:{tenant.org_id}:{tenant.workspace_id}:{knowledge_base_id}:alias:"

    def _alias_glob(self, tenant: Tenant) -> str:
        return f"termindex:{tenant.org_id}:{tenant.workspace_id}:*:alias:"

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

    def _document_terms_key(self, tenant: Tenant, knowledge_base_id: str, chunk_id: str) -> str:
        return f"termindex:{tenant.org_id}:{tenant.workspace_id}:{knowledge_base_id}:chunk:{chunk_id}"

    def _document_terms_glob(self, tenant: Tenant, chunk_id: str) -> str:
        return f"termindex:{tenant.org_id}:{tenant.workspace_id}:*:chunk:{chunk_id}"

    def _extract_kb_from_terms_key(self, tenant: Tenant, key: str) -> str | None:
        prefix = f"termindex:{tenant.org_id}:{tenant.workspace_id}:"
        if not key.startswith(prefix):
            return None
        remainder = key[len(prefix) :]
        parts = remainder.split(":")
        if len(parts) < 2:
            return None
        return parts[0]


def _normalize_aliases(aliases: Iterable[str]) -> List[str]:
    return [alias.strip().lower() for alias in aliases if alias and alias.strip()]


def _tenant_key(tenant: Tenant) -> str:
    return f"{tenant.org_id}:{tenant.workspace_id}"


def _tenant_prefix(tenant: Tenant) -> str:
    return f"{tenant.org_id}:{tenant.workspace_id}:"


def _kb_key(tenant: Tenant, knowledge_base_id: str) -> str:
    return f"{tenant.org_id}:{tenant.workspace_id}:{knowledge_base_id}"


def _chunk_key(document_id: str, chunk_id: str) -> str:
    return f"{document_id}:{chunk_id}"


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]{3,}", text.lower())


__all__ = ["TermIndex", "InMemoryTermIndex", "RedisTermIndex"]
