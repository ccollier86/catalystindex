from __future__ import annotations

import time
from contextlib import contextmanager
from threading import RLock
from typing import Iterator


class WorkloadLimiter:
    """Coarse work limiter to cap concurrent document processing."""

    def __init__(self, *, max_active: int = 0, redis_client: object | None = None, key_prefix: str = "catalystindex"):
        self._max_active = max_active
        self._redis = redis_client
        self._key = f"{key_prefix}:active_docs"
        self._ttl = 180
        self._lock = RLock()
        self._local_tokens: set[str] = set()

    def active_count(self) -> int:
        if self._redis:
            try:
                return int(self._redis.scard(self._key) or 0)
            except Exception:
                return 0
        with self._lock:
            return len(self._local_tokens)

    @contextmanager
    def slot(self, document_id: str, *, wait_timeout: float = 30.0, poll_interval: float = 0.5) -> Iterator[dict]:
        token = ""
        try:
            token = self.acquire(document_id, wait_timeout=wait_timeout, poll_interval=poll_interval)
            yield {"active": self.active_count()}
        finally:
            if token:
                self.release(token)

    def acquire(self, document_id: str, *, wait_timeout: float = 30.0, poll_interval: float = 0.5) -> str:
        if self._max_active and self._max_active < 0:
            raise ValueError("max_active must be >= 0")
        token = f"{document_id}:{int(time.time() * 1000)}"
        if not self._max_active:
            self._store_token(token)
            return token
        deadline = time.monotonic() + max(wait_timeout, 0.0)
        while self.active_count() >= self._max_active:
            if time.monotonic() >= deadline:
                raise TimeoutError("ingestion workers are at capacity; try again later")
            time.sleep(max(poll_interval, 0.05))
        self._store_token(token)
        return token

    def release(self, token: str) -> None:
        if not token:
            return
        if self._redis:
            try:
                self._redis.srem(self._key, token)
            except Exception:
                return
        else:
            with self._lock:
                self._local_tokens.discard(token)

    # Internal helpers -------------------------------------------------
    def _store_token(self, token: str) -> None:
        if self._redis:
            try:
                self._redis.sadd(self._key, token)
                self._redis.expire(self._key, self._ttl)
                return
            except Exception:
                pass
        with self._lock:
            self._local_tokens.add(token)


__all__ = ["WorkloadLimiter"]
