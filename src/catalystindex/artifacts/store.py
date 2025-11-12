from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Protocol, runtime_checkable

from ..models.common import Tenant


@dataclass(slots=True)
class ArtifactRecord:
    """Represents a stored artifact reference."""

    uri: str
    content_type: str | None
    metadata: Dict[str, object]
    stored_at: datetime


@runtime_checkable
class ArtifactStore(Protocol):
    """Interface for persisting ingestion artifacts."""

    def store_document(
        self,
        tenant: Tenant,
        *,
        job_id: str,
        document_id: str,
        content: bytes,
        content_type: str | None,
        metadata: Dict[str, object] | None = None,
    ) -> ArtifactRecord:
        """Persist the raw document content and return its reference."""


class InMemoryArtifactStore:
    """Stores artifacts in-memory for testing purposes."""

    def __init__(self) -> None:
        self._store: Dict[str, ArtifactRecord] = {}

    def store_document(
        self,
        tenant: Tenant,
        *,
        job_id: str,
        document_id: str,
        content: bytes,
        content_type: str | None,
        metadata: Dict[str, object] | None = None,
    ) -> ArtifactRecord:
        key = self._key(tenant, job_id, document_id)
        meta = dict(metadata or {})
        meta.setdefault("payload_size", len(content))
        record = ArtifactRecord(
            uri=key,
            content_type=content_type,
            metadata=meta,
            stored_at=datetime.utcnow(),
        )
        self._store[key] = record
        return record

    def _key(self, tenant: Tenant, job_id: str, document_id: str) -> str:
        return f"memory://{tenant.org_id}/{tenant.workspace_id}/{job_id}/{document_id}"


class LocalArtifactStore:
    """Writes artifacts to the filesystem under a configured base path."""

    def __init__(self, base_path: str) -> None:
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)

    def store_document(
        self,
        tenant: Tenant,
        *,
        job_id: str,
        document_id: str,
        content: bytes,
        content_type: str | None,
        metadata: Dict[str, object] | None = None,
    ) -> ArtifactRecord:
        tenant_path = self._base_path / tenant.org_id / tenant.workspace_id / job_id
        tenant_path.mkdir(parents=True, exist_ok=True)
        extension = self._extension_from_type(content_type)
        file_path = tenant_path / f"{document_id}{extension}"
        file_path.write_bytes(content)
        record = ArtifactRecord(
            uri=str(file_path),
            content_type=content_type,
            metadata=metadata or {},
            stored_at=datetime.utcnow(),
        )
        return record

    def _extension_from_type(self, content_type: Optional[str]) -> str:
        if not content_type:
            return ".bin"
        mapping = {
            "text/plain": ".txt",
            "text/html": ".html",
            "application/pdf": ".pdf",
        }
        return mapping.get(content_type.lower(), ".bin")


__all__ = [
    "ArtifactRecord",
    "ArtifactStore",
    "InMemoryArtifactStore",
    "LocalArtifactStore",
]

