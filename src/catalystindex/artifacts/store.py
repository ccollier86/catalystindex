from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable

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
            stored_at=datetime.now(timezone.utc),
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
        stored_at = datetime.now(timezone.utc)
        meta = dict(metadata or {})
        meta.setdefault("payload_size", len(content))
        meta_path = file_path.with_suffix(f"{file_path.suffix}.metadata.json")
        meta_payload = {
            "metadata": meta,
            "stored_at": stored_at.isoformat(),
        }
        meta_path.write_text(json.dumps(meta_payload, default=str), encoding="utf-8")
        return ArtifactRecord(
            uri=str(file_path),
            content_type=content_type,
            metadata=meta,
            stored_at=stored_at,
        )

    def _extension_from_type(self, content_type: Optional[str]) -> str:
        if not content_type:
            return ".bin"
        mapping = {
            "text/plain": ".txt",
            "text/html": ".html",
            "application/pdf": ".pdf",
        }
        return mapping.get(content_type.lower(), ".bin")


class S3ArtifactStore:
    """Persists artifacts to S3-compatible object storage."""

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "",
        client: Any | None = None,
        region_name: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        if not bucket:
            raise ValueError("S3ArtifactStore requires a bucket name")
        if client is None:
            try:
                import boto3  # type: ignore
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("S3 artifact backend requires 'boto3' to be installed") from exc
            client = boto3.client("s3", region_name=region_name, endpoint_url=endpoint_url)
        self._client = client
        self._bucket = bucket
        self._prefix = prefix.strip("/")

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
        key = "/".join(
            filter(
                None,
                [
                    self._prefix,
                    tenant.org_id,
                    tenant.workspace_id,
                    job_id,
                    f"{document_id}{self._extension_from_type(content_type)}",
                ],
            )
        )
        meta = dict(metadata or {})
        meta.setdefault("payload_size", len(content))
        stored_at = datetime.now(timezone.utc)
        s3_metadata = {
            "artifact-metadata": json.dumps(meta, default=str),
            "stored-at": stored_at.isoformat(),
        }
        extra_args: Dict[str, Any] = {"Metadata": s3_metadata}
        if content_type:
            extra_args["ContentType"] = content_type
        self._client.put_object(Bucket=self._bucket, Key=key, Body=content, **extra_args)
        return ArtifactRecord(
            uri=f"s3://{self._bucket}/{key}",
            content_type=content_type,
            metadata=meta,
            stored_at=stored_at,
        )

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
    "S3ArtifactStore",
]
