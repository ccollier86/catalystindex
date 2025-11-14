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

    def store_json_artifact(
        self,
        tenant: Tenant,
        *,
        job_id: str,
        document_id: str,
        name: str,
        payload: object,
    ) -> ArtifactRecord:
        """Persist a structured artifact (e.g., chunks.json) and return its reference."""

    def load_json_artifact(
        self,
        tenant: Tenant,
        *,
        job_id: str,
        document_id: str,
        name: str,
    ) -> object | None:
        """Load a previously stored structured artifact if it exists."""

    def json_artifact_exists(
        self,
        tenant: Tenant,
        *,
        job_id: str,
        document_id: str,
        name: str,
    ) -> bool:
        """Return True if the structured artifact exists."""


class InMemoryArtifactStore:
    """Stores artifacts in-memory for testing purposes."""

    def __init__(self) -> None:
        self._store: Dict[str, ArtifactRecord] = {}
        self._json_assets: Dict[str, Dict[str, object]] = {}

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

    def store_json_artifact(
        self,
        tenant: Tenant,
        *,
        job_id: str,
        document_id: str,
        name: str,
        payload: object,
    ) -> ArtifactRecord:
        key = self._key(tenant, job_id, document_id)
        assets = self._json_assets.setdefault(key, {})
        assets[name] = payload
        record = ArtifactRecord(
            uri=f"{key}#{name}",
            content_type="application/json",
            metadata={"asset": name},
            stored_at=datetime.now(timezone.utc),
        )
        return record

    def load_json_artifact(
        self,
        tenant: Tenant,
        *,
        job_id: str,
        document_id: str,
        name: str,
    ) -> object | None:
        key = self._key(tenant, job_id, document_id)
        return self._json_assets.get(key, {}).get(name)

    def json_artifact_exists(
        self,
        tenant: Tenant,
        *,
        job_id: str,
        document_id: str,
        name: str,
    ) -> bool:
        return self.load_json_artifact(tenant, job_id=job_id, document_id=document_id, name=name) is not None


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
        file_path = self._document_path(tenant, job_id, document_id, content_type)
        file_path.parent.mkdir(parents=True, exist_ok=True)
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

    def _document_path(
        self,
        tenant: Tenant,
        job_id: str,
        document_id: str,
        content_type: Optional[str],
    ) -> Path:
        tenant_path = self._base_path / tenant.org_id / tenant.workspace_id / job_id
        extension = self._extension_from_type(content_type)
        return tenant_path / f"{document_id}{extension}"

    def _json_path(self, tenant: Tenant, job_id: str, document_id: str, name: str) -> Path:
        tenant_path = self._base_path / tenant.org_id / tenant.workspace_id / job_id
        tenant_path.mkdir(parents=True, exist_ok=True)
        return tenant_path / f"{document_id}.{name}.json"

    def store_json_artifact(
        self,
        tenant: Tenant,
        *,
        job_id: str,
        document_id: str,
        name: str,
        payload: object,
    ) -> ArtifactRecord:
        path = self._json_path(tenant, job_id, document_id, name)
        payload_text = json.dumps(payload, default=_json_default, ensure_ascii=False, indent=2)
        path.write_text(payload_text, encoding="utf-8")
        stored_at = datetime.now(timezone.utc)
        return ArtifactRecord(
            uri=str(path),
            content_type="application/json",
            metadata={"asset": name},
            stored_at=stored_at,
        )

    def load_json_artifact(
        self,
        tenant: Tenant,
        *,
        job_id: str,
        document_id: str,
        name: str,
    ) -> object | None:
        path = self._json_path(tenant, job_id, document_id, name)
        if not path.exists():
            return None
        content = path.read_text(encoding="utf-8")
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None

    def json_artifact_exists(
        self,
        tenant: Tenant,
        *,
        job_id: str,
        document_id: str,
        name: str,
    ) -> bool:
        path = self._json_path(tenant, job_id, document_id, name)
        return path.exists()


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
        key = self._object_key(tenant, job_id, document_id, suffix=self._extension_from_type(content_type))
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

    def _object_key(self, tenant: Tenant, job_id: str, document_id: str, *, suffix: str) -> str:
        return "/".join(
            filter(
                None,
                [self._prefix, tenant.org_id, tenant.workspace_id, job_id, f"{document_id}{suffix}"],
            )
        )

    def store_json_artifact(
        self,
        tenant: Tenant,
        *,
        job_id: str,
        document_id: str,
        name: str,
        payload: object,
    ) -> ArtifactRecord:
        body = json.dumps(payload, default=_json_default).encode("utf-8")
        key = self._object_key(tenant, job_id, document_id, suffix=f".{name}.json")
        stored_at = datetime.now(timezone.utc)
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            Metadata={"stored-at": stored_at.isoformat(), "asset": name},
        )
        return ArtifactRecord(
            uri=f"s3://{self._bucket}/{key}",
            content_type="application/json",
            metadata={"asset": name},
            stored_at=stored_at,
        )

    def load_json_artifact(
        self,
        tenant: Tenant,
        *,
        job_id: str,
        document_id: str,
        name: str,
    ) -> object | None:
        key = self._object_key(tenant, job_id, document_id, suffix=f".{name}.json")
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
        except Exception:  # pragma: no cover - relies on boto3 client
            return None
        body = response.get("Body")
        if body is None:
            return None
        content = body.read().decode("utf-8")
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None

    def json_artifact_exists(
        self,
        tenant: Tenant,
        *,
        job_id: str,
        document_id: str,
        name: str,
    ) -> bool:
        key = self._object_key(tenant, job_id, document_id, suffix=f".{name}.json")
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception:  # pragma: no cover - relies on boto3 client
            return False


def _json_default(value: object):
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


__all__ = [
    "ArtifactRecord",
    "ArtifactStore",
    "InMemoryArtifactStore",
    "LocalArtifactStore",
    "S3ArtifactStore",
]
