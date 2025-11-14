from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Iterable, List, Sequence

from ..models.common import Tenant


@dataclass(slots=True)
class KnowledgeBaseRecord:
    """Represents metadata about a logical knowledge base."""

    knowledge_base_id: str
    org_id: str
    workspace_id: str
    user_id: str
    description: str | None
    document_count: int
    keywords: List[str]
    created_at: datetime
    updated_at: datetime
    last_document_title: str | None = None
    last_ingested_at: datetime | None = None


class KnowledgeBaseStore:
    """Persistence layer for knowledge base catalog entries."""

    def __init__(self, *, connection: Any, logger: logging.Logger | None = None) -> None:
        self._connection = connection
        self._logger = logger or logging.getLogger(__name__)
        self._lock = RLock()
        module_name = type(connection).__module__
        self._placeholder = "%s" if "psycopg" in module_name else "?"
        self._ensure_schema()

    # -- public API -------------------------------------------------
    def ensure(
        self,
        tenant: Tenant,
        knowledge_base_id: str,
        *,
        description: str | None = None,
        keywords: Sequence[str] | None = None,
    ) -> KnowledgeBaseRecord:
        """Create the knowledge base if it does not exist, otherwise return the current record."""

        normalized_id = self._normalize_identifier(knowledge_base_id)
        if not normalized_id:
            raise ValueError("knowledge_base_id is required")
        with self._lock:
            row = self._execute(
                """
                SELECT knowledge_base_id, org_id, workspace_id, user_id, description, document_count,
                       keywords, created_at, updated_at, last_document_title, last_ingested_at
                FROM knowledge_bases
                WHERE knowledge_base_id = ?
                """,
                (normalized_id,),
            ).fetchone()
            now_iso = datetime.now(timezone.utc).isoformat()
            if row:
                record = self._row_to_record(row)
                self._assert_tenant_access(record, tenant)
                updates: dict[str, object] = {}
                if description and description.strip() and description != record.description:
                    updates["description"] = description
                if keywords:
                    merged_keywords = self._merge_keywords(record.keywords, keywords)
                    if merged_keywords != record.keywords:
                        updates["keywords"] = json.dumps(merged_keywords)
                if updates:
                    updates["updated_at"] = now_iso
                    assignments = ", ".join(f"{column} = ?" for column in updates)
                    params = list(updates.values()) + [normalized_id]
                    self._execute(f"UPDATE knowledge_bases SET {assignments} WHERE knowledge_base_id = ?", params)
                    self._connection.commit()
                    record = replace(
                        record,
                        description=updates.get("description", record.description),
                        keywords=json.loads(updates.get("keywords", json.dumps(record.keywords))),
                        updated_at=datetime.fromisoformat(now_iso),
                    )
                return record

            keywords_payload = json.dumps(self._normalize_keywords(keywords))
            self._execute(
                """
                INSERT INTO knowledge_bases (
                    knowledge_base_id,
                    org_id,
                    workspace_id,
                    user_id,
                    description,
                    document_count,
                    keywords,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)
                """,
                (
                    normalized_id,
                    tenant.org_id,
                    tenant.workspace_id,
                    tenant.user_id,
                    description,
                    keywords_payload,
                    now_iso,
                    now_iso,
                ),
            )
            self._connection.commit()
            return KnowledgeBaseRecord(
                knowledge_base_id=normalized_id,
                org_id=tenant.org_id,
                workspace_id=tenant.workspace_id,
                user_id=tenant.user_id,
                description=description,
                document_count=0,
                keywords=json.loads(keywords_payload),
                created_at=datetime.fromisoformat(now_iso),
                updated_at=datetime.fromisoformat(now_iso),
            )

    def list(self, tenant: Tenant) -> List[KnowledgeBaseRecord]:
        """Return all knowledge bases owned by the tenant."""

        cursor = self._execute(
            """
            SELECT knowledge_base_id, org_id, workspace_id, user_id, description, document_count,
                   keywords, created_at, updated_at, last_document_title, last_ingested_at
            FROM knowledge_bases
            WHERE org_id = ? AND workspace_id = ?
            ORDER BY updated_at DESC
            """,
            (tenant.org_id, tenant.workspace_id),
        )
        return [self._row_to_record(row) for row in cursor.fetchall()]

    def get(self, tenant: Tenant, knowledge_base_id: str) -> KnowledgeBaseRecord | None:
        normalized_id = self._normalize_identifier(knowledge_base_id)
        if not normalized_id:
            return None
        row = self._execute(
            """
            SELECT knowledge_base_id, org_id, workspace_id, user_id, description, document_count,
                   keywords, created_at, updated_at, last_document_title, last_ingested_at
            FROM knowledge_bases
            WHERE knowledge_base_id = ? AND org_id = ? AND workspace_id = ?
            """,
            (normalized_id, tenant.org_id, tenant.workspace_id),
        ).fetchone()
        if not row:
            return None
        return self._row_to_record(row)

    def record_document_ingested(
        self,
        tenant: Tenant,
        knowledge_base_id: str,
        *,
        document_title: str,
        keywords: Sequence[str] | None = None,
    ) -> None:
        """Increment document counters and merge keywords for the knowledge base."""

        normalized_id = self._normalize_identifier(knowledge_base_id)
        if not normalized_id:
            raise ValueError("knowledge_base_id is required")
        keyword_list = self._normalize_keywords(keywords)
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            row = self._execute(
                """
                SELECT keywords, org_id, workspace_id
                FROM knowledge_bases
                WHERE knowledge_base_id = ?
                """,
                (normalized_id,),
            ).fetchone()
            if row is None:
                # Create the knowledge base on the fly to avoid dropping ingestion results.
                self.ensure(tenant, normalized_id)
                existing_keywords: list[str] = []
            else:
                record_org, record_workspace = row[1], row[2]
                if record_org != tenant.org_id or record_workspace != tenant.workspace_id:
                    raise PermissionError("Tenant does not own the requested knowledge base")
                existing_keywords = self._deserialize_keywords(row[0])
            merged_keywords = self._merge_keywords(existing_keywords, keyword_list)
            self._execute(
                """
                UPDATE knowledge_bases
                SET document_count = COALESCE(document_count, 0) + 1,
                    updated_at = ?,
                    last_document_title = ?,
                    last_ingested_at = ?,
                    keywords = ?
                WHERE knowledge_base_id = ? AND org_id = ? AND workspace_id = ?
                """,
                (
                    now_iso,
                    document_title,
                    now_iso,
                    json.dumps(merged_keywords),
                    normalized_id,
                    tenant.org_id,
                    tenant.workspace_id,
                ),
            )
            self._connection.commit()

    # -- internal helpers -------------------------------------------
    def _ensure_schema(self) -> None:
        with self._lock:
            self._execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    knowledge_base_id TEXT PRIMARY KEY,
                    org_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    description TEXT,
                    document_count INTEGER NOT NULL DEFAULT 0,
                    keywords TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_document_title TEXT,
                    last_ingested_at TEXT
                )
                """
            )
            self._connection.commit()
        self._maybe_add_column("last_document_title TEXT")
        self._maybe_add_column("last_ingested_at TEXT")

    def _maybe_add_column(self, column_def: str) -> None:
        try:
            self._execute(f"ALTER TABLE knowledge_bases ADD COLUMN {column_def}")
            self._connection.commit()
        except Exception:  # pragma: no cover - best effort schema migration
            try:
                self._connection.rollback()
            except Exception:  # pragma: no cover - defensive cleanup
                pass

    def _execute(self, sql: str, parameters: Sequence[object] | None = None):
        cursor = self._connection.cursor()
        statement = self._prepare_sql(sql)
        if parameters is None:
            cursor.execute(statement)
        else:
            cursor.execute(statement, parameters)
        return cursor

    def _prepare_sql(self, sql: str) -> str:
        if self._placeholder == "?":
            return sql
        return sql.replace("?", self._placeholder)

    def _row_to_record(self, row) -> KnowledgeBaseRecord:
        keywords = self._deserialize_keywords(row[6])
        created_at = datetime.fromisoformat(row[7]) if row[7] else datetime.now(timezone.utc)
        updated_at = datetime.fromisoformat(row[8]) if row[8] else created_at
        last_ingested_at = datetime.fromisoformat(row[10]) if row[10] else None
        return KnowledgeBaseRecord(
            knowledge_base_id=row[0],
            org_id=row[1],
            workspace_id=row[2],
            user_id=row[3],
            description=row[4],
            document_count=row[5] or 0,
            keywords=keywords,
            created_at=created_at,
            updated_at=updated_at,
            last_document_title=row[9],
            last_ingested_at=last_ingested_at,
        )

    def _normalize_identifier(self, identifier: str | None) -> str:
        return (identifier or "").strip()

    def _assert_tenant_access(self, record: KnowledgeBaseRecord, tenant: Tenant) -> None:
        if record.org_id != tenant.org_id or record.workspace_id != tenant.workspace_id:
            raise PermissionError("Tenant does not own the requested knowledge base")

    def _deserialize_keywords(self, payload: str | None) -> List[str]:
        if not payload:
            return []
        try:
            values = json.loads(payload)
        except json.JSONDecodeError:
            return []
        if not isinstance(values, list):
            return []
        return [str(value) for value in values if value]

    def _normalize_keywords(self, keywords: Sequence[str] | None) -> List[str]:
        if not keywords:
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for keyword in keywords:
            if not keyword:
                continue
            candidate = str(keyword).strip()
            if not candidate:
                continue
            lowered = candidate.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(candidate)
        return normalized

    def _merge_keywords(self, existing: Sequence[str], new_keywords: Sequence[str]) -> List[str]:
        merged = list(existing)
        seen = {kw.lower(): kw for kw in merged}
        for keyword in new_keywords:
            lowered = keyword.lower()
            if lowered in seen:
                continue
            merged.append(keyword)
            seen[lowered] = keyword
            if len(merged) >= 50:
                break
        return merged


__all__ = ["KnowledgeBaseRecord", "KnowledgeBaseStore"]

