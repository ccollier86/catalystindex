from __future__ import annotations

from typing import Dict

from ..models.common import Tenant
from ..services.ingestion_jobs import DocumentSubmission, IngestionCoordinator
from ..api.dependencies import get_ingestion_coordinator


def process_ingestion_document(*, job_id: str, tenant: Dict[str, str], submission: Dict[str, object]) -> Dict[str, object]:
    """Worker entry point for processing a single ingestion document."""

    coordinator = _get_coordinator()
    tenant_model = Tenant(**tenant)
    submission_model = _build_submission(submission)
    record = coordinator.process_document_task(tenant_model, job_id, submission_model)
    document = next(
        (doc for doc in record.documents if doc.document_id == submission_model.document_id),
        None,
    )
    return {
        "job_id": record.job_id,
        "status": record.status.value,
        "document": {
            "document_id": submission_model.document_id,
            "status": document.status.value if document else "unknown",
            "error": document.error if document else None,
        },
    }


def _get_coordinator() -> IngestionCoordinator:
    return get_ingestion_coordinator()


def _build_submission(payload: Dict[str, object]) -> DocumentSubmission:
    return DocumentSubmission(
        document_id=payload["document_id"],
        document_title=payload.get("document_title", payload["document_id"]),
        schema=payload.get("schema"),
        source_type=payload.get("source_type", "inline"),
        parser_hint=payload.get("parser_hint"),
        metadata=dict(payload.get("metadata") or {}),
        content=payload.get("content"),
        content_uri=payload.get("content_uri"),
    )


__all__ = ["process_ingestion_document"]
