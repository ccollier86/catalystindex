from catalystindex.models.common import Tenant
from catalystindex.services.feedback import FeedbackService
from catalystindex.storage.term_index import InMemoryTermIndex
from catalystindex.telemetry.logger import AuditLogger, MetricsRecorder


def test_feedback_submission_updates_metrics_and_audit():
    term_index = InMemoryTermIndex()
    metrics = MetricsRecorder()
    service = FeedbackService(
        term_index=term_index,
        metrics=metrics,
        audit_logger=AuditLogger(),
    )
    tenant = Tenant(org_id="org", workspace_id="ws", user_id="user")

    record = service.submit(
        tenant,
        query="ptsd treatment",
        chunk_ids=["doc:chunk-1"],
        positive=True,
        comment="great match",
        metadata={"source": "unit-test"},
    )

    assert record.positive is True
    assert "doc:chunk-1" in record.chunk_ids
    assert metrics.feedback_positive == 1
    assert metrics.feedback_positive_ratio == 1.0
    analytics = service.analytics(tenant)
    assert analytics.total_positive == 1
    assert analytics.total_negative == 0
    assert analytics.chunks[0].identifier == "doc:chunk-1"


def test_feedback_submission_requires_chunk_ids():
    service = FeedbackService(
        term_index=InMemoryTermIndex(),
        metrics=MetricsRecorder(),
        audit_logger=AuditLogger(),
    )
    tenant = Tenant(org_id="org", workspace_id="ws", user_id="user")

    try:
        service.submit(tenant, query="query", chunk_ids=[], positive=False)
    except ValueError as exc:
        assert "at least one chunk" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError for missing chunk ids")


def test_feedback_analytics_tracks_positive_and_negative_counts():
    service = FeedbackService(
        term_index=InMemoryTermIndex(),
        metrics=MetricsRecorder(),
        audit_logger=AuditLogger(),
    )
    tenant = Tenant(org_id="org", workspace_id="ws", user_id="user")
    service.submit(
        tenant,
        query="trauma recovery",
        chunk_ids=["doc:1", "doc:2"],
        positive=True,
        comment="helpful",
    )
    service.submit(
        tenant,
        query="trauma recovery",
        chunk_ids=["doc:1"],
        positive=False,
        comment="not relevant",
    )
    snapshot = service.analytics(tenant)
    assert snapshot.total_positive == 2
    assert snapshot.total_negative == 1
    chunk_stats = {item.identifier: item for item in snapshot.chunks}
    assert chunk_stats["doc:1"].positive == 1
    assert chunk_stats["doc:1"].negative == 1
    assert chunk_stats["doc:2"].positive == 1
