import base64
import json
import hmac
import hashlib

from fastapi.testclient import TestClient

from catalystindex.config.settings import get_settings
from catalystindex.main import create_app


def build_token(scopes):
    settings = get_settings()
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "org_id": "org",
        "workspace_id": "ws",
        "user_id": "user",
        "scopes": scopes,
    }
    header_segment = _b64(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_segment = _b64(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_segment}.{payload_segment}".encode("utf-8")
    signature = hmac.new(settings.security.jwt_secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    signature_segment = _b64(signature)
    return f"{header_segment}.{payload_segment}.{signature_segment}"


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def test_ingest_and_search_flow():
    app = create_app()
    client = TestClient(app)

    token = build_token(["ingest:write", "search:read", "generate:write"])
    headers = {"Authorization": f"Bearer {token}"}

    ingest_response = client.post(
        "/ingest/document",
        json={
            "document_id": "doc-1",
            "document_title": "PTSD Criteria",
            "content": "Criterion A. Exposure to trauma Criterion B. Intrusion",
        },
        headers=headers,
    )
    assert ingest_response.status_code == 200
    data = ingest_response.json()
    assert data["status"] == "completed"
    assert data["document"]["chunk_count"] > 0
    assert data["document"]["chunks"]

    search_response = client.post(
        "/search/query",
        json={"query": "trauma exposure", "debug": True},
        headers=headers,
    )
    assert search_response.status_code == 200
    search_payload = search_response.json()
    assert search_payload["results"]
    assert search_payload["mode"] in {"economy", "premium"}
    first_result = search_payload["results"][0]
    assert "metadata" in first_result
    if search_payload.get("debug"):
        assert "expanded_query" in search_payload["debug"]

    generation_response = client.post(
        "/generate/summary",
        json={"query": "summarize ptsd"},
        headers=headers,
    )
    assert generation_response.status_code == 200
    summary = generation_response.json()
    assert summary["chunk_count"] <= 6
