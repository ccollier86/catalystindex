from __future__ import annotations

import base64
import hashlib
import hmac
import json

from fastapi.testclient import TestClient

from catalystindex.config.settings import get_settings
from catalystindex.main import create_app


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _build_token(scopes: list[str]) -> str:
    settings = get_settings()
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "org_id": "org",
        "workspace_id": "ws",
        "user_id": "tester",
        "scopes": scopes,
    }
    header_b64 = _b64url(json.dumps(header).encode("utf-8"))
    payload_b64 = _b64url(json.dumps(payload).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = hmac.new(settings.security.jwt_secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    signature_b64 = _b64url(signature)
    return f"{header_b64}.{payload_b64}.{signature_b64}"


def _auth_headers(*scopes: str) -> dict[str, str]:
    token = _build_token(list(scopes))
    return {"Authorization": f"Bearer {token}"}


def test_ingest_and_search_round_trip():
    app = create_app()
    client = TestClient(app)
    headers = _auth_headers("ingest:write", "ingest:read", "search:read")
    ingest_payload = {
        "document_id": "demo-doc",
        "document_title": "Demo Criteria",
        "content": "Criterion A. Exposure to trauma Criterion B. Intrusion",
        "schema": "dsm5",
        "parser_hint": "plain_text",
    }
    ingest_response = client.post("/ingest/document", json=ingest_payload, headers=headers)
    assert ingest_response.status_code == 200, ingest_response.text
    job_payload = ingest_response.json()
    assert job_payload["document"]["status"].lower() in {"succeeded", "completed"}

    status_resp = client.get(f"/ingest/jobs/{job_payload['job_id']}/status", headers=headers)
    if status_resp.status_code == 200:
        status_data = status_resp.json()
        assert status_data["documents_completed"] >= 1

    search_payload = {"query": "ptsd trauma", "mode": "economy", "debug": True}
    search_resp = client.post("/search/query", json=search_payload, headers=headers)
    assert search_resp.status_code == 200, search_resp.text
    search_data = search_resp.json()
    assert search_data["results"], "expected at least one search result after ingestion"
