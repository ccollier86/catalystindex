from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Iterable, List

from fastapi import HTTPException, status

from ..config.settings import get_settings
from ..models.common import Tenant


class JWTAuthError(HTTPException):
    def __init__(self, detail: str) -> None:
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


def decode_jwt(token: str) -> dict:
    settings = get_settings()
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
    except ValueError as exc:
        raise JWTAuthError("Invalid token format") from exc
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = _b64url_decode(signature_b64)
    expected = hmac.new(settings.security.jwt_secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        raise JWTAuthError("Invalid token signature")
    payload_bytes = _b64url_decode(payload_b64)
    try:
        return json.loads(payload_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise JWTAuthError("Invalid token payload") from exc


def ensure_scopes(claims: dict, required_scopes: Iterable[str]) -> None:
    scopes: List[str] = claims.get("scopes", []) or []
    missing = [scope for scope in required_scopes if scope not in scopes]
    if missing:
        raise JWTAuthError(f"Missing required scopes: {', '.join(missing)}")


def extract_tenant(claims: dict) -> Tenant:
    try:
        return Tenant(
            org_id=str(claims["org_id"]),
            workspace_id=str(claims["workspace_id"]),
            user_id=str(claims["user_id"]),
        )
    except KeyError as exc:
        raise JWTAuthError("Invalid tenant claims") from exc


def _b64url_decode(segment: str) -> bytes:
    padding = '=' * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


__all__ = ["decode_jwt", "ensure_scopes", "extract_tenant", "JWTAuthError"]
