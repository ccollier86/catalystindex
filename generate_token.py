#!/usr/bin/env python3
"""Generate a valid JWT token using the actual secret from .env"""

import base64
import hashlib
import hmac
import json

# Secret from .env file
SECRET = "rfWQFIgFkJc5zefMRRaMkNN2QyKhU6Pp0zNb3amLoc00YbniRvgyFFCCmGl0c7VLMqZk8i07oKDvz9GLf4iPLg=="

def b64url_encode(data: bytes) -> str:
    """Base64 URL encode without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')

# Create header and payload
header = {"alg": "HS256", "typ": "JWT"}
payload = {
    "org_id": "test-org",
    "workspace_id": "test-workspace",
    "user_id": "test-user",
    "scopes": ["*"]
}

# Encode header and payload
header_b64 = b64url_encode(json.dumps(header).encode('utf-8'))
payload_b64 = b64url_encode(json.dumps(payload).encode('utf-8'))

# Create signature
signing_input = f"{header_b64}.{payload_b64}".encode('utf-8')
signature = hmac.new(SECRET.encode('utf-8'), signing_input, hashlib.sha256).digest()
signature_b64 = b64url_encode(signature)

# Create final token
token = f"{header_b64}.{payload_b64}.{signature_b64}"
print(token)
