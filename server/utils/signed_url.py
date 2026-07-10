"""Signed, expiring URLs for secure document access.

Local files live under ``static/`` (publicly mounted), so the citation viewer
must not link to them directly. Instead it links to an authenticated file
endpoint guarded by a short-lived HMAC token that binds a specific document to a
specific user and an expiry — no server-side state required to validate it.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from typing import Optional

from server.utils.auth import SECRET_KEY


def _sign(payload: str) -> str:
    return hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()


def create_document_token(
    document_id: str, user_id: str, expires_in: int = 3600
) -> str:
    """Return an opaque token granting access to ``document_id`` for ``user_id``."""
    expires_at = int(time.time()) + int(expires_in)
    payload = f"{document_id}:{user_id}:{expires_at}"
    token = f"{payload}:{_sign(payload)}"
    return base64.urlsafe_b64encode(token.encode()).decode()


def verify_document_token(token: str) -> Optional[dict]:
    """Validate a token; return its claims or ``None`` if invalid/expired."""
    try:
        raw = base64.urlsafe_b64decode(token.encode()).decode()
        document_id, user_id, expires_at, signature = raw.rsplit(":", 3)
    except Exception:
        return None

    payload = f"{document_id}:{user_id}:{expires_at}"
    if not hmac.compare_digest(signature, _sign(payload)):
        return None
    if int(expires_at) < int(time.time()):
        return None
    return {
        "document_id": document_id,
        "user_id": user_id,
        "expires_at": int(expires_at),
    }
