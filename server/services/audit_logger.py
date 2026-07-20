"""Audit Logging Service.

Provides centralized structured audit logging to both the console and logs/audit.log.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from contextlib import contextmanager
import functools
import time
import inspect

# Base directory is workspace root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_LOG_PATH = LOGS_DIR / "audit.log"

AUDIT_LOGGER_NAME = "audit_logger"


def setup_audit_logger() -> logging.Logger:
    """Configures the audit logger with console and file handlers."""
    logger = logging.getLogger(AUDIT_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(message)s")

        # File Handler
        file_handler = logging.FileHandler(AUDIT_LOG_PATH, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger


_logger = setup_audit_logger()


class AuditLogger:
    """Service to record structured audit events across the platform."""

    @staticmethod
    def log(
        action: str,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        document_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        http_method: Optional[str] = None,
        status: Optional[str] = None,
        processing_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write structured audit event as JSON to stdout and logs/audit.log."""
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "action": action,
            "user_id": user_id,
            "workspace_id": workspace_id,
            "chat_id": chat_id,
            "document_id": document_id,
            "endpoint": endpoint,
            "http_method": http_method,
            "status": status,
            "processing_time_ms": (
                int(processing_time * 1000) if processing_time is not None else None
            ),
            "metadata": metadata or {},
        }
        try:
            _logger.info(json.dumps(event))
        except Exception as e:
            fallback = logging.getLogger(__name__)
            fallback.error("Failed to write audit log: %s", e)


@contextmanager
def audit_context(
    action: str,
    user_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    document_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    http_method: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Context manager to dynamically log audit events and track processing time.
    Automatically captures exceptions, updates status to failed, and calculates latency.
    """
    start_time = time.time()
    status = "success"
    err = None
    meta = dict(metadata or {})
    try:
        yield meta
    except Exception as exc:
        status = "failed"
        err = str(exc)
        meta["error"] = err
        raise exc
    finally:
        status = meta.pop("status", status)
        elapsed = time.time() - start_time
        AuditLogger.log(
            action=action,
            user_id=user_id,
            workspace_id=workspace_id,
            chat_id=chat_id,
            document_id=document_id,
            endpoint=endpoint,
            http_method=http_method,
            status=status,
            processing_time=elapsed,
            metadata=meta,
        )


def audited(action: str, extract_args: Optional[Dict[str, str]] = None, **static_meta):
    """Decorator to automatically audit functions. Supports both async and sync execution,
    dynamically extracting arguments by binding function signature parameters.
    """

    def decorator(func):
        sig = inspect.signature(func)
        is_async = inspect.iscoroutinefunction(func)

        def _get_audit_details(*args, **kwargs):
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            resolved = {}
            if extract_args:
                for audit_field, param_name in extract_args.items():
                    if param_name in bound.arguments:
                        resolved[audit_field] = bound.arguments[param_name]
            meta = {**static_meta}
            for param in ["ticker", "query", "limit", "filename"]:
                if param in bound.arguments:
                    val = bound.arguments[param]
                    if not isinstance(val, (bytes, bytearray)):
                        meta[param] = val
            return resolved, meta

        if is_async:

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                resolved, meta = _get_audit_details(*args, **kwargs)
                start_time = time.time()
                status = "success"
                err = None
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    status = "failed"
                    err = str(exc)
                    meta["error"] = err
                    raise exc
                finally:
                    elapsed = time.time() - start_time
                    AuditLogger.log(
                        action=action,
                        user_id=resolved.get("user_id"),
                        workspace_id=resolved.get("workspace_id"),
                        chat_id=resolved.get("chat_id"),
                        document_id=resolved.get("document_id"),
                        status=status,
                        processing_time=elapsed,
                        metadata=meta,
                    )

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                resolved, meta = _get_audit_details(*args, **kwargs)
                start_time = time.time()
                status = "success"
                err = None
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    status = "failed"
                    err = str(exc)
                    meta["error"] = err
                    raise exc
                finally:
                    elapsed = time.time() - start_time
                    AuditLogger.log(
                        action=action,
                        user_id=resolved.get("user_id"),
                        workspace_id=resolved.get("workspace_id"),
                        chat_id=resolved.get("chat_id"),
                        document_id=resolved.get("document_id"),
                        status=status,
                        processing_time=elapsed,
                        metadata=meta,
                    )

            return sync_wrapper

    return decorator
