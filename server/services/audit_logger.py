"""Audit Logging Service.

Provides centralized structured audit logging to both the console and logs/audit.log.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

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
            "processing_time_ms": int(processing_time * 1000)
            if processing_time is not None
            else None,
            "metadata": metadata or {},
        }
        try:
            _logger.info(json.dumps(event))
        except Exception as e:
            fallback = logging.getLogger(__name__)
            fallback.error("Failed to write audit log: %s", e)
