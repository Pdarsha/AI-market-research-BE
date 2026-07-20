"""Prompt Template Registry Service.

Discovers, validates, sorts, and caches prompt templates from the server/prompt_templates/ directory.
Ensures file paths are not exposed to clients.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Base templates directory: server/prompt_templates/
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "prompt_templates"

REQUIRED_FIELDS = [
    "id",
    "title",
    "description",
    "tag",
    "icon",
    "prompt",
    "enabled",
    "display_order",
]


class PromptTemplate:
    """Represents a discovered and validated prompt template."""

    def __init__(self, id: str, manifest: Dict[str, Any], template_dir: Path):
        self.id = id
        self.manifest = manifest
        self._template_dir = template_dir

    def get_metadata(self) -> Dict[str, Any]:
        """Return only metadata required by the frontend."""
        return {
            "id": self.manifest.get("id"),
            "title": self.manifest.get("title"),
            "description": self.manifest.get("description"),
            "tag": self.manifest.get("tag"),
            "icon": self.manifest.get("icon"),
            "prompt": self.manifest.get("prompt"),
        }

    def read_file(self, filename: str) -> Optional[str]:
        """Read an associated helper file from the template directory (e.g., prompt.md).

        This prevents directory traversal and does not expose filesystem paths.
        """
        file_path = (self._template_dir / filename).resolve()
        if not file_path.is_relative_to(self._template_dir.resolve()):
            logger.warning(
                "Attempted directory traversal access to %s in template %s",
                filename,
                self.id,
            )
            return None

        if not file_path.is_file():
            return None

        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(
                "Failed to read file %s from template %s: %s",
                filename,
                self.id,
                e,
            )
            return None


class PromptTemplateRegistry:
    """Registry to scan, validate, and manage prompt templates."""

    def __init__(self, templates_dir: Path = TEMPLATES_DIR):
        self.templates_dir = templates_dir
        self._cache: Dict[str, PromptTemplate] = {}
        self.reload()

    def reload(self) -> None:
        """Scan the templates directory, validate manifests, and cache active ones."""
        new_cache = {}
        if not self.templates_dir.exists() or not self.templates_dir.is_dir():
            logger.warning("Prompt templates directory not found: %s", self.templates_dir)
            self._cache = {}
            return

        for path in self.templates_dir.iterdir():
            if not path.is_dir():
                continue

            manifest_path = path / "manifest.json"
            if not manifest_path.exists():
                logger.warning(
                    "Skipping template directory %s: manifest.json is missing",
                    path.name,
                )
                continue

            try:
                content = manifest_path.read_text(encoding="utf-8")
                manifest = json.loads(content)
            except json.JSONDecodeError as e:
                logger.warning(
                    "Skipping template directory %s: invalid JSON in manifest.json: %s",
                    path.name,
                    e,
                )
                continue
            except Exception as e:
                logger.warning(
                    "Skipping template directory %s: failed to read manifest.json: %s",
                    path.name,
                    e,
                )
                continue

            # Validate required fields
            missing_fields = [f for f in REQUIRED_FIELDS if f not in manifest]
            if missing_fields:
                logger.warning(
                    "Skipping template directory %s: manifest.json is missing required fields: %s",
                    path.name,
                    ", ".join(missing_fields),
                )
                continue

            # Check if enabled
            if not manifest.get("enabled"):
                logger.debug("Template %s is disabled. Skipping.", path.name)
                continue

            template_id = manifest["id"]
            if template_id != path.name:
                logger.warning(
                    "Template folder name '%s' does not match manifest id '%s'",
                    path.name,
                    template_id,
                )

            new_cache[template_id] = PromptTemplate(
                id=template_id, manifest=manifest, template_dir=path
            )

        self._cache = new_cache

    def list_templates(self) -> List[PromptTemplate]:
        """Return list of active templates sorted by display_order."""
        return sorted(
            self._cache.values(),
            key=lambda t: t.manifest.get("display_order", 9999),
        )

    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Retrieve a cached template by its ID."""
        return self._cache.get(template_id)


@lru_cache(maxsize=1)
def get_prompt_template_registry() -> PromptTemplateRegistry:
    """Return the shared PromptTemplateRegistry singleton instance."""
    return PromptTemplateRegistry()
