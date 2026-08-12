"""Skill Suggestion Registry Service.

Discovers, validates, sorts, and caches skill generation prompt suggestions
from the server/skill_suggestions/ directory.
Follows the same architectural pattern as PromptTemplateRegistry.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Base suggestions directory: server/skill_suggestions/
SUGGESTIONS_DIR = Path(__file__).resolve().parent.parent / "skill_suggestions"

REQUIRED_FIELDS = [
    "id",
    "title",
    "prompt",
    "enabled",
    "display_order",
]


class SkillSuggestion:
    """Represents a discovered and validated skill suggestion."""

    def __init__(self, id: str, manifest: Dict[str, Any], suggestion_dir: Path):
        self.id = id
        self.manifest = manifest
        self._suggestion_dir = suggestion_dir

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata required by the frontend."""
        chip_text = self.manifest.get("chip_text") or self.manifest.get("prompt")
        return {
            "id": self.manifest.get("id"),
            "title": self.manifest.get("title"),
            "prompt": self.manifest.get("prompt"),
            "chip_text": chip_text,
            "chipText": chip_text,
            "description": self.manifest.get("description", ""),
            "category": self.manifest.get("category", "general"),
            "enabled": bool(self.manifest.get("enabled", True)),
            "display_order": self.manifest.get("display_order", 9999),
        }


class SkillSuggestionRegistry:
    """Registry to scan, validate, and manage skill suggestion prompt templates."""

    def __init__(self, suggestions_dir: Path = SUGGESTIONS_DIR):
        self.suggestions_dir = suggestions_dir
        self._cache: Dict[str, SkillSuggestion] = {}

    def list_suggestions(self, include_disabled: bool = False) -> List[SkillSuggestion]:
        """Return list of suggestions sorted by display_order."""
        suggestions = list(self._cache.values())
        if not include_disabled:
            suggestions = [s for s in suggestions if s.manifest.get("enabled")]
        return sorted(
            suggestions,
            key=lambda s: s.manifest.get("display_order", 9999),
        )


@lru_cache(maxsize=1)
def get_skill_suggestion_registry() -> SkillSuggestionRegistry:
    """Return the shared SkillSuggestionRegistry singleton instance."""
    return SkillSuggestionRegistry()
