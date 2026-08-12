import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from server.main import app
from server.services.skill_suggestion_registry import SkillSuggestionRegistry, SkillSuggestion

client = TestClient(app)


class TestSkillSuggestionsAPI:
    """Tests for GET /v1/skills/suggestions endpoint and SkillSuggestionRegistry."""

    def test_get_skill_suggestions_success(self):
        """GET /v1/skills/suggestions should return HTTP 200 with list of default suggestions."""
        response = client.get("/v1/skills/suggestions")
        assert response.status_code == 200
        
        json_data = response.json()
        assert json_data["success"] is True
        assert json_data["error"] is None
        assert isinstance(json_data["data"], list)
        assert len(json_data["data"]) == 6

        # Check default suggestions
        ids = [item["id"] for item in json_data["data"]]
        assert "ascii-art-diagrams" in ids
        assert "improve-skill-md" in ids
        assert "validation-and-packaging" in ids
        assert "pdf-report-extractor" in ids
        assert "scheduled-morning-brief" in ids
        assert "api-webhook-integrator" in ids

    def test_correct_response_schema(self):
        """Each suggestion item should contain required fields for frontend rendering."""
        response = client.get("/v1/skills/suggestions")
        assert response.status_code == 200

        data = response.json()["data"]
        for item in data:
            assert "id" in item
            assert "title" in item
            assert "prompt" in item
            assert "chip_text" in item
            assert "chipText" in item
            assert "description" in item
            assert "category" in item
            assert "enabled" in item
            assert "display_order" in item

    def test_active_inactive_filtering(self, tmp_path):
        """Suggestions with enabled=false should be excluded by list_suggestions()."""
        registry = SkillSuggestionRegistry(suggestions_dir=tmp_path)
        
        # Create 2 suggestions: one active, one disabled
        active_dir = tmp_path / "active"
        active_dir.mkdir()
        (active_dir / "manifest.json").write_text(
            '{"id": "active-1", "title": "Active", "prompt": "P1", "enabled": true, "display_order": 1}'
        )

        disabled_dir = tmp_path / "disabled"
        disabled_dir.mkdir()
        (disabled_dir / "manifest.json").write_text(
            '{"id": "disabled-1", "title": "Disabled", "prompt": "P2", "enabled": false, "display_order": 2}'
        )

        registry.reload()
        suggestions = registry.list_suggestions()
        assert len(suggestions) == 1
        assert suggestions[0].id == "active-1"

    def test_ordering(self, tmp_path):
        """Suggestions should be ordered by display_order ascending."""
        registry = SkillSuggestionRegistry(suggestions_dir=tmp_path)

        dir2 = tmp_path / "s2"
        dir2.mkdir()
        (dir2 / "manifest.json").write_text(
            '{"id": "s2", "title": "Second", "prompt": "P2", "enabled": true, "display_order": 2}'
        )

        dir1 = tmp_path / "s1"
        dir1.mkdir()
        (dir1 / "manifest.json").write_text(
            '{"id": "s1", "title": "First", "prompt": "P1", "enabled": true, "display_order": 1}'
        )

        registry.reload()
        suggestions = registry.list_suggestions()
        assert len(suggestions) == 2
        assert suggestions[0].id == "s1"
        assert suggestions[1].id == "s2"

    def test_empty_suggestion_handling(self, tmp_path):
        """When suggestions directory is empty or missing, registry returns empty list without error."""
        empty_path = tmp_path / "non_existent_dir"
        registry = SkillSuggestionRegistry(suggestions_dir=empty_path)
        
        suggestions = registry.list_suggestions()
        assert suggestions == []


class TestRichTextHtmlPreview:
    """Tests for rich text / HTML conversion rendering."""

    def test_render_rich_text_formatting_options(self):
        """render_rich_text should render bold, italic, underline, strikethrough, lists, headings, quotes, code, links into clean HTML."""
        from server.utils.markdown import render_rich_text

        sample_markdown = """# Skill Title
**Bold text** and *italic text* and <u>underline text</u> and ~~strikethrough~~

1. Step one
2. Step two

- Bullet point

> Operational note

`code snippet`

[Link Label](https://example.com)
"""

        html = render_rich_text(sample_markdown)

        # Verify structured HTML elements exist
        assert "<h1>Skill Title</h1>" in html
        assert "<strong>Bold text</strong>" in html
        assert "<em>italic text</em>" in html
        assert "<u>underline text</u>" in html
        assert "<del>strikethrough</del>" in html
        assert "<ol>" in html
        assert "<li>Step one</li>" in html
        assert "<ul>" in html
        assert "<li>Bullet point</li>" in html
        assert "<blockquote>" in html
        assert "<code>code snippet</code>" in html
        assert '<a href="https://example.com">Link Label</a>' in html
