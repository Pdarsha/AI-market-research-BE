import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy.exc import IntegrityError

from server.crud.crud_skill import CRUDSkill, skill as crud_skill
from server.models.skill import Skill
from server.schemas.skill import SkillCreate
from server.services.skill_generation_service import SkillGenerationService


class TestUniqueSkillNameStrategy:
    """Tests for unique skill name generation, pre-save checks, and race condition handling."""

    def test_get_unique_skill_name_first_time(self):
        """When skill name does not exist, base name should be returned."""
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        name = crud_skill.get_unique_skill_name(
            mock_db, user_id="user-123", base_name="Render Morning Brief"
        )
        assert name == "Render Morning Brief"

    def test_get_unique_skill_name_second_time(self):
        """When base name exists, 'Render Morning Brief 2' should be returned."""
        mock_db = MagicMock()
        
        # First query for 'Render Morning Brief' returns an existing skill
        # Second query for 'Render Morning Brief 2' returns None (available)
        existing = MagicMock(id="existing-1", name="Render Morning Brief")
        mock_db.query.return_value.filter.return_value.first.side_effect = [existing, None]

        name = crud_skill.get_unique_skill_name(
            mock_db, user_id="user-123", base_name="Render Morning Brief"
        )
        assert name == "Render Morning Brief 2"

    def test_get_unique_skill_name_third_time(self):
        """When base name and variant 2 exist, 'Render Morning Brief 3' should be returned."""
        mock_db = MagicMock()
        
        existing1 = MagicMock(id="existing-1", name="Render Morning Brief")
        existing2 = MagicMock(id="existing-2", name="Render Morning Brief 2")
        mock_db.query.return_value.filter.return_value.first.side_effect = [existing1, existing2, None]

        name = crud_skill.get_unique_skill_name(
            mock_db, user_id="user-123", base_name="Render Morning Brief"
        )
        assert name == "Render Morning Brief 3"

    def test_get_unique_skill_name_base_with_number(self):
        """When base name is 'Render Morning Brief 2' and it exists, 'Render Morning Brief 3' should be returned."""
        mock_db = MagicMock()
        
        existing = MagicMock(id="existing-2", name="Render Morning Brief 2")
        mock_db.query.return_value.filter.return_value.first.side_effect = [existing, None]

        name = crud_skill.get_unique_skill_name(
            mock_db, user_id="user-123", base_name="Render Morning Brief 2"
        )
        assert name == "Render Morning Brief 3"

    def test_create_skill_auto_generates_unique_name(self):
        """Creating a skill when base name exists automatically creates a unique variant."""
        mock_db = MagicMock()
        
        existing = MagicMock(id="existing-1", name="Render Morning Brief")
        mock_db.query.return_value.filter.return_value.first.side_effect = [existing, None]

        added_objects = []
        mock_db.add.side_effect = lambda obj: added_objects.append(obj)

        skill_in = SkillCreate(
            name="Render Morning Brief",
            description="Description",
            trigger_description="Trigger",
            instructions="Instructions"
        )

        saved_skill = crud_skill.create_skill(mock_db, user_id="user-123", obj_in=skill_in)
        assert saved_skill.name == "Render Morning Brief 2"
        mock_db.commit.assert_called()

    def test_create_skill_race_condition_retry(self):
        """When race condition triggers IntegrityError on save, retry generates next variant."""
        mock_db = MagicMock()
        
        # Attempt 0: 'Render Morning Brief' -> None (free)
        # Attempt 1: 'Render Morning Brief' -> existing (taken by concurrent process), 'Render Morning Brief 2' -> None (free)
        existing = MagicMock(id="race-win", name="Render Morning Brief")
        mock_db.query.return_value.filter.return_value.first.side_effect = [None, existing, None]

        # First commit throws IntegrityError (race condition), second commit succeeds
        mock_db.commit.side_effect = [IntegrityError("duplicate key", params={}, orig=Exception("duplicate")), None]

        skill_in = SkillCreate(
            name="Render Morning Brief",
            description="Description",
            trigger_description="Trigger",
            instructions="Instructions"
        )

        saved_skill = crud_skill.create_skill(mock_db, user_id="user-123", obj_in=skill_in)
        
        assert mock_db.rollback.call_count == 1
        assert saved_skill.name == "Render Morning Brief 2"

    @patch("server.services.skill_generation_service.client")
    def test_generate_draft_with_db_returns_unique_name(self, mock_client):
        """Skill generation service checks DB and returns unique name variant."""
        import json
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = json.dumps({
            "name": "Render Morning Brief",
            "description": "Extracts key information for morning briefs",
            "trigger_description": "When user requests morning brief",
            "instructions": "1. Fetch news\n2. Summarize key points\n3. Deliver brief"
        })
        mock_client.chat.completions.create.return_value = mock_completion

        mock_db = MagicMock()
        existing = MagicMock(id="existing-1", name="Render Morning Brief")
        mock_db.query.return_value.filter.return_value.first.side_effect = [existing, None]

        service = SkillGenerationService()
        draft = service.generate_draft("Create morning brief skill", user_id="user-123", db=mock_db)

        assert draft["name"] == "Render Morning Brief 2"

    def test_validate_existing_skill_same_id_and_name(self):
        """Validating an existing skill draft with the same ID and name should pass validation."""
        from server.services.skill_validation_service import skill_validation_service
        
        mock_db = MagicMock()
        # Querying excluding current skill ID returns None (no collision with DIFFERENT skill)
        mock_db.query.return_value.filter.return_value.filter.return_value.first.return_value = None

        draft = {
            "id": "4941c381-f1c7-45de-aabc-7186aed09579",
            "name": "Render Morning Brief 2",
            "description": "Extracts key information for morning briefs",
            "trigger_description": "When user uploads financial report PDF",
            "instructions": "1. Fetch news\n2. Summarize key points\n3. Deliver brief"
        }

        with patch.object(skill_validation_service, "_get_db", return_value=mock_db):
            result = skill_validation_service.validate_draft(draft, user_id="user-123", check_uniqueness=True)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_different_skill_same_name_fails(self):
        """Validating a skill draft whose name belongs to another skill ID should fail validation."""
        from server.services.skill_validation_service import skill_validation_service
        
        mock_db = MagicMock()
        another_skill = MagicMock(id="another-uuid-bbb", name="Render Morning Brief 2")
        mock_db.query.return_value.filter.return_value.filter.return_value.first.return_value = another_skill

        draft = {
            "id": "4941c381-f1c7-45de-aabc-7186aed09579",
            "name": "Render Morning Brief 2",
            "description": "Extracts key information for morning briefs",
            "trigger_description": "When user uploads financial report PDF",
            "instructions": "1. Fetch news\n2. Summarize key points\n3. Deliver brief"
        }

        with patch.object(skill_validation_service, "_get_db", return_value=mock_db):
            result = skill_validation_service.validate_draft(draft, user_id="user-123", check_uniqueness=True)

        assert result.is_valid is False
        assert any(e["code"] == "DUPLICATE_NAME" for e in result.errors)
        assert result.errors[0]["details"]["existing_skill_id"] == "another-uuid-bbb"
