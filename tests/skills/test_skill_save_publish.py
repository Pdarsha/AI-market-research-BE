import pytest
from unittest.mock import patch, MagicMock

from server.crud.crud_skill import CRUDSkill
from server.models.skill import Skill, SkillStatus
from server.schemas.skill import (
    SkillDraftResponse,
    SkillSaveRequest,
    SkillPublishRequest,
    SkillResponse,
    SkillCreate,
)


class TestSkillSavePublishSchemas:
    """Tests for skill save/publish schemas."""
    
    def test_skill_save_request_valid(self):
        """Valid save request should parse."""
        draft = SkillDraftResponse(
            name="Test Skill",
            description="Description of the skill",
            trigger_description="When user uploads a document",
            instructions="1. Parse\n2. Extract\n3. Summarize"
        )
        request = SkillSaveRequest(draft=draft, status="draft")
        assert request.draft.name == "Test Skill"
        assert request.status == "draft"
    
    def test_skill_save_request_default_status(self):
        """Save request should default to draft status."""
        draft = SkillDraftResponse(
            name="Test Skill",
            description="Description",
            trigger_description="Trigger",
            instructions="Instructions"
        )
        request = SkillSaveRequest(draft=draft)
        assert request.status == "draft"
    
    def test_skill_save_request_published_status(self):
        """Save request with published status should parse."""
        draft = SkillDraftResponse(
            name="Test Skill",
            description="Description",
            trigger_description="Trigger",
            instructions="Instructions"
        )
        request = SkillSaveRequest(draft=draft, status="published")
        assert request.status == "published"
    
    def test_skill_publish_request(self):
        """Publish request should parse."""
        request = SkillPublishRequest(skill_id="test-skill-id")
        assert request.skill_id == "test-skill-id"
    
    def test_skill_response_draft(self):
        """SkillResponse for draft should parse."""
        response = SkillResponse(
            id="test-id",
            user_id="user-id",
            name="Test Skill",
            description="Description",
            trigger_description="Trigger",
            instructions="Instructions",
            status="draft",
            version="1.0.0",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
        )
        assert response.id == "test-id"
        assert response.status == "draft"
        assert response.published_at is None
    
    def test_skill_response_published(self):
        """SkillResponse for published should parse."""
        response = SkillResponse(
            id="test-id",
            user_id="user-id",
            name="Test Skill",
            description="Description",
            trigger_description="Trigger",
            instructions="Instructions",
            status="published",
            version="2.0.0",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
            published_at="2024-01-01T00:00:00",
        )
        assert response.status == "published"
        assert response.published_at is not None


class TestSkillCRUDSavePublish:
    """Tests for CRUD save/publish methods."""
    
    @pytest.fixture
    def mock_db(self):
        return MagicMock()
    
    @pytest.fixture
    def crud_skill(self):
        return CRUDSkill(Skill)
    
    @pytest.fixture
    def sample_draft(self):
        return {
            "name": "Analyze Financial Report",
            "description": "Extracts key financial metrics from reports",
            "trigger_description": "When user uploads a financial report PDF",
            "instructions": "1. Parse document\n2. Extract metrics\n3. Generate summary"
        }
    
    @pytest.fixture
    def sample_skill_create(self, sample_draft):
        return SkillCreate(**sample_draft)
    
    def test_create_skill_success(self, crud_skill, mock_db, sample_skill_create):
        """Creating a valid skill should create skill record."""
        # Mock query for duplicate check
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        # Track added objects
        added_objects = []
        def track_add(obj):
            added_objects.append(obj)
        mock_db.add.side_effect = track_add
        
        # Test the actual method
        skill = crud_skill.create_skill(
            db=mock_db,
            user_id="user-123",
            obj_in=sample_skill_create,
            status="draft"
        )
        
        assert skill is not None
        assert skill.user_id == "user-123"
        assert skill.name == sample_skill_create.name
        assert skill.status == "draft"
        assert skill.version == "1.0.0"
        mock_db.add.assert_called()
        mock_db.commit.assert_called()
        mock_db.refresh.assert_called()
    
    def test_create_skill_duplicate_name(self, crud_skill, mock_db, sample_skill_create):
        """Creating with duplicate name should raise ValueError."""
        # Mock existing skill with same name
        existing_skill = MagicMock()
        existing_skill.id = "existing-id"
        existing_skill.name = sample_skill_create.name
        mock_db.query.return_value.filter.return_value.first.return_value = existing_skill
        
        with pytest.raises(ValueError) as exc_info:
            crud_skill.create_skill(
                mock_db, user_id="user-123", obj_in=sample_skill_create
            )
        
        assert "already exists" in str(exc_info.value)
    
    def test_publish_skill_success(self, crud_skill, mock_db):
        """Publishing a draft skill should succeed."""
        skill = MagicMock()
        skill.id = "skill-123"
        skill.user_id = "user-123"
        skill.status = SkillStatus.DRAFT.value
        skill.version = "1.0.0"
        skill.published_at = None
        
        mock_db.query.return_value.filter.return_value.first.return_value = skill
        
        result = crud_skill.publish_skill(mock_db, skill_id="skill-123", user_id="user-123")
        
        assert result == skill
        assert skill.status == SkillStatus.PUBLISHED.value
        assert skill.published_at is not None
        assert skill.version == "2.0.0"
        mock_db.commit.assert_called()
    
    def test_publish_skill_not_found(self, crud_skill, mock_db):
        """Publishing non-existent skill should return None."""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        result = crud_skill.publish_skill(mock_db, skill_id="nonexistent", user_id="user-123")
        
        assert result is None
    
    def test_publish_skill_wrong_owner(self, crud_skill, mock_db):
        """Publishing skill owned by another user should return None."""
        skill = MagicMock()
        skill.id = "skill-123"
        skill.user_id = "other-user"
        skill.status = SkillStatus.DRAFT.value
        
        mock_db.query.return_value.filter.return_value.first.return_value = None  # Filter by user_id fails
        
        result = crud_skill.publish_skill(mock_db, skill_id="skill-123", user_id="user-123")
        
        assert result is None
    
    def test_publish_published_skill_fails(self, crud_skill, mock_db):
        """Publishing already published skill should raise ValueError."""
        skill = MagicMock()
        skill.id = "skill-123"
        skill.user_id = "user-123"
        skill.status = SkillStatus.PUBLISHED.value
        skill.version = "2.0.0"
        
        mock_db.query.return_value.filter.return_value.first.return_value = skill
        
        with pytest.raises(ValueError) as exc_info:
            crud_skill.publish_skill(mock_db, skill_id="skill-123", user_id="user-123")
        
        assert "Cannot publish skill with status 'published'" in str(exc_info.value)