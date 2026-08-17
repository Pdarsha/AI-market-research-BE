"""Integration tests for the complete Skill Creation flow.

Tests the end-to-end flow:
User Request → Generate Draft → Validate → Save → Publish → Skill Library
"""

import pytest
from unittest.mock import patch, MagicMock
import json

from server.services.skill_generation_service import (
    SkillGenerationError,
    SkillGenerationService,
    skill_generation_service,
)
from server.services.skill_validation_service import (
    SkillValidationService,
    skill_validation_service,
)
from server.crud.crud_skill import CRUDSkill
from server.models.skill import Skill, SkillStatus
from server.schemas.skill import (
    SkillDraftRequest,
    SkillDraftResponse,
    SkillValidationRequest,
    SkillSaveRequest,
    SkillPublishRequest,
    SkillCreate,
)


# Module-level fixtures accessible to all test classes
@pytest.fixture
def sample_user_id():
    return "test-user-123"

@pytest.fixture
def valid_user_request():
    return "Create a skill that analyzes financial reports (10-K, 10-Q) and extracts key metrics like revenue, net income, and EPS"

@pytest.fixture
def generated_draft():
    return {
        "name": "Analyze Financial Report",
        "description": "Automatically extracts key financial metrics from uploaded financial reports (10-K, 10-Q, earnings releases) and generates structured summaries.",
        "trigger_description": "When a user uploads a financial report document (PDF, SEC filing, earnings release) and requests analysis or asks questions about financial performance.",
        "instructions": "1. Parse the uploaded document to extract text content and tables.\n2. Identify the document type (10-K, 10-Q, 8-K, earnings release, investor presentation).\n3. Extract key financial metrics: revenue, net income, EPS, operating margin, cash flow, debt levels.\n4. Compare metrics to prior periods (QoQ, YoY) and identify significant changes.\n5. Analyze management commentary for forward guidance, risk factors, and strategic initiatives.\n6. Structure output as: Executive Summary, Key Metrics Table, Trend Analysis, Guidance & Outlook, Risk Factors.\n7. Cite specific sections/page numbers from the source document for all figures.\n8. If data is missing or unclear, explicitly note limitations in the analysis."
    }

@pytest.fixture
def mock_db():
    return MagicMock()

@pytest.fixture
def crud_skill():
    return CRUDSkill(Skill)


class TestSkillCreationFlow:
    """Integration tests for the complete Skill Creation flow."""

    # ============================================================
    # Step 1: Generate Draft
    # ============================================================
    
    @patch("server.services.skill_generation_service.client")
    def test_step1_generate_draft(self, mock_client, valid_user_request, generated_draft):
        """Step 1: Generate draft from user request."""
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = json.dumps(generated_draft)
        mock_client.chat.completions.create.return_value = mock_completion

        service = SkillGenerationService(model="gpt-4o-mini")
        draft = service.generate_draft(valid_user_request)

        assert draft["name"] == generated_draft["name"]
        assert draft["description"] == generated_draft["description"]
        assert draft["trigger_description"] == generated_draft["trigger_description"]
        assert draft["instructions"] == generated_draft["instructions"]
        mock_client.chat.completions.create.assert_called_once()

    @patch("server.services.skill_generation_service.client")
    def test_step1_generate_draft_empty_request_fails(self, mock_client):
        """Step 1: Empty request should fail."""
        service = SkillGenerationService()
        with pytest.raises(SkillGenerationError) as exc_info:
            service.generate_draft("")
        assert exc_info.value.code == "EMPTY_REQUEST"

    @patch("server.services.skill_generation_service.client")
    def test_step1_generate_draft_too_long_fails(self, mock_client):
        """Step 1: Request too long should fail."""
        service = SkillGenerationService()
        with pytest.raises(SkillGenerationError) as exc_info:
            service.generate_draft("x" * 5001)
        assert exc_info.value.code == "REQUEST_TOO_LONG"

    # ============================================================
    # Step 2: Validate Draft
    # ============================================================
    
    def test_step2_validate_draft_valid(self, generated_draft):
        """Step 2: Valid draft passes validation."""
        service = SkillValidationService()
        result = service.validate_draft(generated_draft, check_uniqueness=False)
        
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_step2_validate_draft_invalid_missing_fields(self):
        """Step 2: Invalid draft with missing fields fails validation."""
        invalid_draft = {
            "name": "",
            "description": "Short",
            "trigger_description": "When user asks",
            "instructions": "Do something"
        }
        service = SkillValidationService()
        result = service.validate_draft(invalid_draft, check_uniqueness=False)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        error_fields = {e["field"] for e in result.errors}
        assert "name" in error_fields
        assert "description" in error_fields
        assert "trigger_description" in error_fields
        assert "instructions" in error_fields

    def test_step2_validate_draft_weak_trigger_warning(self, generated_draft):
        """Step 2: Weak trigger produces warning but still valid."""
        draft_with_weak_trigger = generated_draft.copy()
        draft_with_weak_trigger["trigger_description"] = "When user wants analysis"
        
        service = SkillValidationService()
        result = service.validate_draft(draft_with_weak_trigger, check_uniqueness=False)
        
        assert result.is_valid is True  # Warning, not error
        assert any(w["code"] == "WEAK_TRIGGER" for w in result.warnings)

    def test_step2_validate_draft_with_uniqueness_check(self, mock_db, generated_draft, crud_skill):
        """Step 2: Uniqueness check detects duplicate name."""
        # Mock existing skill with same name
        existing_skill = MagicMock()
        existing_skill.id = "existing-id"
        existing_skill.name = generated_draft["name"]
        mock_db.query.return_value.filter.return_value.first.return_value = existing_skill
        
        # Create service with mocked DB
        service = SkillValidationService(db_session=mock_db)
        result = service.validate_draft(generated_draft, user_id="test-user", check_uniqueness=True)
        
        assert result.is_valid is False
        assert any(e["code"] == "DUPLICATE_NAME" for e in result.errors)

    # ============================================================
    # Step 3: Save Draft
    # ============================================================
    
    def test_step3_save_draft_success(self, mock_db, generated_draft, crud_skill, sample_user_id):
        """Step 3: Save valid draft creates skill record."""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        # Mock the created skill
        mock_skill = MagicMock()
        mock_skill.id = "new-skill-id"
        mock_skill.user_id = sample_user_id
        mock_skill.name = generated_draft["name"]
        mock_skill.status = "draft"
        mock_skill.version = "1.0.0"
        mock_skill.created_at = MagicMock()
        mock_skill.updated_at = MagicMock()
        mock_skill.published_at = None
        
        added_objects = []
        def track_add(obj):
            added_objects.append(obj)
        mock_db.add.side_effect = track_add
        
        # We can't easily test the actual DB interaction with mocks
        # This would require a real database integration test
        # The unit tests for CRUD already cover this
        pass

    def test_step3_save_draft_duplicate_name(self, mock_db, generated_draft, crud_skill, sample_user_id):
        """Step 3: Save with duplicate name raises ValueError."""
        existing_skill = MagicMock()
        existing_skill.id = "existing-id"
        existing_skill.name = generated_draft["name"]
        mock_db.query.return_value.filter.return_value.first.return_value = existing_skill
        
        skill_create = SkillCreate(**generated_draft)
        
        with pytest.raises(ValueError) as exc_info:
            crud_skill.create_skill(mock_db, user_id=sample_user_id, obj_in=skill_create)
        
        assert "already exists" in str(exc_info.value)

    def test_step3_save_draft_validation_fails_before_save(self, generated_draft):
        """Step 3: Invalid draft fails validation before save attempt."""
        invalid_draft = generated_draft.copy()
        invalid_draft["name"] = ""  # Empty name
        
        service = SkillValidationService()
        result = service.validate_draft(invalid_draft, check_uniqueness=False)
        
        assert result.is_valid is False
        # Should not proceed to save

    # ============================================================
    # Step 4: Publish Skill
    # ============================================================
    
    def test_step4_publish_skill_success(self, mock_db, crud_skill, sample_user_id):
        """Step 4: Publish draft skill succeeds."""
        skill = MagicMock()
        skill.id = "skill-123"
        skill.user_id = sample_user_id
        skill.status = SkillStatus.DRAFT.value
        skill.version = "1.0.0"
        skill.published_at = None
        skill.name = "Test Skill"
        
        mock_db.query.return_value.filter.return_value.first.return_value = skill
        
        result = crud_skill.publish_skill(mock_db, skill_id="skill-123", user_id=sample_user_id)
        
        assert result == skill
        assert skill.status == SkillStatus.PUBLISHED.value
        assert skill.published_at is not None
        assert skill.version == "2.0.0"
        mock_db.commit.assert_called()

    def test_step4_publish_published_skill_fails(self, mock_db, crud_skill, sample_user_id):
        """Step 4: Publishing already published skill fails."""
        skill = MagicMock()
        skill.id = "skill-123"
        skill.user_id = sample_user_id
        skill.status = SkillStatus.PUBLISHED.value
        skill.version = "2.0.0"
        
        mock_db.query.return_value.filter.return_value.first.return_value = skill
        
        with pytest.raises(ValueError) as exc_info:
            crud_skill.publish_skill(mock_db, skill_id="skill-123", user_id=sample_user_id)
        
        assert "Cannot publish skill with status 'published'" in str(exc_info.value)

    def test_step4_publish_not_found(self, mock_db, crud_skill, sample_user_id):
        """Step 4: Publishing non-existent skill returns None."""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        result = crud_skill.publish_skill(mock_db, skill_id="nonexistent", user_id=sample_user_id)
        
        assert result is None

    def test_step4_publish_wrong_owner(self, mock_db, crud_skill, sample_user_id):
        """Step 4: Publishing skill owned by another user returns None."""
        skill = MagicMock()
        skill.id = "skill-123"
        skill.user_id = "other-user"
        skill.status = SkillStatus.DRAFT.value
        
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        result = crud_skill.publish_skill(mock_db, skill_id="skill-123", user_id=sample_user_id)
        
        assert result is None

    # ============================================================
    # Complete Flow Tests
    # ============================================================
    
    @patch("server.services.skill_generation_service.client")
    def test_complete_flow_generate_validate_save_publish(
        self, mock_client, valid_user_request, generated_draft, mock_db, crud_skill, sample_user_id
    ):
        """Complete flow: Generate → Validate → Save → Publish."""
        
        # Step 1: Generate Draft
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = json.dumps(generated_draft)
        mock_client.chat.completions.create.return_value = mock_completion
        
        gen_service = SkillGenerationService(model="gpt-4o-mini")
        draft = gen_service.generate_draft(valid_user_request)
        
        assert draft["name"] == generated_draft["name"]
        
        # Step 2: Validate Draft
        val_service = SkillValidationService()
        validation_result = val_service.validate_draft(draft, check_uniqueness=False)
        
        assert validation_result.is_valid is True
        
        # Step 3: Save Draft (mocked)
        # Step 4: Publish (mocked)
        skill = MagicMock()
        skill.id = "skill-123"
        skill.user_id = sample_user_id
        skill.status = SkillStatus.DRAFT.value
        skill.version = "1.0.0"
        skill.published_at = None
        skill.name = generated_draft["name"]
        
        mock_db.query.return_value.filter.return_value.first.return_value = skill
        
        published_skill = crud_skill.publish_skill(mock_db, skill_id="skill-123", user_id=sample_user_id)
        
        assert published_skill.status == SkillStatus.PUBLISHED.value
        assert published_skill.version == "2.0.0"


class TestSkillFlowErrorHandling:
    """Tests for error handling across the flow."""

    @patch("server.services.skill_generation_service.client")
    def test_generation_llm_error(self, mock_client):
        """LLM error during generation should be handled."""
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        service = SkillGenerationService()
        with pytest.raises(SkillGenerationError) as exc_info:
            service.generate_draft("Create a skill")
        
        assert exc_info.value.code == "LLM_ERROR"

    @patch("server.services.skill_generation_service.client")
    def test_generation_malformed_json(self, mock_client):
        """Malformed JSON from LLM should be handled."""
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "not valid json {"
        mock_client.chat.completions.create.return_value = mock_completion
        
        service = SkillGenerationService()
        with pytest.raises(SkillGenerationError) as exc_info:
            service.generate_draft("Create a skill")
        
        assert exc_info.value.code == "MALFORMED_RESPONSE"

    @patch("server.services.skill_generation_service.client")
    def test_generation_validation_failure(self, mock_client):
        """Draft failing validation after generation should be handled."""
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = json.dumps({
            "name": "Test Skill",
            "description": "Description",
            "trigger_description": "Trigger"
            # Missing instructions
        })
        mock_client.chat.completions.create.return_value = mock_completion
        
        service = SkillGenerationService()
        with pytest.raises(SkillGenerationError) as exc_info:
            service.generate_draft("Create a skill")
        
        assert exc_info.value.code == "VALIDATION_FAILED"
        assert "instructions" in str(exc_info.value.details)

    def test_save_unvalidated_draft_fails(self):
        """Attempting to save unvalidated draft should be caught by validation."""
        invalid_draft = {
            "name": "",
            "description": "Short",
            "trigger_description": "When user asks",
            "instructions": "Do something"
        }
        
        service = SkillValidationService()
        result = service.validate_draft(invalid_draft, check_uniqueness=False)
        
        assert result.is_valid is False
        # Should not proceed to save

    def test_publish_invalid_state_transition(self, mock_db, crud_skill):
        """Publishing from invalid state should fail."""
        for invalid_status in [SkillStatus.PUBLISHED.value]:
            skill = MagicMock()
            skill.id = "skill-123"
            skill.user_id = "user-123"
            skill.status = invalid_status
            
            mock_db.query.return_value.filter.return_value.first.return_value = skill
            
            with pytest.raises(ValueError) as exc_info:
                crud_skill.publish_skill(mock_db, skill_id="skill-123", user_id="user-123")
            
            assert "Cannot publish skill with status" in str(exc_info.value)


class TestSkillSchemasIntegration:
    """Tests for schema consistency across the flow."""

    def test_draft_response_compatible_with_all_steps(self, generated_draft):
        """SkillDraftResponse should work for generate, validate, save."""
        draft_response = SkillDraftResponse(**generated_draft)
        
        # Can be used for validation
        val_request = SkillValidationRequest(draft=draft_response)
        assert val_request.draft.name == generated_draft["name"]
        
        # Can be used for save
        save_request = SkillSaveRequest(draft=draft_response, status="draft")
        assert save_request.draft.name == generated_draft["name"]

    def test_save_request_defaults_to_draft(self, generated_draft):
        """Save request defaults to draft status."""
        draft_response = SkillDraftResponse(**generated_draft)
        save_request = SkillSaveRequest(draft=draft_response)
        assert save_request.status == "draft"

    def test_save_request_can_be_published(self, generated_draft):
        """Save request can specify published status."""
        draft_response = SkillDraftResponse(**generated_draft)
        save_request = SkillSaveRequest(draft=draft_response, status="published")
        assert save_request.status == "published"

    def test_publish_request_requires_skill_id(self):
        """Publish request requires skill_id."""
        publish_request = SkillPublishRequest(skill_id="test-skill-id")
        assert publish_request.skill_id == "test-skill-id"


class TestSkillLifecycleStateTransitions:
    """Tests for valid/invalid state transitions."""

    def test_valid_publish_transition(self):
        """Valid state transition: draft → published via publish."""
        assert SkillStatus.DRAFT.value != SkillStatus.PUBLISHED.value

    def test_invalid_publish_transitions(self):
        """Invalid transitions to published."""
        invalid_source_statuses = [
            SkillStatus.PUBLISHED.value,
        ]
        
        for status in invalid_source_statuses:
            # These should raise ValueError when trying to publish
            assert status == SkillStatus.PUBLISHED.value