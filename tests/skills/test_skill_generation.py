import pytest
import json
from unittest.mock import patch, MagicMock

from server.prompts.skill_generation import (
    skill_generation_system_prompt,
    skill_generation_prompt,
    skill_edit_system_prompt,
    skill_edit_prompt,
)
from server.services.skill_generation_service import (
    SkillGenerationError,
    SkillGenerationService,
)
from server.schemas.skill import (
    SkillDraftRequest,
    SkillDraftResponse,
    SkillDraftError,
)
from server.services.skill_validation_service import skill_validation_service


class TestSkillGenerationPrompt:
    """Tests for skill generation prompt templates and validation."""
    
    def test_system_prompt_exists(self):
        """System prompt should be a non-empty string."""
        prompt = skill_generation_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert "JSON" in prompt
        assert "name" in prompt
        assert "description" in prompt
        assert "trigger_description" in prompt
        assert "instructions" in prompt
    
    def test_user_prompt_includes_request(self):
        """User prompt should include the user's request."""
        request = "Create a skill to analyze financial reports"
        prompt = skill_generation_prompt(request)
        assert request in prompt
    
    def test_edit_system_prompt_exists(self):
        """Edit system prompt should be a non-empty string."""
        prompt = skill_edit_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert "JSON" in prompt
        assert "COMPLETE REGENERATION" in prompt
        assert "PARTIAL CHANGE" in prompt
    
    def test_edit_prompt_includes_draft_and_feedback(self):
        """Edit prompt should include current draft and user feedback."""
        current_draft = {
            "name": "Test Skill",
            "description": "Description",
            "trigger_description": "Trigger",
            "instructions": "Instructions"
        }
        feedback = "Change the trigger to daily"
        prompt = skill_edit_prompt(current_draft, feedback, regenerate_all=False)
        assert "Test Skill" in prompt
        assert "Change the trigger to daily" in prompt
        assert "PARTIAL CHANGE" in prompt or "specific field" in prompt
    
    def test_edit_prompt_regenerate_all(self):
        """Edit prompt with regenerate_all should indicate complete regeneration."""
        current_draft = {
            "name": "Test Skill",
            "description": "Description",
            "trigger_description": "Trigger",
            "instructions": "Instructions"
        }
        feedback = "Make it for SEC filings instead"
        prompt = skill_edit_prompt(current_draft, feedback, regenerate_all=True)
        assert "REGENERATE" in prompt
        assert "from scratch" in prompt
    
    def test_validate_skill_draft_valid(self):
        """Valid draft should pass validation."""
        draft = {
            "name": "Analyze Financial Report",
            "description": "Extracts key metrics from financial reports",
            "trigger_description": "When user uploads a financial report",
            "instructions": "1. Parse document\n2. Extract metrics\n3. Generate summary"
        }
        is_valid, errors = skill_validation_service.validate_skill_draft(draft)
        assert is_valid is True
        assert errors == []
    
    def test_validate_skill_draft_missing_field(self):
        """Missing required field should fail validation."""
        draft = {
            "name": "Test Skill",
            "description": "Description",
            "trigger_description": "Trigger",
            # Missing instructions
        }
        is_valid, errors = skill_validation_service.validate_skill_draft(draft)
        assert is_valid is False
        assert any("instructions" in e for e in errors)
    
    def test_validate_skill_draft_empty_string(self):
        """Empty string field should fail validation."""
        draft = {
            "name": "",
            "description": "Description",
            "trigger_description": "Trigger",
            "instructions": "Instructions"
        }
        is_valid, errors = skill_validation_service.validate_skill_draft(draft)
        assert is_valid is False
        assert any("name" in e for e in errors)
    
    def test_validate_skill_draft_field_too_long(self):
        """Field exceeding max length should fail validation."""
        draft = {
            "name": "x" * 101,  # Max 100
            "description": "Description",
            "trigger_description": "Trigger",
            "instructions": "Instructions"
        }
        is_valid, errors = skill_validation_service.validate_skill_draft(draft)
        assert is_valid is False
        assert any("100 characters" in e for e in errors)
    
    def test_validate_skill_draft_all_fields_too_long(self):
        """All fields exceeding max length should fail validation."""
        draft = {
            "name": "x" * 101,
            "description": "x" * 501,
            "trigger_description": "x" * 501,
            "instructions": "x" * 2001,
        }
        is_valid, errors = skill_validation_service.validate_skill_draft(draft)
        assert is_valid is False
        assert len(errors) == 4


class TestSkillGenerationService:
    """Tests for SkillGenerationService."""
    
    @patch("server.services.skill_generation_service.client")
    def test_generate_draft_success(self, mock_client):
        """Successful generation should return structured draft."""
        # Mock LLM response
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = json.dumps({
            "name": "Analyze Financial Report",
            "description": "Extracts key financial metrics from uploaded reports",
            "trigger_description": "When a user uploads a financial report and requests analysis",
            "instructions": "1. Parse the document\n2. Extract metrics\n3. Compare to prior periods\n4. Generate structured summary"
        })
        mock_client.chat.completions.create.return_value = mock_completion
        
        service = SkillGenerationService(model="gpt-4o-mini")
        draft = service.generate_draft("Create a skill to analyze financial reports")
        
        assert draft["name"] == "Analyze Financial Report"
        assert "financial metrics" in draft["description"]
        assert "upload" in draft["trigger_description"]
        assert "Parse" in draft["instructions"]
    
    @patch("server.services.skill_generation_service.client")
    def test_generate_draft_empty_request(self, mock_client):
        """Empty request should raise SkillGenerationError."""
        service = SkillGenerationService()
        
        with pytest.raises(SkillGenerationError) as exc_info:
            service.generate_draft("")
        
        assert exc_info.value.code == "EMPTY_REQUEST"
        mock_client.chat.completions.create.assert_not_called()
    
    @patch("server.services.skill_generation_service.client")
    def test_generate_draft_request_too_long(self, mock_client):
        """Request exceeding max length should raise SkillGenerationError."""
        service = SkillGenerationService()
        long_request = "x" * 5001
        
        with pytest.raises(SkillGenerationError) as exc_info:
            service.generate_draft(long_request)
        
        assert exc_info.value.code == "REQUEST_TOO_LONG"
        mock_client.chat.completions.create.assert_not_called()
    
    @patch("server.services.skill_generation_service.client")
    def test_generate_draft_malformed_json(self, mock_client):
        """Malformed JSON response should raise SkillGenerationError."""
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "not valid json {"
        mock_client.chat.completions.create.return_value = mock_completion
        
        service = SkillGenerationService()
        
        with pytest.raises(SkillGenerationError) as exc_info:
            service.generate_draft("Create a skill")
        
        assert exc_info.value.code == "MALFORMED_RESPONSE"
    
    @patch("server.services.skill_generation_service.client")
    def test_generate_draft_validation_failure(self, mock_client):
        """Draft failing validation should raise SkillGenerationError."""
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        # Missing required field
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
    
    @patch("server.services.skill_generation_service.client")
    def test_generate_draft_llm_error(self, mock_client):
        """LLM API error should raise SkillGenerationError."""
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        service = SkillGenerationService()
        
        with pytest.raises(SkillGenerationError) as exc_info:
            service.generate_draft("Create a skill")
        
        assert exc_info.value.code == "LLM_ERROR"
    
    # --- Edit/Regenerate Tests ---
    
    @patch("server.services.skill_generation_service.client")
    def test_edit_draft_partial_change(self, mock_client):
        """Partial edit should update only specified fields."""
        current_draft = {
            "name": "Monitor Competitor News",
            "description": "Tracks competitor news",
            "trigger_description": "When user adds to watchlist",
            "instructions": "1. Search news\n2. Summarize"
        }
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        # Only trigger_description changed
        mock_completion.choices[0].message.content = json.dumps({
            "name": "Monitor Competitor News",
            "description": "Tracks competitor news",
            "trigger_description": "On a scheduled basis daily at 9 AM",
            "instructions": "1. Search news\n2. Summarize"
        })
        mock_client.chat.completions.create.return_value = mock_completion
        
        service = SkillGenerationService()
        updated = service.edit_draft(current_draft, "Change the trigger to run daily at 9 AM", regenerate_all=False)
        
        assert updated["trigger_description"] == "On a scheduled basis daily at 9 AM"
        assert updated["name"] == "Monitor Competitor News"  # Unchanged
        assert updated["description"] == "Tracks competitor news"  # Unchanged
        assert updated["instructions"] == "1. Search news\n2. Summarize"  # Unchanged
    
    @patch("server.services.skill_generation_service.client")
    def test_edit_draft_rename(self, mock_client):
        """Rename should update only the name field."""
        current_draft = {
            "name": "Analyze Financial Report",
            "description": "Extracts metrics",
            "trigger_description": "When upload",
            "instructions": "1. Parse\n2. Extract"
        }
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = json.dumps({
            "name": "Financial Report Analyzer",
            "description": "Extracts metrics",
            "trigger_description": "When upload",
            "instructions": "1. Parse\n2. Extract"
        })
        mock_client.chat.completions.create.return_value = mock_completion
        
        service = SkillGenerationService()
        updated = service.edit_draft(current_draft, "Rename to Financial Report Analyzer", regenerate_all=False)
        
        assert updated["name"] == "Financial Report Analyzer"
        assert updated["description"] == "Extracts metrics"
    
    @patch("server.services.skill_generation_service.client")
    def test_edit_draft_make_instructions_detailed(self, mock_client):
        """Make instructions more detailed should expand instructions field."""
        current_draft = {
            "name": "Test Skill",
            "description": "Description",
            "trigger_description": "Trigger",
            "instructions": "1. Parse\n2. Extract"
        }
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = json.dumps({
            "name": "Test Skill",
            "description": "Description",
            "trigger_description": "Trigger",
            "instructions": "1. Parse the uploaded document to extract text content and tables. Handle PDF, DOCX, and text formats. If parsing fails, log error and notify user.\n2. Identify the document type...\n3. Extract key financial metrics...\n4. Compare metrics to prior periods...\n5. Analyze management commentary...\n6. Structure output...\n7. Cite specific sections...\n8. If data is missing, explicitly note limitations."
        })
        mock_client.chat.completions.create.return_value = mock_completion
        
        service = SkillGenerationService()
        updated = service.edit_draft(current_draft, "Make instructions more detailed with error handling", regenerate_all=False)
        
        assert len(updated["instructions"]) > len(current_draft["instructions"])
        assert "error handling" in updated["instructions"].lower() or "fails" in updated["instructions"].lower()
        assert updated["name"] == "Test Skill"
    
    @patch("server.services.skill_generation_service.client")
    def test_edit_draft_regenerate_all(self, mock_client):
        """Complete regeneration should create new skill from scratch."""
        current_draft = {
            "name": "Monitor Competitor News",
            "description": "Tracks competitor news",
            "trigger_description": "When user adds to watchlist",
            "instructions": "1. Search news\n2. Summarize"
        }
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        # Completely different skill
        mock_completion.choices[0].message.content = json.dumps({
            "name": "Track SEC Filings",
            "description": "Monitors SEC EDGAR for new filings from tracked companies",
            "trigger_description": "When a new SEC filing is detected for tracked companies",
            "instructions": "1. Monitor SEC EDGAR RSS feed for new filings\n2. Filter by tracked company tickers\n3. Parse filing type (10-K, 10-Q, 8-K)\n4. Extract key data points\n5. Alert user with summary"
        })
        mock_client.chat.completions.create.return_value = mock_completion
        
        service = SkillGenerationService()
        updated = service.edit_draft(current_draft, "Regenerate completely for tracking SEC filings instead", regenerate_all=True)
        
        assert updated["name"] == "Track SEC Filings"
        assert "SEC" in updated["description"]
        assert "SEC" in updated["trigger_description"]
        assert "EDGAR" in updated["instructions"]
    
    @patch("server.services.skill_generation_service.client")
    def test_edit_draft_empty_current_draft(self, mock_client):
        """Empty current draft should raise SkillGenerationError."""
        service = SkillGenerationService()
        
        with pytest.raises(SkillGenerationError) as exc_info:
            service.edit_draft({}, "Change the trigger", regenerate_all=False)
        
        assert exc_info.value.code == "EMPTY_DRAFT"
        mock_client.chat.completions.create.assert_not_called()
    
    @patch("server.services.skill_generation_service.client")
    def test_edit_draft_invalid_current_draft(self, mock_client):
        """Invalid current draft should raise SkillGenerationError."""
        service = SkillGenerationService()
        invalid_draft = {"name": "Test"}  # Missing required fields
        
        with pytest.raises(SkillGenerationError) as exc_info:
            service.edit_draft(invalid_draft, "Change the trigger", regenerate_all=False)
        
        assert exc_info.value.code == "INVALID_DRAFT"
        mock_client.chat.completions.create.assert_not_called()
    
    @patch("server.services.skill_generation_service.client")
    def test_edit_draft_empty_feedback(self, mock_client):
        """Empty feedback should raise SkillGenerationError."""
        current_draft = {
            "name": "Test Skill",
            "description": "Description",
            "trigger_description": "Trigger",
            "instructions": "Instructions"
        }
        service = SkillGenerationService()
        
        with pytest.raises(SkillGenerationError) as exc_info:
            service.edit_draft(current_draft, "", regenerate_all=False)
        
        assert exc_info.value.code == "EMPTY_FEEDBACK"
        mock_client.chat.completions.create.assert_not_called()
    
    @patch("server.services.skill_generation_service.client")
    def test_edit_draft_feedback_too_long(self, mock_client):
        """Feedback exceeding max length should raise SkillGenerationError."""
        current_draft = {
            "name": "Test Skill",
            "description": "Description",
            "trigger_description": "Trigger",
            "instructions": "Instructions"
        }
        service = SkillGenerationService()
        long_feedback = "x" * 5001
        
        with pytest.raises(SkillGenerationError) as exc_info:
            service.edit_draft(current_draft, long_feedback, regenerate_all=False)
        
        assert exc_info.value.code == "FEEDBACK_TOO_LONG"
        mock_client.chat.completions.create.assert_not_called()
    
    @patch("server.services.skill_generation_service.client")
    def test_edit_draft_malformed_json(self, mock_client):
        """Malformed JSON response should raise SkillGenerationError."""
        current_draft = {
            "name": "Test Skill",
            "description": "Description",
            "trigger_description": "Trigger",
            "instructions": "Instructions"
        }
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "not valid json {"
        mock_client.chat.completions.create.return_value = mock_completion
        
        service = SkillGenerationService()
        
        with pytest.raises(SkillGenerationError) as exc_info:
            service.edit_draft(current_draft, "Change the trigger", regenerate_all=False)
        
        assert exc_info.value.code == "MALFORMED_RESPONSE"
    
    @patch("server.services.skill_generation_service.client")
    def test_edit_draft_validation_failure(self, mock_client):
        """Draft failing validation should raise SkillGenerationError."""
        current_draft = {
            "name": "Test Skill",
            "description": "Description",
            "trigger_description": "Trigger",
            "instructions": "Instructions"
        }
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        # Missing required field
        mock_completion.choices[0].message.content = json.dumps({
            "name": "Test Skill",
            "description": "Description",
            "trigger_description": "Trigger"
            # Missing instructions
        })
        mock_client.chat.completions.create.return_value = mock_completion
        
        service = SkillGenerationService()
        
        with pytest.raises(SkillGenerationError) as exc_info:
            service.edit_draft(current_draft, "Change the trigger", regenerate_all=False)
        
        assert exc_info.value.code == "VALIDATION_FAILED"
        assert "instructions" in str(exc_info.value.details)
    
    @patch("server.services.skill_generation_service.client")
    def test_edit_draft_llm_error(self, mock_client):
        """LLM API error should raise SkillGenerationError."""
        current_draft = {
            "name": "Test Skill",
            "description": "Description",
            "trigger_description": "Trigger",
            "instructions": "Instructions"
        }
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        service = SkillGenerationService()
        
        with pytest.raises(SkillGenerationError) as exc_info:
            service.edit_draft(current_draft, "Change the trigger", regenerate_all=False)
        
        assert exc_info.value.code == "LLM_ERROR"


class TestSkillSchemas:
    """Tests for skill generation schemas."""
    
    def test_skill_draft_request_valid(self):
        """Valid request should parse correctly."""
        request = SkillDraftRequest(request="Create a skill to analyze reports")
        assert request.request == "Create a skill to analyze reports"
    
    def test_skill_draft_request_empty(self):
        """Empty request should fail validation."""
        with pytest.raises(Exception):  # Pydantic validation error
            SkillDraftRequest(request="")
    
    def test_skill_draft_request_too_long(self):
        """Too long request should fail validation."""
        with pytest.raises(Exception):
            SkillDraftRequest(request="x" * 5001)
    
    def test_skill_draft_response_valid(self):
        """Valid response should parse correctly."""
        response = SkillDraftResponse(
            name="Test Skill",
            description="Description",
            trigger_description="Trigger",
            instructions="Instructions"
        )
        assert response.name == "Test Skill"
    
    def test_skill_draft_error(self):
        """Error schema should parse correctly."""
        error = SkillDraftError(
            code="TEST_ERROR",
            message="Test message",
            details={"key": "value"}
        )
        assert error.code == "TEST_ERROR"
        assert error.details == {"key": "value"}
    
    def test_skill_draft_error_no_details(self):
        """Error schema without details should work."""
        error = SkillDraftError(code="TEST", message="Message")
        assert error.details is None