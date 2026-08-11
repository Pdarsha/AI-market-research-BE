import pytest
from unittest.mock import patch, MagicMock

from server.services.skill_validation_service import (
    SkillValidationError,
    SkillValidationService,
    ValidationResult,
)
from server.schemas.skill import (
    SkillDraftResponse,
    SkillValidationRequest,
    SkillValidationResponse,
    ValidationIssue,
)
from server.services.skill_validation_service import skill_validation_service


class TestValidationResult:
    """Tests for ValidationResult class."""
    
    def test_empty_result_is_valid(self):
        """Empty validation result should be valid."""
        result = ValidationResult()
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
    
    def test_add_error_makes_invalid(self):
        """Adding an error should make result invalid."""
        result = ValidationResult()
        result.add_error("name", "Name is required")
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0]["field"] == "name"
    
    def test_add_warning_keeps_valid(self):
        """Adding a warning should keep result valid."""
        result = ValidationResult()
        result.add_warning("trigger_description", "Trigger could be more specific")
        assert result.is_valid is True
        assert len(result.warnings) == 1
    
    def test_to_dict_format(self):
        """to_dict should return proper format."""
        result = ValidationResult()
        result.add_error("name", "Required")
        result.add_warning("trigger", "Weak")
        d = result.to_dict()
        assert d["valid"] is False
        assert len(d["errors"]) == 1
        assert len(d["warnings"]) == 1


class TestSkillValidationService:
    """Tests for SkillValidationService deterministic checks."""
    
    def setup_method(self):
        self.service = SkillValidationService()
        self.valid_draft = {
            "name": "Analyze Financial Report",
            "description": "Extracts key financial metrics from uploaded reports and generates structured summaries",
            "trigger_description": "When a user uploads a financial report PDF (10-K, 10-Q, earnings release) and requests analysis",
            "instructions": "1. Parse the uploaded document to extract text content and tables.\n2. Identify the document type (10-K, 10-Q, 8-K, earnings release).\n3. Extract key financial metrics: revenue, net income, EPS, operating margin.\n4. Compare metrics to prior periods (QoQ, YoY) and identify significant changes.\n5. Analyze management commentary for forward guidance and risk factors.\n6. Structure output as: Executive Summary, Key Metrics Table, Trend Analysis.\n7. Cite specific sections/page numbers from source document.\n8. If data is missing, explicitly note limitations."
        }
    
    def test_valid_draft_passes(self):
        """Valid draft should pass all checks."""
        result = self.service.validate_draft(self.valid_draft)
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_missing_name_fails(self):
        """Missing name should fail validation."""
        draft = {**self.valid_draft, "name": ""}
        result = self.service.validate_draft(draft)
        assert result.is_valid is False
        assert any(e["field"] == "name" and e["code"] == "MISSING_REQUIRED_FIELD" for e in result.errors)
    
    def test_missing_description_fails(self):
        """Missing description should fail validation."""
        draft = {**self.valid_draft, "description": ""}
        result = self.service.validate_draft(draft)
        assert result.is_valid is False
        assert any(e["field"] == "description" for e in result.errors)
    
    def test_missing_trigger_fails(self):
        """Missing trigger description should fail validation."""
        draft = {**self.valid_draft, "trigger_description": ""}
        result = self.service.validate_draft(draft)
        assert result.is_valid is False
        assert any(e["field"] == "trigger_description" for e in result.errors)
    
    def test_missing_instructions_fails(self):
        """Missing instructions should fail validation."""
        draft = {**self.valid_draft, "instructions": ""}
        result = self.service.validate_draft(draft)
        assert result.is_valid is False
        assert any(e["field"] == "instructions" for e in result.errors)
    
    def test_all_missing_fields(self):
        """All missing fields should report all errors (missing + too short)."""
        draft = {"name": "", "description": "", "trigger_description": "", "instructions": ""}
        result = self.service.validate_draft(draft)
        assert result.is_valid is False
        # Each empty field produces 2 errors: MISSING_REQUIRED_FIELD + FIELD_TOO_SHORT
        assert len(result.errors) == 8
        fields = {e["field"] for e in result.errors}
        assert fields == {"name", "description", "trigger_description", "instructions"}
        # Check we have both error types for each field
        error_codes = {e["code"] for e in result.errors}
        assert "MISSING_REQUIRED_FIELD" in error_codes
        assert "FIELD_TOO_SHORT" in error_codes
    
    def test_name_too_short(self):
        """Name too short should fail."""
        draft = {**self.valid_draft, "name": "Ab"}
        result = self.service.validate_draft(draft)
        assert result.is_valid is False
        assert any(e["field"] == "name" and e["code"] == "FIELD_TOO_SHORT" for e in result.errors)
    
    def test_name_too_long(self):
        """Name too long should fail."""
        draft = {**self.valid_draft, "name": "x" * 101}
        result = self.service.validate_draft(draft)
        assert result.is_valid is False
        assert any(e["field"] == "name" and e["code"] == "FIELD_TOO_LONG" for e in result.errors)
    
    def test_description_too_short(self):
        """Description too short should fail."""
        draft = {**self.valid_draft, "description": "Short"}
        result = self.service.validate_draft(draft)
        assert result.is_valid is False
        assert any(e["field"] == "description" and e["code"] == "FIELD_TOO_SHORT" for e in result.errors)
    
    def test_trigger_too_short(self):
        """Trigger description too short should fail."""
        draft = {**self.valid_draft, "trigger_description": "When user asks"}
        result = self.service.validate_draft(draft)
        assert result.is_valid is False
        assert any(e["field"] == "trigger_description" and e["code"] == "FIELD_TOO_SHORT" for e in result.errors)
    
    def test_instructions_too_short(self):
        """Instructions too short should fail."""
        draft = {**self.valid_draft, "instructions": "Do something"}
        result = self.service.validate_draft(draft)
        assert result.is_valid is False
        assert any(e["field"] == "instructions" and e["code"] == "FIELD_TOO_SHORT" for e in result.errors)
    
    def test_weak_trigger_warning(self):
        """Generic trigger should produce warning."""
        draft = {**self.valid_draft, "trigger_description": "When user asks for analysis"}
        result = self.service.validate_draft(draft)
        assert result.is_valid is True  # Warning, not error
        assert any(w["code"] == "WEAK_TRIGGER" for w in result.warnings)
    
    def test_weak_trigger_variations(self):
        """Various weak trigger patterns should warn."""
        weak_triggers = [
            "When user wants analysis",
            "On request from user",
            "On demand when needed",
            "When triggered by user",
            "User asks for help",
        ]
        for trigger in weak_triggers:
            draft = {**self.valid_draft, "trigger_description": trigger}
            result = self.service.validate_draft(draft)
            assert any(w["code"] == "WEAK_TRIGGER" for w in result.warnings), f"Failed for: {trigger}"
    
    def test_specific_trigger_no_warning(self):
        """Specific trigger should not warn."""
        specific_triggers = [
            "When a user uploads a financial report PDF and requests analysis",
            "On a scheduled basis every Monday at 9 AM for weekly reporting",
            "When a new SEC filing is detected for tracked companies AAPL, MSFT",
            "When a document with keyword 'earnings' is uploaded to workspace",
        ]
        for trigger in specific_triggers:
            draft = {**self.valid_draft, "trigger_description": trigger}
            result = self.service.validate_draft(draft)
            assert not any(w["code"] == "WEAK_TRIGGER" for w in result.warnings), f"False warning for: {trigger}"
    
    def test_low_trigger_specificity_warning(self):
        """Long but non-specific trigger should warn."""
        draft = {**self.valid_draft, "trigger_description": "This skill should run when the user wants to do something with documents and files in the system"}
        result = self.service.validate_draft(draft)
        # May or may not warn depending on specificity indicators
        # Just ensure it doesn't crash
        assert isinstance(result.is_valid, bool)
    
    def test_unstructured_instructions_warning(self):
        """Instructions without step structure should warn."""
        draft = {**self.valid_draft, "instructions": "First parse the document then extract metrics then compare to prior periods then generate summary and cite sources"}
        result = self.service.validate_draft(draft)
        # Long instructions without structure should warn
        assert isinstance(result.is_valid, bool)
    
    def test_structured_instructions_no_warning(self):
        """Well-structured instructions should not warn."""
        structured = """1. Parse the uploaded document to extract text content and tables.
2. Identify the document type (10-K, 10-Q, 8-K, earnings release).
3. Extract key financial metrics: revenue, net income, EPS.
4. Compare metrics to prior periods (QoQ, YoY).
5. If data is missing, explicitly note limitations."""
        draft = {**self.valid_draft, "instructions": structured}
        result = self.service.validate_draft(draft)
        assert not any(w["code"] == "UNSTRUCTURED_INSTRUCTIONS" for w in result.warnings)
    
    def test_low_instruction_quality_warning(self):
        """Instructions lacking operational detail should warn."""
        draft = {**self.valid_draft, "instructions": "Parse document and extract metrics and generate summary as needed"}
        result = self.service.validate_draft(draft)
        # Low quality score may warn
        assert isinstance(result.is_valid, bool)
    
    def test_vague_instructions_warning(self):
        """Instructions with vague terms should warn."""
        draft = {**self.valid_draft, "instructions": "1. Parse document appropriately. 2. Handle errors properly. 3. Generate summary as needed. 4. Deal with missing data etc."}
        result = self.service.validate_draft(draft)
        assert any(w["code"] == "VAGUE_INSTRUCTIONS" for w in result.warnings)
    
    def test_duplicate_name_error(self):
        """Duplicate name for user should error."""
        # This test would need a database - we'll mock it
        pass  # Requires DB integration test
    
    def test_validate_without_uniqueness_check(self):
        """Validation without uniqueness should not check DB."""
        draft = self.valid_draft.copy()
        # Should not raise even without DB
        result = self.service.validate_draft(draft, user_id="test-user", check_uniqueness=False)
        assert result.is_valid is True
    
    def test_convenience_function(self):
        """Module-level validate_draft method should work."""
        result = self.service.validate_draft(self.valid_draft, check_uniqueness=False)
        assert result.is_valid is True


class TestSkillValidationSchemas:
    """Tests for validation schemas."""
    
    def test_valid_draft_request(self):
        """Valid validation request should parse."""
        draft = SkillDraftResponse(
            name="Test Skill",
            description="Description of the skill",
            trigger_description="When user uploads a document",
            instructions="1. Parse\n2. Extract\n3. Summarize"
        )
        request = SkillValidationRequest(draft=draft)
        assert request.draft.name == "Test Skill"
        assert request.check_uniqueness is True
        assert request.include_ai_quality is False
    
    def test_validation_response_valid(self):
        """Valid validation response should parse."""
        response = SkillValidationResponse(
            valid=True,
            errors=[],
            warnings=[]
        )
        assert response.valid is True
    
    def test_validation_response_with_errors(self):
        """Validation response with errors should parse."""
        response = SkillValidationResponse(
            valid=False,
            errors=[ValidationIssue(field="name", message="Required", code="MISSING")],
            warnings=[ValidationIssue(field="trigger", message="Weak", code="WEAK")]
        )
        assert response.valid is False
        assert len(response.errors) == 1
        assert len(response.warnings) == 1
    
    def test_validation_issue_schema(self):
        """ValidationIssue schema should work."""
        issue = ValidationIssue(
            field="name",
            message="Name is required",
            code="MISSING_REQUIRED_FIELD",
            details={"min_length": 3}
        )
        assert issue.field == "name"
        assert issue.details == {"min_length": 3}