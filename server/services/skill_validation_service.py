"""Skill Validation Service.

Provides deterministic structural validation and optional AI-based quality checks
for Skill drafts before saving to the Skill Library.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from server.config import settings
from server.models.skill import Skill, SkillStatus
from server.db.session import SessionLocal

logger = logging.getLogger(__name__)


class SkillValidationError(Exception):
    """Raised when skill validation fails structurally."""
    def __init__(self, message: str, code: str = "VALIDATION_FAILED", details: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class ValidationResult:
    """Result of skill validation with errors and warnings."""
    
    def __init__(self):
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
    
    def add_error(self, field: str, message: str, code: str = "VALIDATION_ERROR", details: Optional[Dict] = None):
        """Add a validation error."""
        self.errors.append({
            "field": field,
            "message": message,
            "code": code,
            "details": details or {}
        })
    
    def add_warning(self, field: str, message: str, code: str = "VALIDATION_WARNING", details: Optional[Dict] = None):
        """Add a validation warning."""
        self.warnings.append({
            "field": field,
            "message": message,
            "code": code,
            "details": details or {}
        })
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings
        }


class SkillValidationService:
    """Service for validating Skill drafts."""
    
    # Minimum length requirements for meaningful content
    MIN_NAME_LENGTH = 3
    MIN_DESCRIPTION_LENGTH = 10
    MIN_TRIGGER_LENGTH = 20
    MIN_INSTRUCTIONS_LENGTH = 50
    
    # Maximum length limits
    MAX_NAME_LENGTH = 100
    MAX_DESCRIPTION_LENGTH = 2000
    MAX_TRIGGER_LENGTH = 1000
    MAX_INSTRUCTIONS_LENGTH = 10000
    
    # Weak trigger patterns (too generic)
    WEAK_TRIGGER_PATTERNS = [
        r"^when user asks?",
        r"^when user wants?",
        r"^when user says?",
        r"^on request",
        r"^on demand",
        r"^when triggered",
        r"^user asks",
        r"^user wants",
    ]
    
    # Quality indicators for instructions
    INSTRUCTION_QUALITY_INDICATORS = [
        r"\d+\.",  # Numbered steps
        r"step\s+\d+",
        r"first|second|third|then|next|finally",
        r"if|else|when|case",
        r"error|exception|fail|handle",
        r"output|return|format|structure",
    ]
    
    def __init__(self, db_session: Optional[Any] = None):
        self._db = db_session
    
    def _get_db(self):
        """Get database session."""
        if self._db:
            return self._db
        return SessionLocal()
    
    def validate_draft(
        self,
        draft: Dict[str, Any],
        user_id: Optional[str] = None,
        check_uniqueness: bool = True,
        include_ai_quality: bool = False
    ) -> ValidationResult:
        """Validate a skill draft with deterministic checks.
        
        Args:
            draft: Skill draft dictionary with name, description, trigger_description, instructions
            user_id: Optional user ID for uniqueness check
            check_uniqueness: Whether to check name uniqueness for user
            include_ai_quality: Whether to include AI-based quality check (separate)
            
        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult()
        
        # Run deterministic structural validation
        self._validate_required_fields(draft, result)
        self._validate_field_lengths(draft, result)
        self._validate_trigger_specificity(draft, result)
        self._validate_instructions_quality(draft, result)
        
        # Check name uniqueness if user_id provided
        if check_uniqueness and user_id and draft.get("name"):
            self._validate_name_uniqueness(draft["name"], user_id, result)
        
        # Run AI quality check if requested (separate from deterministic)
        if include_ai_quality and result.is_valid:
            self._run_ai_quality_check(draft, result)
        
        return result
    
    def _validate_required_fields(self, draft: Dict[str, Any], result: ValidationResult):
        """Validate all required fields are present and non-empty."""
        required_fields = {
            "name": "Skill name is required",
            "description": "Skill description is required",
            "trigger_description": "Trigger description is required",
            "instructions": "Instructions are required",
        }
        
        for field, message in required_fields.items():
            value = draft.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                result.add_error(
                    field=field,
                    message=message,
                    code="MISSING_REQUIRED_FIELD"
                )
    
    def _validate_field_lengths(self, draft: Dict[str, Any], result: ValidationResult):
        """Validate field lengths are within limits."""
        length_checks = [
            ("name", self.MIN_NAME_LENGTH, self.MAX_NAME_LENGTH),
            ("description", self.MIN_DESCRIPTION_LENGTH, self.MAX_DESCRIPTION_LENGTH),
            ("trigger_description", self.MIN_TRIGGER_LENGTH, self.MAX_TRIGGER_LENGTH),
            ("instructions", self.MIN_INSTRUCTIONS_LENGTH, self.MAX_INSTRUCTIONS_LENGTH),
        ]
        
        for field, min_len, max_len in length_checks:
            value = draft.get(field, "")
            if isinstance(value, str):
                stripped = value.strip()
                if len(stripped) < min_len:
                    result.add_error(
                        field=field,
                        message=f"{field.replace('_', ' ').title()} must be at least {min_len} characters",
                        code="FIELD_TOO_SHORT",
                        details={"min_length": min_len, "actual_length": len(stripped)}
                    )
                elif len(stripped) > max_len:
                    result.add_error(
                        field=field,
                        message=f"{field.replace('_', ' ').title()} must be at most {max_len} characters",
                        code="FIELD_TOO_LONG",
                        details={"max_length": max_len, "actual_length": len(stripped)}
                    )
    
    def _validate_trigger_specificity(self, draft: Dict[str, Any], result: ValidationResult):
        """Validate trigger description is specific enough for future matching."""
        trigger = draft.get("trigger_description", "")
        if not isinstance(trigger, str) or not trigger.strip():
            return  # Already caught by required field check
        
        trigger_lower = trigger.lower().strip()
        
        # Check for weak/generic trigger patterns
        for pattern in self.WEAK_TRIGGER_PATTERNS:
            if re.search(pattern, trigger_lower):
                result.add_warning(
                    field="trigger_description",
                    message=f"Trigger description may be too generic: '{trigger[:50]}...'. Consider specifying concrete conditions (e.g., file upload, scheduled time, specific keywords).",
                    code="WEAK_TRIGGER",
                    details={"matched_pattern": pattern}
                )
                break  # Only warn once
        
        # Check for specificity indicators
        specificity_indicators = [
            r"upload|file|document|pdf",
            r"schedule|cron|daily|weekly|monthly",
            r"keyword|phrase|mention|contain",
            r"event|webhook|api|callback",
            r"time|date|interval",
            r"workspace|chat|session",
        ]
        
        has_specificity = any(re.search(ind, trigger_lower) for ind in specificity_indicators)
        if not has_specificity and len(trigger_lower) > 30:
            result.add_warning(
                field="trigger_description",
                message="Trigger description lacks specific triggering conditions. Consider adding: file upload, scheduled time, keywords, events, or workspace context.",
                code="LOW_TRIGGER_SPECIFICITY"
            )
    
    def _validate_instructions_quality(self, draft: Dict[str, Any], result: ValidationResult):
        """Validate instructions have meaningful operational content."""
        instructions = draft.get("instructions", "")
        if not isinstance(instructions, str) or not instructions.strip():
            return  # Already caught by required field check
        
        instructions_lower = instructions.lower()
        
        # Count quality indicators
        quality_score = 0
        for indicator in self.INSTRUCTION_QUALITY_INDICATORS:
            if re.search(indicator, instructions_lower):
                quality_score += 1
        
        # Check for step-like structure
        has_numbered_steps = bool(re.search(r"\d+\.", instructions))
        has_bullet_points = bool(re.search(r"[-*•]\s", instructions))
        has_structured_steps = has_numbered_steps or has_bullet_points
        
        if not has_structured_steps and len(instructions) > 100:
            result.add_warning(
                field="instructions",
                message="Instructions lack clear step structure. Consider using numbered steps (1., 2., 3.) or bullet points for clarity.",
                code="UNSTRUCTURED_INSTRUCTIONS"
            )
        
        if quality_score < 3 and len(instructions) > 100:
            result.add_warning(
                field="instructions",
                message="Instructions may lack operational detail. Consider adding: conditional logic, error handling, output formats, tool/data source references.",
                code="LOW_INSTRUCTION_QUALITY",
                details={"quality_score": quality_score, "max_score": len(self.INSTRUCTION_QUALITY_INDICATORS)}
            )
        
        # Check for vague language
        vague_terms = ["etc", "and so on", "as needed", "appropriately", "handle properly", "deal with"]
        vague_count = sum(1 for term in vague_terms if term in instructions_lower)
        if vague_count > 2:
            result.add_warning(
                field="instructions",
                message=f"Instructions contain {vague_count} vague terms. Replace with specific actions and criteria.",
                code="VAGUE_INSTRUCTIONS"
            )
    
    def _validate_name_uniqueness(self, name: str, user_id: str, result: ValidationResult):
        """Check if skill name is unique for the user."""
        db = self._get_db()
        try:
            existing = db.query(Skill).filter(
                Skill.user_id == user_id,
                Skill.name == name
            ).first()
            
            if existing:
                result.add_error(
                    field="name",
                    message=f"A skill with name '{name}' already exists for this user",
                    code="DUPLICATE_NAME",
                    details={"existing_skill_id": existing.id}
                )
        finally:
            if not self._db:
                db.close()
    
    def _run_ai_quality_check(self, draft: Dict[str, Any], result: ValidationResult):
        """Run AI-based quality assessment (separate from deterministic validation).
        
        This is kept separate from deterministic validation as requested.
        """
        # This would call an LLM to assess quality - placeholder for future implementation
        # For now, we just log that it would run
        logger.info("AI quality check would run here (placeholder)")
        
        # Example of what AI check could add:
        # result.add_warning(
        #     field="overall",
        #     message="AI assessment: Instructions could be more specific about error handling",
        #     code="AI_QUALITY_SUGGESTION"
        # )

    def validate_skill_draft(self, draft: Dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate a generated skill draft. Returns (is_valid, errors).
        
        This is a simplified validation for use during generation (before full validation).
        """
        errors = []
        
        required_fields = ["name", "description", "trigger_description", "instructions"]
        for field in required_fields:
            if field not in draft:
                errors.append(f"Missing required field: {field}")
                continue
            value = draft[field]
            if not isinstance(value, str) or not value.strip():
                errors.append(f"Field '{field}' must be a non-empty string")
        
        if "name" in draft and len(draft["name"]) > 100:
            errors.append("Name must be 100 characters or less")
        if "description" in draft and len(draft["description"]) > 2000:
            errors.append("Description must be 2000 characters or less")
        if "trigger_description" in draft and len(draft["trigger_description"]) > 1000:
            errors.append("Trigger description must be 1000 characters or less")
        if "instructions" in draft and len(draft["instructions"]) > 10000:
            errors.append("Instructions must be 10000 characters or less")
        
        return len(errors) == 0, errors


# Singleton instance
skill_validation_service = SkillValidationService()