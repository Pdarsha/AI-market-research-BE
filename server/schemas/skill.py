from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SkillBase(BaseModel):
    name: str = Field(..., description="Name of the skill", min_length=1, max_length=255)
    description: Optional[str] = Field(None, description="Description of the skill")
    trigger_description: Optional[str] = Field(None, description="Description of when this skill should be triggered")
    instructions: Optional[str] = Field(None, description="Instructions for the skill")
    description_html: Optional[str] = Field(None, description="Rendered HTML description")
    trigger_description_html: Optional[str] = Field(None, description="Rendered HTML trigger description")
    instructions_html: Optional[str] = Field(None, description="Rendered HTML instructions")
    status: str = Field(default="draft", description="Status of the skill: draft, published")
    version: str = Field(default="1.0.0", description="Version of the skill")

    class Config:
        from_attributes = True


class SkillCreate(SkillBase):
    user_id: Optional[str] = None


class SkillUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    trigger_description: Optional[str] = None
    instructions: Optional[str] = None
    description_html: Optional[str] = None
    trigger_description_html: Optional[str] = None
    instructions_html: Optional[str] = None
    status: Optional[str] = None
    version: Optional[str] = None


class SkillResponse(SkillBase):
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    published_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class SkillListItem(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    status: str
    version: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# --- Skill Generation Schemas ---

class SkillDraftRequest(BaseModel):
    """Request to generate a skill draft from natural language."""
    request: str = Field(..., description="Natural language description of the desired skill", min_length=1, max_length=5000)


class SkillDraftResponse(BaseModel):
    """Generated skill draft (not yet persisted)."""
    id: Optional[str] = Field(default=None, description="Optional skill ID if updating existing draft")
    name: str = Field(..., description="Generated skill name")
    description: str = Field(..., description="Generated skill description")
    trigger_description: str = Field(..., description="Generated trigger description")
    instructions: str = Field(..., description="Generated operational instructions")
    description_html: Optional[str] = Field(default=None, description="Rendered HTML description")
    trigger_description_html: Optional[str] = Field(default=None, description="Rendered HTML trigger description")
    instructions_html: Optional[str] = Field(default=None, description="Rendered HTML instructions")


class SkillDraftError(BaseModel):
    """Error response for skill draft generation."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(None, description="Additional error details")


# --- Skill Validation Schemas ---

class ValidationIssue(BaseModel):
    """Individual validation issue (error or warning)."""
    field: str = Field(..., description="Field name that has the issue")
    message: str = Field(..., description="Human-readable issue description")
    code: str = Field(..., description="Issue code for programmatic handling")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")


class SkillValidationRequest(BaseModel):
    """Request to validate a skill draft."""
    draft: SkillDraftResponse = Field(..., description="The skill draft to validate")
    check_uniqueness: bool = Field(default=True, description="Whether to check name uniqueness for the current user")
    include_ai_quality: bool = Field(default=False, description="Whether to include AI-based quality check (separate from deterministic validation)")


class SkillValidationResponse(BaseModel):
    """Response for skill validation."""
    valid: bool = Field(..., description="Whether the draft passed all validation checks")
    errors: List[ValidationIssue] = Field(default=[], description="Validation errors that must be fixed")
    warnings: List[ValidationIssue] = Field(default=[], description="Validation warnings for improvement")


# --- Skill Save/Publish Schemas ---

class SkillSaveRequest(BaseModel):
    """Request to save a validated skill draft to the Skill Library."""
    draft: SkillDraftResponse = Field(..., description="The validated skill draft to save")
    status: str = Field(default="draft", description="Initial status: draft")


class SkillPublishRequest(BaseModel):
    """Request to publish a saved skill."""
    skill_id: str = Field(..., description="ID of the skill to publish")