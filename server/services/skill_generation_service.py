"""Skill Generation Service.

Generates structured Skill drafts from natural language user requests using LLM.
Supports editing and regenerating existing drafts based on user feedback.
"""

import json
import logging
from typing import Any, Dict, Optional
from server.utils.markdown import render_rich_text

from openai import OpenAI

from server.config import settings
from server.prompts.skill_generation import (
    skill_generation_prompt,
    skill_generation_system_prompt,
    skill_edit_prompt,
    skill_edit_system_prompt,
)
from server.services.skill_validation_service import skill_validation_service

logger = logging.getLogger(__name__)

client = OpenAI(api_key=settings.OPENAI_API_KEY)


class SkillGenerationError(Exception):
    """Raised when skill generation fails."""
    def __init__(self, message: str, code: str = "GENERATION_FAILED", details: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class SkillGenerationService:
    """Service for generating and editing Skill drafts from user requests."""
    
    def __init__(self, model: Optional[str] = None):
        self.model = model or settings.LLM_MODEL
    
    def generate_draft(
        self,
        user_request: str,
        user_id: Optional[str] = None,
        db: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Generate a Skill draft from a natural language request.
        
        Args:
            user_request: User's natural language description of the desired skill
            user_id: Optional user ID for ensuring unique skill name
            db: Optional database session for ensuring unique skill name
            
        Returns:
            Dictionary with keys: name, description, trigger_description, instructions
            
        Raises:
            SkillGenerationError: On validation failure, LLM error, or malformed response
        """
        if not user_request or not user_request.strip():
            raise SkillGenerationError(
                "User request cannot be empty",
                code="EMPTY_REQUEST"
            )
        
        if len(user_request) > 5000:
            raise SkillGenerationError(
                "User request exceeds maximum length of 5000 characters",
                code="REQUEST_TOO_LONG"
            )
        
        messages = [
            {"role": "system", "content": skill_generation_system_prompt()},
            {"role": "user", "content": skill_generation_prompt(user_request.strip())},
        ]
        
        draft = self._call_llm_and_validate(messages, "generate")

        if user_id and db and draft.get("name"):
            from server import crud
            draft["name"] = crud.skill.get_unique_skill_name(
                db, user_id=user_id, base_name=draft["name"]
            )

        draft["description_html"] = render_rich_text(draft.get("description", ""))
        draft["trigger_description_html"] = render_rich_text(draft.get("trigger_description", ""))
        draft["instructions_html"] = render_rich_text(draft.get("instructions", ""))

        return draft
    
    def edit_draft(
        self,
        current_draft: Dict[str, Any],
        user_feedback: str,
        regenerate_all: bool = False,
        user_id: Optional[str] = None,
        db: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Edit or regenerate a Skill draft based on user feedback.
        
        Args:
            current_draft: The existing skill draft with all four fields
            user_feedback: Natural language description of desired changes
            regenerate_all: If True, regenerate complete skill; if False, apply partial changes
            user_id: Optional user ID for ensuring unique skill name
            db: Optional database session for ensuring unique skill name
            
        Returns:
            Updated dictionary with all four fields
            
        Raises:
            SkillGenerationError: On validation failure, LLM error, or malformed response
        """
        if not current_draft:
            raise SkillGenerationError(
                "Current draft cannot be empty",
                code="EMPTY_DRAFT"
            )
        
        # Validate current draft structure
        is_valid, errors = skill_validation_service.validate_skill_draft(current_draft)
        if not is_valid:
            raise SkillGenerationError(
                "Current draft is invalid",
                code="INVALID_DRAFT",
                details={"errors": errors}
            )
        
        if not user_feedback or not user_feedback.strip():
            raise SkillGenerationError(
                "User feedback cannot be empty",
                code="EMPTY_FEEDBACK"
            )
        
        if len(user_feedback) > 5000:
            raise SkillGenerationError(
                "User feedback exceeds maximum length of 5000 characters",
                code="FEEDBACK_TOO_LONG"
            )
        
        messages = [
            {"role": "system", "content": skill_edit_system_prompt()},
            {"role": "user", "content": skill_edit_prompt(current_draft, user_feedback.strip(), regenerate_all)},
        ]
        
        draft = self._call_llm_and_validate(messages, "edit")

        if user_id and db and draft.get("name"):
            from server import crud
            exclude_id = current_draft.get("id")
            draft["name"] = crud.skill.get_unique_skill_name(
                db, user_id=user_id, base_name=draft["name"], exclude_skill_id=exclude_id
            )

        draft["description_html"] = render_rich_text(draft.get("description", ""))
        draft["trigger_description_html"] = render_rich_text(draft.get("trigger_description", ""))
        draft["instructions_html"] = render_rich_text(draft.get("instructions", ""))

        return draft
    
    def _call_llm_and_validate(self, messages: list, operation: str) -> Dict[str, Any]:
        """Call LLM and validate response. Shared logic for generate and edit."""
        try:
            logger.info(f"Skill {operation}: calling LLM with {len(messages)} messages")
            
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                **({"max_completion_tokens": 2048} if "gpt-5" in self.model.lower() else {"max_tokens": 2048}),
                **({} if any(p in self.model.lower() for p in ("o1", "o2", "o3")) else {"temperature": 0.1}),
                response_format={"type": "json_object"},
            )
            
            raw_response = completion.choices[0].message.content
            if not raw_response:
                raise SkillGenerationError(
                    "LLM returned empty response",
                    code="EMPTY_RESPONSE"
                )
            
            # Parse JSON response
            try:
                draft = json.loads(raw_response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {raw_response[:500]}")
                raise SkillGenerationError(
                    "LLM returned malformed JSON",
                    code="MALFORMED_RESPONSE",
                    details={"parse_error": str(e), "raw_response": raw_response[:500]}
                )
            
            # Validate the draft structure
            is_valid, errors = skill_validation_service.validate_skill_draft(draft)
            if not is_valid:
                logger.error(f"Generated draft validation failed: {errors}")
                raise SkillGenerationError(
                    "Generated draft failed validation",
                    code="VALIDATION_FAILED",
                    details={"errors": errors, "draft": draft}
                )
            
            logger.info(f"Successfully {operation}d skill draft: {draft.get('name')}")
            return draft
            
        except SkillGenerationError:
            raise
        except Exception as e:
            logger.exception(f"Skill {operation} failed: {e}")
            raise SkillGenerationError(
                f"Skill {operation} failed: {str(e)}",
                code="LLM_ERROR",
                details={"error_type": type(e).__name__}
            )
    
    def generate_draft_streaming(self, user_request: str):
        """Generate a Skill draft with streaming response (for future use).
        
        Currently not implemented - returns full draft at once.
        """
        draft = self.generate_draft(user_request)
        yield draft


# Singleton instance
skill_generation_service = SkillGenerationService()