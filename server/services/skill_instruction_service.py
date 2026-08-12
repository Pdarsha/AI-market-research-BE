"""Skill Instruction Service.

Fetches complete skill data directly from the database (the single source of truth)
and builds the skill instruction context for agent execution.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple
from sqlalchemy.orm import Session

from server import crud
from server.models.skill import Skill

logger = logging.getLogger(__name__)


class SkillInstructionService:
    """Service to load and format complete skill instructions directly from database records."""

    def fetch_skills_from_db(
        self,
        db: Session,
        skill_ids: List[str],
        user_id: str,
        max_skills: int = 5,
    ) -> List[Skill]:
        """Fetch published skill records directly from the database table.

        Args:
            db: SQLAlchemy database session
            skill_ids: List of target skill IDs to fetch
            user_id: Current user ID for ownership validation
            max_skills: Maximum number of skills to fetch (default 5)

        Returns:
            List of Skill database model instances belonging to user_id
        """
        if not skill_ids:
            return []

        unique_ids = list(dict.fromkeys(skill_ids))[:max_skills]
        skills = crud.skill.fetch_skills_by_ids(
                db, skill_ids=unique_ids, user_id=user_id
            )

        skill_map = {s.id: s for s in skills}
        ordered_skills = [skill_map[sid] for sid in unique_ids if sid in skill_map]
        return ordered_skills[:max_skills]

    def load_skill_instructions(self, skill: Skill) -> str:
        """Retrieve complete instructions for a skill directly from the database record.

        Args:
            skill: Skill database record model instance

        Returns:
            Instructions string from DB, or empty string if missing
        """
        if not skill:
            return ""

        if skill.instructions and isinstance(skill.instructions, str) and skill.instructions.strip():
            return skill.instructions.strip()

        logger.warning("No instructions available in DB for skill ID: %s (name: %s)", skill.id, getattr(skill, "name", "N/A"))
        return ""

    def load_and_format_instructions(
        self,
        skills: List[Skill],
        max_skills: int = 5,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Load instructions directly from DB skill records and format system context block.

        Args:
            skills: List of Skill database records (from Step 1 or Step 2)
            max_skills: Hard upper bound on applied skills (default 5)

        Returns:
            Tuple of:
            - skills_context: Formatted system prompt instructions string
            - skills_metadata: List of dict metadata for client API response
        """
        if not skills:
            return "", []

        applied_skills = skills[:max_skills]
        formatted_blocks = []
        skills_metadata = []

        for idx, skill in enumerate(applied_skills, 1):
            instructions = self.load_skill_instructions(skill)
            
            meta_item = {
                "id": skill.id,
                "name": skill.name,
                "description": skill.description or "",
                "trigger_description": skill.trigger_description or "",
                "instructions_loaded": bool(instructions),
            }
            skills_metadata.append(meta_item)

            if instructions:
                formatted_blocks.append(
                    f"### Skill {idx}: {skill.name}\n"
                    f"**Description**: {skill.description or 'N/A'}\n"
                    f"**Trigger Condition**: {skill.trigger_description or 'N/A'}\n"
                    f"**Operational Instructions**:\n{instructions}"
                )
            else:
                logger.warning("Skill instruction in DB is empty for skill ID: %s (name: %s)", skill.id, skill.name)
                formatted_blocks.append(
                    f"### Skill {idx}: {skill.name}\n"
                    f"**Description**: {skill.description or 'N/A'}\n"
                    "**Operational Instructions**: [Unavailable/Missing in DB]"
                )

        if not formatted_blocks:
            return "", skills_metadata

        skills_context = (
            "--- ACTIVE USER SKILLS INSTRUCTIONS ---\n"
            "The following operational skills have been selected for this query. "
            "Strictly follow and execute their detailed instructions when generating your response:\n\n"
            + "\n\n".join(formatted_blocks)
            + "\n--- END ACTIVE USER SKILLS ---"
        )

        return skills_context, skills_metadata


def get_skill_instruction_service() -> SkillInstructionService:
    """Return singleton instance of SkillInstructionService."""
    return SkillInstructionService()
