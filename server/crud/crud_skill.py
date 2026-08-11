from datetime import datetime
from typing import List, Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from server.crud.base import CRUDBase
from server.models.skill import Skill, SkillStatus
from server.schemas.skill import SkillCreate, SkillUpdate


class CRUDSkill(CRUDBase[Skill, SkillCreate, SkillUpdate]):

    def get_multi_by_user(
        self,
        db: Session,
        *,
        user_id: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Skill]:
        return (
            db.query(self.model)
            .filter(self.model.user_id == user_id)
            .order_by(desc(self.model.updated_at))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_by_user_and_name(
        self, db: Session, *, user_id: str, name: str
    ) -> Optional[Skill]:
        return (
            db.query(self.model)
            .filter(self.model.user_id == user_id, self.model.name == name)
            .first()
        )

    def get_by_user_and_id(
        self, db: Session, *, user_id: str, skill_id: str
    ) -> Optional[Skill]:
        """Get a skill by ID, verifying ownership."""
        return (
            db.query(self.model)
            .filter(self.model.id == skill_id, self.model.user_id == user_id)
            .first()
        )

    def create_skill(
        self,
        db: Session,
        *,
        user_id: str,
        obj_in: SkillCreate,
        status: str = "draft"
    ) -> Skill:
        """Create a new skill with ownership and status.
        
        Args:
            db: Database session
            user_id: Owner's user ID
            obj_in: Validated skill data (SkillCreate schema)
            status: Initial status (draft)
            
        Returns:
            Created Skill record
            
        Raises:
            ValueError: If skill with same name already exists for user
        """
        # Check for duplicate name
        existing = self.get_by_user_and_name(db, user_id=user_id, name=obj_in.name)
        if existing:
            raise ValueError(f"Skill with name '{obj_in.name}' already exists")

        # Create skill object with user_id and status
        skill_data = obj_in.model_dump()
        skill_data["user_id"] = user_id
        skill_data["status"] = status
        skill_data["version"] = "1.0.0"
        
        skill = self.model(**skill_data)
        db.add(skill)
        db.commit()
        db.refresh(skill)
        return skill

    def publish_skill(
        self,
        db: Session,
        *,
        skill_id: str,
        user_id: str,
    ) -> Optional[Skill]:
        """Publish a skill (change status to published).
        
        Args:
            db: Database session
            skill_id: ID of the skill to publish
            user_id: Owner's user ID (for ownership verification)
            
        Returns:
            Updated Skill record or None if not found/not owned
            
        Raises:
            ValueError: If skill is not in a valid state for publishing
        """
        skill = self.get_by_user_and_id(db, user_id=user_id, skill_id=skill_id)
        
        if not skill:
            return None
        
        # Validate state transition - only draft can be published
        if skill.status != SkillStatus.DRAFT.value:
            raise ValueError(f"Cannot publish skill with status '{skill.status}'. Must be draft.")
        
        skill.status = SkillStatus.PUBLISHED.value
        skill.published_at = datetime.utcnow()
        # Increment version for publish (major version bump)
        try:
            version_parts = skill.version.split(".")
            if len(version_parts) >= 1:
                major = int(version_parts[0]) + 1
                skill.version = f"{major}.0.0"
        except (ValueError, IndexError):
            skill.version = "2.0.0"
        
        db.commit()
        db.refresh(skill)
        return skill


skill = CRUDSkill(Skill)