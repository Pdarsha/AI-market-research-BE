import re
from datetime import datetime
from typing import List, Optional

from sqlalchemy import desc
from sqlalchemy.exc import IntegrityError
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
        status: str = None
    ) -> List[Skill]:
        db_query = db.query(Skill).filter(Skill.user_id == user_id)

        if status:
            db_query = db_query.filter(Skill.status == status)

        return (
            db_query
            .order_by(desc(Skill.updated_at))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_by_user_and_name(
        self, db: Session, *, user_id: str, name: str
    ) -> Optional[Skill]:
        return (
            db.query(Skill)
            .filter(Skill.user_id == user_id, Skill.name == name)
            .first()
        )

    def get_by_user_and_id(
        self, db: Session, *, user_id: str, skill_id: str
    ) -> Optional[Skill]:
        """Get a skill by ID, verifying ownership."""
        return (
            db.query(Skill)
            .filter(Skill.id == skill_id, Skill.user_id == user_id)
            .first()
        )

    def get_unique_skill_name(
        self,
        db: Session,
        *,
        user_id: str,
        base_name: str,
        exclude_skill_id: Optional[str] = None,
        max_attempts: int = 100
    ) -> str:
        """Generate a unique skill name for a given user.
        
        If base_name is not already used by user_id, returns base_name.
        Otherwise appends incrementing integer (2, 3, 4...) until a unique name is found.
        Example:
        'Render Morning Brief' -> 'Render Morning Brief 2' -> 'Render Morning Brief 3'
        """
        base_name = (base_name or "").strip()
        if not base_name:
            base_name = "Untitled Skill"

        # Check if base_name is available
        query = db.query(Skill).filter(
            Skill.user_id == user_id,
            Skill.name == base_name
        )
        if exclude_skill_id:
            query = query.filter(Skill.id != exclude_skill_id)
        if not query.first():
            return base_name

        # Extract root name and starting counter if base_name ends with a number
        match = re.match(r"^(.*?)(?:\s+(\d+))?$", base_name)
        if match and match.group(2):
            root_name = match.group(1).strip()
            counter = int(match.group(2)) + 1
        else:
            root_name = base_name
            counter = 2

        attempts = 0
        while attempts < max_attempts:
            candidate = f"{root_name} {counter}"
            if len(candidate) > 100:
                suffix = f" {counter}"
                candidate = f"{root_name[:100 - len(suffix)]}{suffix}"

            q = db.query(Skill).filter(
                Skill.user_id == user_id,
                Skill.name == candidate
            )
            if exclude_skill_id:
                q = q.filter(Skill.id != exclude_skill_id)
            if not q.first():
                return candidate
            counter += 1
            attempts += 1

        raise ValueError(f"Skill with name '{base_name}' already exists")

    def create_skill(
        self,
        db: Session,
        *,
        user_id: str,
        obj_in: SkillCreate,
        status: str = "draft"
    ) -> Skill:
        """Create a new skill with ownership and status, ensuring name uniqueness and handling race conditions.
        
        Args:
            db: Database session
            user_id: Owner's user ID
            obj_in: Validated skill data (SkillCreate schema)
            status: Initial status (draft)
            
        Returns:
            Created Skill record
            
        Raises:
            ValueError: If unique skill name cannot be created or saved
        """
        candidate_name = obj_in.name
        max_attempts = 5

        for attempt in range(max_attempts):
            unique_name = self.get_unique_skill_name(
                db, user_id=user_id, base_name=candidate_name
            )

            skill_data = obj_in.model_dump()
            skill_data["name"] = unique_name
            skill_data["user_id"] = user_id
            skill_data["status"] = status
            skill_data["version"] = "1.0.0"

            db_data = {k: v for k, v in skill_data.items() if k in Skill.__table__.columns}
            skill = Skill(**db_data)
            try:
                db.add(skill)
                db.commit()
                db.refresh(skill)
                return skill
            except IntegrityError:
                db.rollback()
                if attempt == max_attempts - 1:
                    raise ValueError(f"Skill with name '{candidate_name}' already exists")

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