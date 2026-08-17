import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import Column, DateTime, Enum, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship
from server.utils.markdown import render_rich_text

from server.db.base_class import Base


class SkillStatus(str, PyEnum):
    DRAFT = "draft"
    PUBLISHED = "published"


class Skill(Base):
    id = Column(String, default=lambda: str(uuid.uuid4()), primary_key=True)
    user_id = Column(
        String, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    trigger_description = Column(Text, nullable=True)
    instructions = Column(Text, nullable=True)
    status = Column(
        Enum(SkillStatus, native_enum=False, values_callable=lambda x: [e.value for e in x]),
        default=SkillStatus.DRAFT,
        nullable=False,
    )
    version = Column(String, default="1.0.0", nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    published_at = Column(DateTime, nullable=True)

    user = relationship("User")

    @property
    def description_html(self) -> str:
        return render_rich_text(self.description) if self.description else ""

    @property
    def trigger_description_html(self) -> str:
        return render_rich_text(self.trigger_description) if self.trigger_description else ""

    @property
    def instructions_html(self) -> str:
        return render_rich_text(self.instructions) if self.instructions else ""

    __table_args__ = (
        Index("ix_skills_user_id", "user_id"),
        Index("ix_skills_status", "status"),
        UniqueConstraint("user_id", "name", name="uix_skills_user_id_name"),
    )