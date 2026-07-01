import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import relationship

from server.db.base_class import Base


class Workspace(Base):
    id = Column(String, default=lambda: str(uuid.uuid4()), primary_key=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User")
    # Deleting a workspace cascades to its chats (and their messages) and to its
    # uploaded files. DB-side ondelete=CASCADE + passive_deletes avoids loading rows.
    chats = relationship(
        "ChatSession",
        back_populates="workspace",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    files = relationship(
        "Embedding",
        back_populates="workspace",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
