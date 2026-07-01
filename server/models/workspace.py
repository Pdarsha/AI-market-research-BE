import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import relationship, Mapped, mapped_column

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
    docs = relationship(
        "Document",
        back_populates="workspace",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Document(Base):
    id = Column(String, default=lambda: str(uuid.uuid4()), primary_key=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    # A file always belongs to a workspace (enforced at the API layer). Nullable
    # at the DB level only so legacy rows created before this refactor survive.
    workspace_id = Column(
        String,
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        index=True,
        nullable=True,
    )
    file_name = Column(String)
    file_format = Column(String)
    storage_path: Mapped[str] = mapped_column(String(500), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User")
    workspace = relationship("Workspace", back_populates="docs")