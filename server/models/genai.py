import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, JSON, String, Text
from sqlalchemy.orm import relationship

from server.db.base_class import Base


class GenAI(Base):
    id = Column(String, default=lambda: str(uuid.uuid4()), primary_key=True)
    email = Column(String, unique=True, nullable=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    user_query = Column(Text)
    ai_response = Column(Text)
    document_id = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User")


class ChatSession(Base):
    id = Column(String, default=lambda: str(uuid.uuid4()), primary_key=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    workspace_id = Column(
        String,
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        index=True,
        nullable=True,
    )
    title = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User")
    workspace = relationship("Workspace", back_populates="chats")
    messages = relationship(
        "ChatMessage",
        back_populates="chat",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ChatMessage(Base):
    id = Column(String, default=lambda: str(uuid.uuid4()), primary_key=True)
    chat_id = Column(String, ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)

    # Chat metadata previously stored on the (now-retired) GenAI table. Keeping it
    # here makes chat_messages the single source of truth for a conversation.
    workspace_id = Column(
        String,
        ForeignKey("workspaces.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    document_ids = Column(JSON, nullable=True)  # document ids referenced by this message
    source_documents = Column(JSON, nullable=True)  # e.g. file names used to ground the answer

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    chat = relationship("ChatSession", back_populates="messages")
    user = relationship("User")


