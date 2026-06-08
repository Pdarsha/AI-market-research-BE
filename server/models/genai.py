import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import relationship

from server.db.base_class import Base


class GenAI(Base):
    id = Column(String, default=lambda: str(uuid.uuid4()), primary_key=True)
    email = Column(String, unique=True, nullable=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    user_query = Column(Text)
    ai_response = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User")


class Embedding(Base):
    id = Column(String, default=lambda: str(uuid.uuid4()), primary_key=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    file_name = Column(String)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User")
