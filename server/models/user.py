import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from server.db.base_class import Base


class User(Base):
    id = Column(String, default=lambda: str(uuid.uuid4()), primary_key=True)
    avatar = Column(String, default="")
    email = Column(String, unique=True, nullable=True)
    is_verified = Column(Boolean, default=False)
    disabled = Column(Boolean, default=False)
    providers = Column(String)
    hashed_password = Column(String)
    role = Column(String, default=0)

    google_id = Column(String, unique=True, nullable=True, index=True)
    display_name = Column(String, nullable=True)

    subscription_tier = Column(String, default="free", nullable=True)
    daily_credits = Column(Integer, default=100, nullable=True)
    credits_remaining = Column(Integer, default=100, nullable=True)
    last_credit_reset_at = Column(DateTime, default=datetime.utcnow, nullable=True)
    subscription_status = Column(String, default="active", nullable=True)
    subscription_period = Column(String, default="monthly", nullable=True)

    # User Profile & App Settings
    display_language = Column(String, default="English", nullable=True)
    nickname = Column(String, nullable=True)
    occupation = Column(String, nullable=True)
    bio = Column(String, nullable=True)
    ai_traits = Column(String, nullable=True)
    custom_instructions = Column(String, nullable=True)
    ai_data_retention = Column(Boolean, default=True, nullable=True)
    settings_json = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class EmailOtp(Base):
    id = Column(String, default=lambda: str(uuid.uuid4()), primary_key=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    email = Column(String)
    otp = Column(String)
    attempts = Column(Integer)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User")


class HelpQuery(Base):
    __tablename__ = "help_queries"

    id = Column(String, default=lambda: str(uuid.uuid4()), primary_key=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    email = Column(String, nullable=False)
    name = Column(String, nullable=True)
    subject = Column(String, nullable=False)
    description = Column(String, nullable=False)
    category = Column(String, default="general", nullable=True)
    status = Column(String, default="pending", nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User")


class CreditUsageLog(Base):
    __tablename__ = "credit_usage_logs"

    id = Column(String, default=lambda: str(uuid.uuid4()), primary_key=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    amount = Column(Integer, nullable=False)
    action_type = Column(String(50), default="consume", nullable=False)  # consume, reset, refund, bonus
    description = Column(String(255), nullable=True)
    balance_after = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")


