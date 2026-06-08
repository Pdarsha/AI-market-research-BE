import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr


class EmailOtpBase(BaseModel):
    id: Optional[uuid.UUID] = None
    otp: Optional[str] = None
    email: Optional[EmailStr] = None
    user_id: Optional[str] = None
    attempts: Optional[int] = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class EmailOtpCreate(EmailOtpBase):
    pass


class EmailOtpUpdate(EmailOtpBase):
    pass
