from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr


class UserBase(BaseModel):
    avatar: Optional[str] = None
    email: Optional[EmailStr] = None
    is_verified: Optional[bool] = False
    role: Optional[str] = "member"
    providers: Optional[str] = None
    hashed_password: Optional[str] = None
    google_id: Optional[str] = None
    display_name: Optional[str] = None

    class Config:
        from_attributes = True


class UserCreate(UserBase):
    pass


class UserUpdate(UserBase):
    pass


class Token(BaseModel):
    access_token: str
    token_type: str
    refresh_token: Optional[str] = None


class RespUser(BaseModel):
    id: Optional[str] = None
    email: Optional[EmailStr] = None
    avatar: Optional[str] = None
    display_name: Optional[str] = None
    is_verified: Optional[bool] = None
    providers: Optional[str] = None
    role: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    token: Optional[Token]


class User(BaseModel):
    hashed_password: Optional[str] = None
    email: Optional[EmailStr] = None


class UserCreateInput(BaseModel):
    email: EmailStr
    password: str
    display_name: Optional[str] = None


class VerifyOTP(BaseModel):
    email: EmailStr
    otp: str


class UserInDB(User):
    hashed_password: str


class UserIn(User):
    password: str


class IdTokenLoginInput(BaseModel):
    id_token: str


class IdTokenValidteLoginInput(BaseModel):
    email: EmailStr


class IdTokenRegisterInput(BaseModel):
    id_token: str


class EmailSchema(BaseModel):
    email: EmailStr


class UserProfile(BaseModel):
    id: Optional[str] = None
    email: Optional[EmailStr] = None
    avatar: Optional[str] = None
    role: Optional[str] = None
    display_name: Optional[str] = None
    username: Optional[str] = None


class ChangePassword(BaseModel):
    old_password: str
    new_password: str


class ResetPassword(BaseModel):
    email: EmailStr
    otp: str
    new_password: str


class RefreshTokenSchema(BaseModel):
    refresh_token: str
