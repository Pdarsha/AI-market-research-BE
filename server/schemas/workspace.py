from typing import Optional

from pydantic import BaseModel


class WorkspaceBase(BaseModel):
    name: str
    description: Optional[str] = None

    class Config:
        from_attributes = True


class WorkspaceCreate(WorkspaceBase):
    user_id: Optional[str] = None


class WorkspaceUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class WorkspaceResponse(WorkspaceBase):
    id: str
    user_id: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        from_attributes = True
