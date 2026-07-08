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


class DocumentBase(BaseModel):
    text: str

    class Config:
        from_attributes = True


class CreateDocument(DocumentBase):
    pass


class updateDocument(DocumentBase):
    pass


class DocumentResponse(BaseModel):
    user_id: str
    workspace_id: Optional[str] = None
    file_name: str
    file_format: str
    storage_path: Optional[str] = None
