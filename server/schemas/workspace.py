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

class DocumentChunkBase(BaseModel):
    document_id: str
    user_id: Optional[str] = None
    workspace_id: Optional[str] = None

    file_name: Optional[str] = None
    chunk_number: Optional[int] = None
    total_chunks: Optional[int] = None

    content: Optional[str] = None

    # Citation metadata
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None

    page_number: Optional[int] = None
    section_title: Optional[str] = None

    document_path: Optional[str] = None
    document_url: Optional[str] = None

class DocumentChunkCreate(DocumentChunkBase):
    embedding: Optional[list[float]] = None


class DocumentChunkUpdate(BaseModel):
    file_name: Optional[str] = None
    chunk_number: Optional[int] = None
    total_chunks: Optional[int] = None

    content: Optional[str] = None
    embedding: Optional[list[float]] = None

    start_offset: Optional[int] = None
    end_offset: Optional[int] = None

    page_number: Optional[int] = None
    section_title: Optional[str] = None

    document_path: Optional[str] = None
    document_url: Optional[str] = None


class DocumentChatLinkBase(BaseModel):
    document_id: str
    chat_id: str
    workspace_id: Optional[str] = None
    user_id: Optional[str] = None

    class Config:
        from_attributes = True


class DocumentChatLinkCreate(DocumentChatLinkBase):
    pass


class DocumentChatLinkUpdate(BaseModel):
    pass


