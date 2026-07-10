from typing import Optional

from pydantic import BaseModel


class GenAIBase(BaseModel):
    user_query: str
    document_ids: Optional[list[str]] = None
    title: Optional[str] = None

    class Config:
        from_attributes = True


class GenAICreate(GenAIBase):
    ai_response: Optional[str] = None


class GenAIUpdate(GenAIBase):
    pass


class ChatSessionBase(BaseModel):
    title: Optional[str] = None

    class Config:
        from_attributes = True


class ChatSessionCreate(ChatSessionBase):
    user_id: Optional[str] = None
    workspace_id: Optional[str] = None


class ChatSessionUpdate(BaseModel):
    title: Optional[str] = None


class ChatSessionTitleUpdate(BaseModel):
    title: str


class ChatMessageBase(BaseModel):
    chat_id: str
    content: str

    class Config:
        from_attributes = True


class ChatMessageRequest(BaseModel):
    content: str
    document_ids: Optional[list[str]] = None
    workspace_id: Optional[str] = None
    title: Optional[str] = None


class CitationMetadata(BaseModel):
    """Citation metadata for a single retrieved chunk.

    Contains all information needed by the Citation View to display and
    highlight the original document.
    """
    id: int
    document_id: str
    document_name: Optional[str] = None
    document_url: Optional[str] = None
    page_number: Optional[int] = None
    chunk_id: str
    highlight_text: str
    section_title: Optional[str] = None
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None

    class Config:
        from_attributes = True


class ChatMessageCreate(ChatMessageBase):
    role: str
    user_id: Optional[str] = None
    workspace_id: Optional[str] = None
    document_ids: Optional[list[str]] = None
    source_documents: Optional[list] = None
    citations: Optional[list[CitationMetadata]] = None


class ChatMessageUpdate(BaseModel):
    content: Optional[str] = None
