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


class AIResponse(BaseModel):
    user_query: str
    response: str


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


class ChatSessionResponse(ChatSessionBase):
    id: str
    user_id: str
    workspace_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        from_attributes = True


class ChatMessageBase(BaseModel):
    chat_id: str
    content: str

    class Config:
        from_attributes = True


class ChatMessageRequest(BaseModel):
    content: str
    document_ids: Optional[list[str]] = None


class ChatMessageCreate(ChatMessageBase):
    role: str
    user_id: Optional[str] = None
    workspace_id: Optional[str] = None
    document_ids: Optional[list[str]] = None
    source_documents: Optional[list] = None


class ChatMessageUpdate(BaseModel):
    content: Optional[str] = None


class ChatMessageResponse(ChatMessageBase):
    id: str
    role: str
    workspace_id: Optional[str] = None
    document_ids: Optional[list[str]] = None
    source_documents: Optional[list] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        from_attributes = True


class ChatSessionDetail(BaseModel):
    id: str
    title: Optional[str] = None
    workspace_id: Optional[str] = None
    document_ids: Optional[list[str]] = None
    messages: list[ChatMessageResponse] = []

    class Config:
        from_attributes = True


