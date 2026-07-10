from typing import List, Optional
from pydantic import BaseModel

class ExportCitation(BaseModel):
    title: Optional[str] = None
    source: Optional[str] = None
    document_name: Optional[str] = None
    url: Optional[str] = None


class ExportAttachment(BaseModel):
    name: Optional[str] = None


class ExportMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = ""
    citations: Optional[List[ExportCitation]] = None
    attachments: Optional[List[ExportAttachment]] = None


class ConversationExportRequest(BaseModel):
    title: Optional[str] = None
    messages: List[ExportMessage]
