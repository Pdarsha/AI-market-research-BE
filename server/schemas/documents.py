from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class ExportCitation(BaseModel):
    id: Optional[str] = None
    reference_index: Optional[int] = None
    citation_id: Optional[str] = None
    title: Optional[str] = None
    source: Optional[str] = None
    document_name: Optional[str] = None
    url: Optional[str] = None
    link: Optional[str] = None
    navigation_target: Optional[str] = None
    document_url: Optional[str] = None


class ExportAttachment(BaseModel):
    name: Optional[str] = None


class ExportMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = ""
    citations: Optional[List[Dict[str, Any]]] = None
    attachments: Optional[List[ExportAttachment]] = None


class ConversationExportRequest(BaseModel):
    title: Optional[str] = None
    messages: List[ExportMessage]
