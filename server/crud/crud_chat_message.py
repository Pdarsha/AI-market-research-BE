from typing import Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from server.crud.base import CRUDBase
from server.models.genai import ChatMessage
from server.schemas.genai import ChatMessageCreate, ChatMessageUpdate


class CRUDChatMessage(CRUDBase[ChatMessage, ChatMessageCreate, ChatMessageUpdate]):
    def create(self, db, *, obj_in: ChatMessageCreate) -> ChatMessage:
        return super().create(db, obj_in=obj_in)

    def get_multi_by_chat(
        self,
        db: Session,
        *,
        chat_id: str,
        skip: int = 0,
        limit: int = 1000,
    ):
        """Return all messages for a chat, oldest first."""
        return (
            db.query(self.model)
            .filter(self.model.chat_id == chat_id)
            .order_by(self.model.created_at.asc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_latest_document_ids(self, db: Session, *, chat_id: str) -> Optional[list]:
        """Most recent ``document_ids`` referenced within a chat.

        Reads exclusively from ``chat_messages`` (previously sourced from the
        retired GenAI table).
        """
        latest = (
            db.query(self.model)
            .filter(
                self.model.chat_id == chat_id,
                self.model.document_ids.isnot(None),
            )
            .order_by(desc(self.model.created_at))
            .first()
        )
        if latest and latest.document_ids:
            return latest.document_ids
        return None

    def get_latest_source_documents(
        self, db: Session, *, chat_id: str, document_ids: Optional[list]
    ) -> Optional[list]:
        """Most recent ``source_documents`` referenced within a chat.

        Reads exclusively from ``chat_messages`` (previously sourced from the
        retired GenAI table).
        """
        latest = (
            db.query(self.model)
            .filter(
                self.model.chat_id == chat_id,
                self.model.source_documents.isnot(None),
            )
            .order_by(desc(self.model.created_at))
            .first()
        )
        if latest and latest.source_documents:
            return latest.source_documents
        return None


chat_message = CRUDChatMessage(ChatMessage)
