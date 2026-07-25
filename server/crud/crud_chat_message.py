from datetime import datetime
from typing import Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from server.crud.base import CRUDBase
from server.models.genai import ChatMessage
from server.schemas.genai import ChatMessageCreate, ChatMessageUpdate
from server.utils.source_attribution import cited_document_names


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

    def get_visible_by_chat(
        self,
        db: Session,
        *,
        chat_id: str,
        skip: int = 0,
        limit: int = 1000,
    ):
        return [
            m
            for m in self.get_multi_by_chat(db, chat_id=chat_id, skip=skip, limit=limit)
            if not (
                m.role == "system"
                and (m.content or "").startswith("[document_upload]")
            )
        ]

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
            .filter(self.model.chat_id == chat_id)
            .order_by(desc(self.model.created_at))
            .first()
        )
        return cited_document_names(latest.citations) if latest else None

    def count_user_daily_calls(self, db: Session, *, user_id: str, mode_prefix: str, since: datetime) -> int:
        """Count the number of daily calls for a user."""
        return (
            db.query(ChatMessage).filter(
                ChatMessage.user_id == str(user_id),
                ChatMessage.mode_used.startswith(mode_prefix),
                ChatMessage.created_at >= since
            ).count()
        )


chat_message = CRUDChatMessage(ChatMessage)
