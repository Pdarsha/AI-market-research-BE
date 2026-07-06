from typing import Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from server.crud.base import CRUDBase
from server.models.genai import ChatHistory
from server.schemas.genai import ChatHistoryCreate, ChatHistoryUpdate


class CRUDChatHistory(CRUDBase[ChatHistory, ChatHistoryCreate, ChatHistoryUpdate]):
    def create(self, db: Session, *, obj_in: ChatHistoryCreate) -> ChatHistory:
        return super().create(db, obj_in=obj_in)

    def get_multi(
        self,
        db: Session,
        *,
        user_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ):
        query = db.query(self.model)
        if user_id:
            query = query.filter(self.model.user_id == user_id)
        return (
            query.order_by(desc(self.model.created_at)).offset(skip).limit(limit).all()
        )


chat_history = CRUDChatHistory(ChatHistory)
