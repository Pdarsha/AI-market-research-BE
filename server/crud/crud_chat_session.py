from typing import Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from server.crud.base import CRUDBase
from server.models.genai import ChatSession
from server.schemas.genai import ChatSessionCreate, ChatSessionUpdate


class CRUDChatSession(CRUDBase[ChatSession, ChatSessionCreate, ChatSessionUpdate]):
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
        return query.order_by(desc(self.model.updated_at)).offset(skip).limit(limit).all()

    def create(self, db: Session, *, obj_in: ChatSessionCreate) -> ChatSession:
        return super().create(db, obj_in=obj_in)

    def create_with_id(
        self,
        db: Session,
        *,
        chat_id: Optional[str] = None,
        user_id: str,
        title: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> ChatSession:
        """Create a session, honouring a client-supplied ``chat_id`` when given.

        Used by the message/agent endpoints so a chat can be started on the first
        message with the id the client already uses in the URL. When ``chat_id``
        is falsy the model's default UUID is generated instead.
        """
        session = ChatSession(
            user_id=user_id, title=title, workspace_id=workspace_id
        )
        if chat_id:
            session.id = chat_id
        db.add(session)
        db.commit()
        db.refresh(session)
        return session


chat_session = CRUDChatSession(ChatSession)
