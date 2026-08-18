import uuid
from datetime import date, datetime
from typing import Optional, Tuple

from sqlalchemy import desc
from sqlalchemy.exc import IntegrityError
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
        return (
            query.order_by(desc(self.model.updated_at)).offset(skip).limit(limit).all()
        )

    def create(self, db: Session, *, obj_in: ChatSessionCreate) -> ChatSession:
        db_obj = ChatSession(
            id=str(uuid.uuid4()),
            user_id=obj_in.user_id,
            title=obj_in.title,
            workspace_id=obj_in.workspace_id,
            agent_id=obj_in.agent_id,
            agent_name=obj_in.agent_name,
            execution_type=obj_in.execution_type or "chat",
            execution_date=obj_in.execution_date,
            chat_metadata=obj_in.chat_metadata,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def create_with_id(
        self,
        db: Session,
        *,
        chat_id: Optional[str] = None,
        user_id: str,
        title: Optional[str] = None,
        workspace_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        execution_type: Optional[str] = "chat",
        execution_date: Optional[date] = None,
        chat_metadata: Optional[dict] = None,
    ) -> ChatSession:
        """Create a session, honouring a client-supplied ``chat_id`` when given.

        Used by the message/agent endpoints so a chat can be started on the first
        message with the id the client already uses in the URL. When ``chat_id``
        is falsy the model's default UUID is generated instead.
        """
        session = ChatSession(
            user_id=user_id,
            title=title,
            workspace_id=workspace_id,
            agent_id=agent_id,
            agent_name=agent_name,
            execution_type=execution_type or "chat",
            execution_date=execution_date,
            chat_metadata=chat_metadata,
        )
        if chat_id:
            session.id = chat_id
        db.add(session)
        db.commit()
        db.refresh(session)
        return session

    def get_daily_agent_session(
        self,
        db: Session,
        *,
        user_id: str,
        agent_id: str,
        execution_date: date,
    ) -> Optional[ChatSession]:
        """Retrieve an existing agent chat session for the specified user, agent, and calendar day."""
        return (
            db.query(self.model)
            .filter(
                self.model.user_id == str(user_id),
                self.model.agent_id == str(agent_id),
                self.model.execution_type == "agent",
                self.model.execution_date == execution_date,
            )
            .first()
        )

    def get_or_create_daily_agent_session(
        self,
        db: Session,
        *,
        user_id: str,
        agent_id: str,
        agent_name: Optional[str] = None,
        workspace_id: Optional[str] = None,
        execution_date: Optional[date] = None,
        title: Optional[str] = None,
        chat_metadata: Optional[dict] = None,
    ) -> Tuple[ChatSession, bool]:
        """Resolve an existing agent chat session for today or create a new one safely.

        Guarantees:
          - (agent_id, calendar_date) uniqueness per user.
          - Reuses existing chat session for the SAME agent on the SAME calendar day.
          - Creates a new chat session for the SAME agent on a DIFFERENT calendar day.
          - Different agents always get different chat sessions.
          - Concurrency-safe against race conditions via DB uniqueness + rollback/retry.

        Returns:
          Tuple[ChatSession, bool]: (session, created) where created is True if new, False if reused.
        """
        exec_date = execution_date or datetime.utcnow().date()

        # 1. First check if a session for (user_id, agent_id, exec_date) already exists
        existing = self.get_daily_agent_session(
            db, user_id=user_id, agent_id=agent_id, execution_date=exec_date
        )
        if existing:
            # Touch updated_at so it bumps to the top of the sidebar
            existing.updated_at = datetime.utcnow()
            db.add(existing)
            db.commit()
            db.refresh(existing)
            return existing, False

        # 2. Try creating a new daily agent chat session with race-condition protection
        try:
            session = ChatSession(
                id=str(uuid.uuid4()),
                user_id=str(user_id),
                workspace_id=workspace_id,
                title=title,
                agent_id=str(agent_id),
                agent_name=agent_name,
                execution_type="agent",
                execution_date=exec_date,
                chat_metadata=chat_metadata,
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            return session, True
        except IntegrityError:
            db.rollback()
            # In case of concurrent race condition, fetch the session created by the other worker
            existing = self.get_daily_agent_session(
                db, user_id=user_id, agent_id=agent_id, execution_date=exec_date
            )
            if existing:
                return existing, False
            raise


chat_session = CRUDChatSession(ChatSession)
