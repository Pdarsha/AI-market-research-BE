from typing import Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from server.crud.base import CRUDBase
from server.models.genai import ChatSession
from server.models.workspace import Workspace
from server.schemas.workspace import WorkspaceCreate, WorkspaceUpdate


class CRUDWorkspace(CRUDBase[Workspace, WorkspaceCreate, WorkspaceUpdate]):
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

    def create(self, db: Session, *, obj_in: WorkspaceCreate) -> Workspace:
        return super().create(db, obj_in=obj_in)

    def update(
        self, db: Session, *, db_obj: Workspace, obj_in: WorkspaceUpdate
    ) -> Workspace:
        return super().update(db, db_obj=db_obj, obj_in=obj_in)

    def remove(self, db: Session, *, id: str) -> Optional[Workspace]:
        return super().remove(db, id=id)

    def _get_owned_workspace(
        self, db: Session, workspace_id: str, user_id: str
    ) -> Optional[Workspace]:
        return (
            db.query(self.model)
            .filter(self.model.id == workspace_id, self.model.user_id == user_id)
            .first()
        )

    def delete_workspace(
        self,
        db: Session,
        *,
        workspace: Workspace,
    ) -> None:
        """
        Delete a workspace and everything it contains.

        The DB-level ``ON DELETE CASCADE`` on ``chat_sessions.workspace_id``,
        ``documents.workspace_id`` and ``document_chunks.workspace_id`` removes the
        workspace's chats (and their messages), its file rows and their pgvector
        embeddings in a single relational cascade.
        """
        db.delete(workspace)
        db.commit()

    def get_sessions_by_workspace_id(
        self, db: Session, *, workspace_id: str, user_id: str
    ) -> list[Workspace]:
        return (
            db.query(ChatSession)
            .filter(
                ChatSession.workspace_id == workspace_id,
                ChatSession.user_id == user_id,
            )
            .all()
        )


workspace = CRUDWorkspace(Workspace)
