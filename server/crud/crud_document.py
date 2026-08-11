from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import func

from server.crud.base import CRUDBase
from server.models.workspace import Document, DocumentChatLink
from server.schemas.workspace import CreateDocument, updateDocument


class CRUDDocument(CRUDBase[Document, CreateDocument, updateDocument]):
    def create(self, db: Session, *, obj_in: CreateDocument) -> Document:
        return super().create(db, obj_in=obj_in)

    def update(
        self, db: Session, *, db_obj: Document, obj_in: updateDocument
    ) -> Document:
        return super().update(db, db_obj=db_obj, obj_in=obj_in)

    def remove(self, db: Session, *, id: str) -> Optional[Document]:
        return super().remove(db, id=id)

    def get_document(self, db: Session, doc_id: str, user_id: str):
        return (
            db.query(self.model)
            .filter(self.model.id == doc_id, self.model.user_id == user_id)
            .first()
        )

    def get_multi_by_workspace(self, db: Session, *, workspace_id: str):
        return (
            db.query(self.model)
            .filter(self.model.workspace_id == workspace_id)
            .order_by(self.model.created_at.desc())
            .all()
        )

    def get_chat_count_for_document(self, db: Session, *, doc_id: str) -> int:
        """Return the number of distinct chat sessions that reference *doc_id*.

        Reads from the ``document_chat_links`` table — the single source of
        truth — instead of scanning ``chat_messages.document_ids`` JSON columns.
        """
        return (
            db.query(func.count(func.distinct(DocumentChatLink.chat_id)))
            .filter(DocumentChatLink.document_id == doc_id)
            .scalar()
        ) or 0

    def get_chat_ids_for_document(self, db: Session, *, doc_id: str) -> list[str]:
        """Return the chat IDs of all chats that reference *doc_id*."""
        rows = (
            db.query(DocumentChatLink.chat_id)
            .filter(DocumentChatLink.document_id == doc_id)
            .distinct()
            .all()
        )
        return [r[0] for r in rows if r[0]]

    def delete_document_cleanup_chats(
        self, db: Session, *, doc_id: str, file_name: str, chats: list
    ) -> None:
        from server.models.genai import ChatMessage

        for chat in chats:
            messages = (
                db.query(ChatMessage).filter(ChatMessage.chat_id == chat.id).all()
            )
            for m in messages:
                # Remove housekeeping upload-link rows entirely.
                if (
                    m.role == "system"
                    and m.content
                    and m.content.startswith("[document_upload]")
                    and m.document_ids
                    and doc_id in m.document_ids
                ):
                    db.delete(m)
                    continue

                modified = False
                if m.document_ids and doc_id in m.document_ids:
                    m.document_ids = [d for d in m.document_ids if d != doc_id] or None
                    modified = True
                if m.source_documents and file_name in m.source_documents:
                    m.source_documents = [
                        n for n in m.source_documents if n != file_name
                    ] or None
                    modified = True
                if modified:
                    db.add(m)

        # Also clean up the explicit association table.
        db.query(DocumentChatLink).filter(
            DocumentChatLink.document_id == doc_id
        ).delete(synchronize_session=False)


document = CRUDDocument(Document)
