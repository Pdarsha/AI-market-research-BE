from typing import Optional, TypeVar

from pydantic import BaseModel
from sqlalchemy.orm import Session

from server.crud.base import CRUDBase
from server.models.workspace import Document
from server.schemas.workspace import CreateDocument, updateDocument

UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDDocument(CRUDBase[Document, CreateDocument, updateDocument]):
    filter_options = ["amount, number of generations"]

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


    def delete_document_cleanup_chats(
        self, db: Session, *, doc_id: str, file_name: str, chats: list
    ) -> None:
        from server.models.genai import ChatMessage
        for chat in chats:
            messages = db.query(ChatMessage).filter(ChatMessage.chat_id == chat.id).all()
            doc_ids = set()
            for m in messages:
                if m.document_ids:
                    doc_ids.update(m.document_ids)
            if doc_id not in doc_ids:
                continue
            if doc_ids == {doc_id}:
                db.delete(chat)  # cascades to its messages
            else:
                for m in messages:
                    modified = False
                    if m.document_ids and doc_id in m.document_ids:
                        m.document_ids = [d for d in m.document_ids if d != doc_id]
                        modified = True
                    if m.source_documents and file_name in m.source_documents:
                        m.source_documents = [
                            n for n in m.source_documents if n != file_name
                        ]
                        modified = True
                    if modified:
                        db.add(m)


document = CRUDDocument(Document)
