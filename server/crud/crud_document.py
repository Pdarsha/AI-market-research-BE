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

    def get_document(self, db: Session, workspace_id:str, doc_id: str, user_id: str):
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


document = CRUDDocument(Document)
