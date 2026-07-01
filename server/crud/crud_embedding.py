from typing import Optional, TypeVar

from pydantic import BaseModel
from sqlalchemy.orm import Session

from server.crud.base import CRUDBase
from server.models.genai import Embedding
from server.schemas.genai import CreateEmbedding, updateEmbedding

UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDEMBEDDING(CRUDBase[Embedding, CreateEmbedding, updateEmbedding]):
    filter_options = ["amount, number of generations"]

    def create(self, db: Session, *, obj_in: CreateEmbedding) -> Embedding:
        return super().create(db, obj_in=obj_in)

    def update(
        self, db: Session, *, db_obj: Embedding, obj_in: updateEmbedding
    ) -> Embedding:
        return super().update(db, db_obj=db_obj, obj_in=obj_in)

    def remove(self, db: Session, *, id: str) -> Optional[Embedding]:
        return super().remove(db, id=id)

    def get_multi_by_workspace(self, db: Session, *, workspace_id: str):
        return (
            db.query(self.model)
            .filter(self.model.workspace_id == workspace_id)
            .order_by(self.model.created_at.desc())
            .all()
        )


enbedding = CRUDEMBEDDING(Embedding)
