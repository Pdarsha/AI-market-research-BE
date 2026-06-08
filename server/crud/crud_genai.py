from typing import Optional, TypeVar

from pydantic import BaseModel
from sqlalchemy.orm import Session

from server.crud.base import CRUDBase
from server.models.genai import GenAI
from server.schemas.genai import GenAICreate, GenAIUpdate

UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDGENAI(CRUDBase[GenAI, GenAICreate, GenAIUpdate]):
    filter_options = ["amount, number of generations"]

    def create(self, db: Session, *, obj_in: GenAICreate) -> GenAI:
        return super().create(db, obj_in=obj_in)

    def update(self, db: Session, *, db_obj: GenAI, obj_in: GenAIUpdate) -> GenAI:
        return super().update(db, db_obj=db_obj, obj_in=obj_in)

    def remove(self, db: Session, *, id: str) -> Optional[GenAI]:
        return super().remove(db, id=id)


genai = CRUDGENAI(GenAI)
