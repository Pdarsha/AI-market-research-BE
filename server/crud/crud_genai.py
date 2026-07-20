from typing import Optional, TypeVar

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy.orm import Session

from server.crud.base import CRUDBase
from server.models.genai import ChatSession, GenAI
from server.schemas.genai import GenAICreate, GenAIUpdate

UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDGENAI(CRUDBase[GenAI, GenAICreate, GenAIUpdate]):
    filter_options = ["amount, number of generations"]

    def create(self, db: Session, *, obj_in: GenAICreate) -> GenAI:
        obj_in_data = jsonable_encoder(obj_in)
        if "document_ids" in obj_in_data:
            obj_in_data["document_id"] = obj_in_data.pop("document_ids")
        db_obj = self.model(**obj_in_data)  # type: ignore
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(self, db: Session, *, db_obj: GenAI, obj_in: GenAIUpdate) -> GenAI:
        obj_data = jsonable_encoder(db_obj)
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.model_dump(exclude_unset=True)
        if "document_ids" in update_data:
            update_data["document_id"] = update_data.pop("document_ids")
        for field in obj_data:
            if field in update_data:
                setattr(db_obj, field, update_data[field])
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def remove(self, db: Session, *, id: str) -> Optional[GenAI]:
        return super().remove(db, id=id)

    def get_sessions_by_workspace_id(
        self, db: Session, *, workspace_id: str, user_id: str, chat_id: str
    ) -> list[GenAI]:
        return (
            db.query(ChatSession)
            .filter(
                ChatSession.workspace_id == workspace_id,
                ChatSession.user_id == user_id,
                ChatSession.id != chat_id,
            )
            .order_by(ChatSession.updated_at.desc())
            .limit(5)
            .all()
        )


genai = CRUDGENAI(GenAI)
