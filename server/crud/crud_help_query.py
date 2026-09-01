from typing import List, Optional
from sqlalchemy.orm import Session

from server.crud.base import CRUDBase
from server.models.user import HelpQuery
from server.schemas.settings import HelpQueryCreate


class CRUDHelpQuery(CRUDBase[HelpQuery, HelpQueryCreate, HelpQueryCreate]):
    def get_by_user(self, db: Session, *, user_id: str) -> List[HelpQuery]:
        return (
            db.query(HelpQuery)
            .filter(HelpQuery.user_id == user_id)
            .order_by(HelpQuery.created_at.desc())
            .all()
        )

    def get_by_email(self, db: Session, *, email: str) -> List[HelpQuery]:
        return (
            db.query(HelpQuery)
            .filter(HelpQuery.email == email)
            .order_by(HelpQuery.created_at.desc())
            .all()
        )

    def create_query(
        self,
        db: Session,
        *,
        email: str,
        subject: str,
        description: str,
        user_id: Optional[str] = None,
        name: Optional[str] = None,
        category: str = "general",
    ) -> HelpQuery:
        db_obj = HelpQuery(
            user_id=user_id,
            email=email,
            name=name,
            subject=subject,
            description=description,
            category=category,
            status="pending",
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj


help_query = CRUDHelpQuery(HelpQuery)
