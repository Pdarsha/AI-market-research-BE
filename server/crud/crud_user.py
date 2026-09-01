from typing import List, Optional
from sqlalchemy.orm import Session

from server.crud.base import CRUDBase
from server.models.user import User
from server.schemas.user import UserCreate, UserUpdate


class CRUDUSER(CRUDBase[User, UserCreate, UserUpdate]):
    def get_all(self, db: Session) -> List[User]:
        return db.query(User).all()

    def get_by_id(self, db: Session, *, id: str) -> Optional[User]:
        return db.query(User).filter(User.id == id).first()

    def get_by_email(self, db: Session, *, email: str) -> Optional[User]:
        return db.query(User).filter(User.email == email).first()

    def get_by_google_id(self, db: Session, *, google_id: str) -> Optional[User]:
        return db.query(User).filter(User.google_id == google_id).first()

    def create(self, db: Session, *, obj_in: UserCreate) -> User:
        return super().create(db, obj_in=obj_in)

    def update(self, db: Session, *, db_obj: User, obj_in: UserUpdate) -> User:
        return super().update(db, db_obj=db_obj, obj_in=obj_in)

    def remove(self, db: Session, *, id: str) -> Optional[User]:
        return super().remove(db, id=id)

    def get_with_lock(self, db: Session, *, id: str) -> Optional[User]:
        """Fetch a user row with an exclusive row-level lock (for credit concurrency safety)."""
        try:
            return (
                db.query(User)
                .filter(User.id == id)
                .with_for_update(nowait=False)
                .first()
            )
        except Exception:
            return db.query(User).filter(User.id == id).first()

    def save(self, db: Session, *, db_obj: User, commit: bool = True) -> User:
        """Persist unsaved changes on a User instance."""
        db.add(db_obj)
        if commit:
            db.commit()
            db.refresh(db_obj)
        return db_obj


user = CRUDUSER(User)
