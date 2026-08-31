from datetime import datetime
from typing import List, Optional
from sqlalchemy.orm import Session

from server.crud.base import CRUDBase
from server.models.user import CreditUsageLog
from server.schemas.credits import CreditUsageLogResponse


class CRUDCreditUsageLog(CRUDBase[CreditUsageLog, CreditUsageLogResponse, CreditUsageLogResponse]):

    def add(
        self,
        db: Session,
        *,
        user_id: str,
        amount: int,
        action_type: str,
        description: str,
        balance_after: int,
        created_at: Optional[datetime] = None,
    ) -> None:
        """Stage a CreditUsageLog ledger entry in the session without committing.

        The caller (service layer) controls the transaction boundary and commit.
        Errors are silently swallowed to prevent a ledger write failure from
        blocking the primary credit operation.
        """
        try:
            log_entry = CreditUsageLog(
                user_id=user_id,
                amount=amount,
                action_type=action_type,
                description=description,
                balance_after=balance_after,
                created_at=created_at or datetime.utcnow(),
            )
            db.add(log_entry)
        except Exception:
            pass

    def get_by_user(
        self, db: Session, *, user_id: str, limit: int = 50
    ) -> List[CreditUsageLog]:
        """Return recent credit ledger entries for a user, ordered newest-first."""
        return (
            db.query(CreditUsageLog)
            .filter(CreditUsageLog.user_id == user_id)
            .order_by(CreditUsageLog.created_at.desc())
            .limit(limit)
            .all()
        )


credit_usage_log = CRUDCreditUsageLog(CreditUsageLog)
