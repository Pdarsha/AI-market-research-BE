"""User Credit & Token Management Service.

Provides complete handling for:
1. Enforcing daily credit limits per user (100 credits/day default for free tier, isolated per user).
2. Credit balance tracking and safe deduction / consumption without negative numbers.
3. Automatic daily reset of credits based on UTC calendar day and 24-hour cycle.
4. User-level isolation with audit logging and immutable transaction ledger (CreditUsageLog).
5. Fast credit retrieval with remaining balance, usage today, percentage, and countdown to reset.
"""

from datetime import datetime, timedelta, timezone
import logging
from typing import Dict, List, Optional
from sqlalchemy.orm import Session

from server.models.user import User
from server.schemas.credits import (
    CreditUsageLogResponse,
    UserCreditsResponse,
)
from server.services.audit_logger import AuditLogger
from server import crud

logger = logging.getLogger(__name__)

# Daily credit limits dynamically sourced from PricingService canonical definitions
def _get_canonical_tier_limits() -> Dict[str, int]:
    try:
        from server.services.pricing_service import PLANS_BY_ID
        return {k: v["daily_credits"] for k, v in PLANS_BY_ID.items()}
    except Exception:
        return {"free": 100, "plus": 1000, "pro": 5000, "enterprise": 20000}


TIER_DAILY_LIMITS = _get_canonical_tier_limits()


class CreditService:
    """Service providing credit allocation, deduction, balance verification, and daily resets."""

    @classmethod
    def get_tier_daily_limit(cls, tier: Optional[str]) -> int:
        """Return the official daily credit quota for the specified tier."""
        tier_key = (tier or "free").strip().lower()
        limits = _get_canonical_tier_limits()
        return limits.get(tier_key, 100)

    @classmethod
    def check_and_apply_daily_reset(
        cls, db: Session, user: User, now: Optional[datetime] = None
    ) -> bool:
        """Check if the user's daily credit allowance should be reset and apply if due."""
        current_time = now or datetime.now(timezone.utc)
        # Ensure current_time is timezone-naive for comparison with naive DB datetimes if needed
        compare_now = (
            current_time.replace(tzinfo=None)
            if hasattr(current_time, "tzinfo") and current_time.tzinfo
            else current_time
        )

        # Lock user row for concurrency-safe check — delegated to CRUD
        locked_user = crud.user.get_with_lock(db, id=user.id)
        if locked_user:
            user = locked_user

        last_reset = user.last_credit_reset_at
        if last_reset is not None and hasattr(last_reset, "tzinfo") and last_reset.tzinfo:
            last_reset = last_reset.replace(tzinfo=None)

        should_reset = False
        if last_reset is None:
            should_reset = True
        elif compare_now.date() > last_reset.date():
            should_reset = True

        if should_reset:
            daily_limit = user.daily_credits or cls.get_tier_daily_limit(user.subscription_tier)
            user.daily_credits = daily_limit
            user.credits_remaining = daily_limit
            user.last_credit_reset_at = compare_now

            # Record in transaction ledger via CRUD (no commit yet — service commits below)
            crud.credit_usage_log.add(
                db,
                user_id=user.id,
                amount=daily_limit,
                action_type="reset",
                description=f"Daily credit reset ({daily_limit} credits for {user.subscription_tier or 'free'} tier)",
                balance_after=daily_limit,
                created_at=compare_now,
            )

            # Persist user + log in a single commit at service boundary
            crud.user.save(db, db_obj=user, commit=True)

            logger.info(
                "[CREDITS] Reset %s credits for user_id=%s tier=%s",
                daily_limit,
                user.id,
                user.subscription_tier,
            )
            AuditLogger.log(
                action="CREDITS_DAILY_RESET",
                user_id=str(user.id),
                status="success",
                metadata={"daily_credits": daily_limit, "tier": user.subscription_tier},
            )
            return True

        return False


    @classmethod
    def get_user_credits(
        cls, db: Session, user: User, now: Optional[datetime] = None
    ) -> UserCreditsResponse:
        """Retrieve the verified credit status, usage, and reset window for a user."""
        cls.check_and_apply_daily_reset(db, user, now=now)

        daily_credits = user.daily_credits or cls.get_tier_daily_limit(user.subscription_tier)
        if user.credits_remaining is None:
            user.credits_remaining = daily_credits
            crud.user.save(db, db_obj=user, commit=True)

        # Safeguard: prevent invalid or out-of-bound credits
        remaining = max(0, min(daily_credits, user.credits_remaining))
        used_today = max(0, daily_credits - remaining)
        usage_pct = round((used_today / daily_credits) * 100, 1) if daily_credits > 0 else 0.0

        current_time = now or datetime.now(timezone.utc)
        compare_now = (
            current_time.replace(tzinfo=None)
            if hasattr(current_time, "tzinfo") and current_time.tzinfo
            else current_time
        )

        last_reset = user.last_credit_reset_at or compare_now
        if hasattr(last_reset, "tzinfo") and last_reset.tzinfo:
            last_reset = last_reset.replace(tzinfo=None)

        # Next reset is midnight of tomorrow UTC or +24h
        next_midnight = (compare_now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        hours_until_reset = max(1, int((next_midnight - compare_now).total_seconds() // 3600))
        if hours_until_reset > 24:
            hours_until_reset = 24

        return UserCreditsResponse(
            daily_credits=daily_credits,
            credits_remaining=remaining,
            credits_used_today=used_today,
            subscription_tier=user.subscription_tier or "free",
            reset_in_hours=hours_until_reset,
            last_reset_at=last_reset,
            next_reset_at=next_midnight,
            usage_percentage=usage_pct,
            is_exhausted=(remaining <= 0),
        )

    @classmethod
    def consume_credits(
        cls,
        db: Session,
        user: User,
        amount: int = 1,
        purpose: str = "ai_generation",
        now: Optional[datetime] = None,
    ) -> UserCreditsResponse:
        """Deduct credits from the user's daily quota safely with validation."""
        if amount <= 0:
            raise ValueError(f"Credit amount must be positive, received {amount}.")

        # Row-level lock on user record for concurrency safety — delegated to CRUD
        locked_user = crud.user.get_with_lock(db, id=user.id)
        if locked_user:
            user = locked_user

        cls.check_and_apply_daily_reset(db, user, now=now)

        daily_credits = user.daily_credits or cls.get_tier_daily_limit(user.subscription_tier)
        current_remaining = user.credits_remaining
        if current_remaining is None:
            current_remaining = daily_credits

        if current_remaining < amount:
            raise ValueError(
                f"Insufficient credits: {current_remaining} available, but {amount} required. Please upgrade your plan or wait for daily reset."
            )

        new_balance = max(0, current_remaining - amount)
        user.credits_remaining = new_balance

        current_time = now or datetime.now(timezone.utc)
        compare_now = (
            current_time.replace(tzinfo=None)
            if hasattr(current_time, "tzinfo") and current_time.tzinfo
            else current_time
        )

        # Record ledger entry via CRUD (no separate commit — committed below with user)
        crud.credit_usage_log.add(
            db,
            user_id=user.id,
            amount=-amount,
            action_type="consume",
            description=purpose,
            balance_after=new_balance,
            created_at=compare_now,
        )

        crud.user.save(db, db_obj=user, commit=True)

        logger.info(
            "[CREDITS] Consumed %s credits for user_id=%s (remaining: %s, purpose: %s)",
            amount,
            user.id,
            new_balance,
            purpose,
        )
        AuditLogger.log(
            action="CREDITS_CONSUMED",
            user_id=str(user.id),
            status="success",
            metadata={"amount": amount, "balance_after": new_balance, "purpose": purpose},
        )

        return cls.get_user_credits(db, user, now=now)

    @classmethod
    def refund_credits(
        cls,
        db: Session,
        user: User,
        amount: int = 1,
        reason: str = "failed_request",
        now: Optional[datetime] = None,
    ) -> UserCreditsResponse:
        """Refund credits back to the user without exceeding their daily allowance."""
        if amount <= 0:
            raise ValueError(f"Refund amount must be positive, received {amount}.")

        cls.check_and_apply_daily_reset(db, user, now=now)

        daily_credits = user.daily_credits or cls.get_tier_daily_limit(user.subscription_tier)
        current_remaining = user.credits_remaining or 0
        new_balance = min(daily_credits, current_remaining + amount)
        user.credits_remaining = new_balance

        current_time = now or datetime.now(timezone.utc)
        compare_now = (
            current_time.replace(tzinfo=None)
            if hasattr(current_time, "tzinfo") and current_time.tzinfo
            else current_time
        )

        crud.credit_usage_log.add(
            db,
            user_id=user.id,
            amount=amount,
            action_type="refund",
            description=reason,
            balance_after=new_balance,
            created_at=compare_now,
        )

        crud.user.save(db, db_obj=user, commit=True)

        logger.info(
            "[CREDITS] Refunded %s credits to user_id=%s (new balance: %s, reason: %s)",
            amount,
            user.id,
            new_balance,
            reason,
        )
        AuditLogger.log(
            action="CREDITS_REFUNDED",
            user_id=str(user.id),
            status="success",
            metadata={"amount": amount, "balance_after": new_balance, "reason": reason},
        )

        return cls.get_user_credits(db, user, now=now)

    @classmethod
    def reset_user_credits(
        cls, db: Session, user: User, now: Optional[datetime] = None, force: bool = True
    ) -> UserCreditsResponse:
        """Reset a user's credit balance to full quota if due, or forcibly if admin/testing."""
        if not force:
            cls.check_and_apply_daily_reset(db, user, now=now)
            return cls.get_user_credits(db, user, now=now)

        daily_credits = user.daily_credits or cls.get_tier_daily_limit(user.subscription_tier)
        user.credits_remaining = daily_credits

        current_time = now or datetime.now(timezone.utc)
        compare_now = (
            current_time.replace(tzinfo=None)
            if hasattr(current_time, "tzinfo") and current_time.tzinfo
            else current_time
        )
        user.last_credit_reset_at = compare_now

        crud.credit_usage_log.add(
            db,
            user_id=user.id,
            amount=daily_credits,
            action_type="reset",
            description=f"Manual/Admin credit reset ({daily_credits} credits)",
            balance_after=daily_credits,
            created_at=compare_now,
        )

        crud.user.save(db, db_obj=user, commit=True)

        return cls.get_user_credits(db, user, now=now)


    @classmethod
    def get_credit_history(
        cls, db: Session, user_id: str, limit: int = 50
    ) -> List[CreditUsageLogResponse]:
        """Retrieve recent credit transaction history for the user."""
        logs = crud.credit_usage_log.get_by_user(db, user_id=user_id, limit=limit)
        return [CreditUsageLogResponse.model_validate(l) for l in logs]
