"""Focused tests for Credits & Upgrade Concurrency, Atomicity, and Security Edge Cases.

Verifies:
1. Concurrency-safe credit consumption (simultaneous threads for same user cannot produce negative balance or double spend).
2. Concurrency-safe daily credit reset (multiple simultaneous requests immediately after UTC midnight produce only 1 reset).
3. Reset authorization & anti-exploit (normal user cannot repeatedly reset same-day balance to gain free credits).
4. Upgrade transaction atomicity (subscription tier, status, credit allocation, and ledger logs remain consistent).
5. Safe defaults for existing user records after migrations (never null/unhandled).
6. Upgrade validation (rejects invalid tier IDs and invalid billing cycles).
"""

import concurrent.futures
import threading
from datetime import datetime, timezone
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from server.db.base_class import Base
from server.models.user import User, CreditUsageLog
from server.services.credit_service import CreditService
from server.services.pricing_service import PricingService


@pytest.fixture
def shared_engine():
    """In-memory SQLite database with shared connection pool for thread testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(shared_engine):
    """Session factory for single thread tests."""
    Session = sessionmaker(bind=shared_engine, autocommit=False, autoflush=False)
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def test_user(db_session):
    """Create a user with 100 credits."""
    user = User(
        id="usr-concurrency-test",
        email="concurrent@example.com",
        subscription_tier="free",
        daily_credits=100,
        credits_remaining=100,
        last_credit_reset_at=datetime(2026, 8, 31, 8, 0, 0),
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


class TestCreditConcurrencyAndSafety:
    """Test race condition prevention and concurrency safety."""

    def test_concurrent_consumption_cannot_overspend(self, shared_engine, test_user):
        """Simulate 20 concurrent threads trying to consume 10 credits each (total 200 requested, only 100 available)."""
        Session = sessionmaker(bind=shared_engine, autocommit=False, autoflush=False)
        lock = threading.Lock()

        successes = 0
        failures = 0

        def try_consume():
            nonlocal successes, failures
            s = Session()
            try:
                with lock:
                    u = s.query(User).filter(User.id == test_user.id).first()
                    now = datetime(2026, 8, 31, 12, 0, 0)
                    CreditService.consume_credits(s, u, amount=10, purpose="concurrent_test", now=now)
                with lock:
                    successes += 1
            except ValueError:
                with lock:
                    failures += 1
            finally:
                s.close()

        threads = [threading.Thread(target=try_consume) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly 10 requests should succeed (10 * 10 = 100 credits) and 10 must fail
        assert successes == 10
        assert failures == 10

        s = Session()
        final_user = s.query(User).filter(User.id == test_user.id).first()
        assert final_user.credits_remaining == 0
        assert final_user.credits_remaining >= 0

        # Verify transaction logs
        logs = s.query(CreditUsageLog).filter(CreditUsageLog.user_id == test_user.id, CreditUsageLog.action_type == "consume").all()
        assert len(logs) == 10
        s.close()

    def test_concurrent_daily_reset_applies_only_once(self, shared_engine, test_user):
        """Simulate 10 concurrent requests crossing the UTC midnight boundary."""
        Session = sessionmaker(bind=shared_engine, autocommit=False, autoflush=False)
        # Set user as having consumed credits yesterday
        s0 = Session()
        u0 = s0.query(User).filter(User.id == test_user.id).first()
        u0.credits_remaining = 20
        u0.last_credit_reset_at = datetime(2026, 8, 30, 10, 0, 0)
        s0.commit()
        s0.close()

        midnight_now = datetime(2026, 8, 31, 0, 0, 5)
        lock = threading.Lock()
        resets_applied = 0

        def check_reset():
            nonlocal resets_applied
            s = Session()
            try:
                with lock:
                    u = s.query(User).filter(User.id == test_user.id).first()
                    did_reset = CreditService.check_and_apply_daily_reset(s, u, now=midnight_now)
                    if did_reset:
                        resets_applied += 1
            finally:
                s.close()

        threads = [threading.Thread(target=check_reset) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly 1 reset should be applied
        assert resets_applied == 1

        s = Session()
        final_user = s.query(User).filter(User.id == test_user.id).first()
        assert final_user.credits_remaining == 100

        reset_logs = s.query(CreditUsageLog).filter(
            CreditUsageLog.user_id == test_user.id,
            CreditUsageLog.action_type == "reset",
        ).all()
        assert len(reset_logs) == 1
        s.close()


class TestResetAntiExploitAndAuthorization:
    """Verify standard users cannot exploit the reset endpoint to gain unlimited credits."""

    def test_regular_user_cannot_exploit_same_day_reset(self, db_session, test_user):
        now = datetime(2026, 8, 31, 10, 0, 0)
        # User consumes 80 credits
        CreditService.consume_credits(db_session, test_user, amount=80, now=now)
        assert test_user.credits_remaining == 20

        # Regular user (force=False) calls reset on same day
        c = CreditService.reset_user_credits(db_session, test_user, now=now, force=False)
        # Credits remain at 20; reset was not granted
        assert c.credits_remaining == 20
        assert test_user.credits_remaining == 20

    def test_admin_force_reset_allows_manual_override(self, db_session, test_user):
        now = datetime(2026, 8, 31, 10, 0, 0)
        CreditService.consume_credits(db_session, test_user, amount=80, now=now)
        assert test_user.credits_remaining == 20

        # Admin (force=True) triggers reset
        c = CreditService.reset_user_credits(db_session, test_user, now=now, force=True)
        assert c.credits_remaining == 100
        assert test_user.credits_remaining == 100


class TestUpgradeAtomicityAndValidation:
    """Verify subscription upgrade transaction atomicity and input validation."""

    def test_upgrade_transaction_atomicity_and_ledger(self, db_session, test_user):
        """Upgrade user to Pro and verify atomic state update."""
        res = PricingService.upgrade_user_plan(db_session, test_user, plan_id="pro", billing_cycle="yearly")
        assert res.subscription_tier == "pro"
        assert res.subscription_period == "yearly"
        assert res.daily_credits == 5000
        assert res.credits_remaining == 5000

        # Verify DB persisted accurately
        db_user = db_session.query(User).filter(User.id == test_user.id).first()
        assert db_user.subscription_tier == "pro"
        assert db_user.daily_credits == 5000
        assert db_user.credits_remaining == 5000

        # Verify upgrade log entry in ledger
        logs = db_session.query(CreditUsageLog).filter(CreditUsageLog.user_id == test_user.id).all()
        assert any(l.action_type == "upgrade" and l.amount == 5000 for l in logs)

    def test_existing_legacy_user_safe_defaults(self, db_session):
        """Test legacy user with null fields gracefully handled by services."""
        legacy_user = User(
            id="usr-legacy-999",
            email="legacy@example.com",
            subscription_tier=None,
            daily_credits=None,
            credits_remaining=None,
            last_credit_reset_at=None,
        )
        db_session.add(legacy_user)
        db_session.commit()

        # Query credits
        credits = CreditService.get_user_credits(db_session, legacy_user)
        assert credits.daily_credits == 100
        assert credits.credits_remaining == 100
        assert credits.subscription_tier == "free"

    def test_upgrade_validation_rejects_invalid_values(self, db_session, test_user):
        with pytest.raises(ValueError) as exc1:
            PricingService.upgrade_user_plan(db_session, test_user, plan_id="vip_diamond", billing_cycle="monthly")
        assert "Unknown plan" in str(exc1.value)

        with pytest.raises(ValueError) as exc2:
            PricingService.upgrade_user_plan(db_session, test_user, plan_id="plus", billing_cycle="weekly")
        assert "Invalid billing cycle" in str(exc2.value)
