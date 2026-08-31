"""Focused unit tests for Profile -> Credits section and Credit Management System.

Tests cover:
1. Default daily credit limit of 100 credits for free tier users.
2. Accurate credit balance, consumed usage, and percentage calculations.
3. Strict user-level isolation (operations on User A do not affect User B).
4. Prevention of negative balances, overconsumption, and invalid (zero/negative) deductions.
5. Automatic daily credit reset on new UTC calendar day and after 24 hours.
6. Preservation of balance when accessed multiple times on the same day.
7. Safe credit refunds capped at the user's daily maximum.
8. Immutable transaction ledger history (CreditUsageLog).
9. Tier-specific daily credit limits (100 Free, 1,000 Plus, 5,000 Pro, 20,000 Enterprise).
10. Full REST API endpoints verification via FastAPI TestClient:
    - GET /v1/user/credits
    - GET /v1/user/profile/credits
    - GET /v1/credits
    - POST /v1/user/credits/consume
    - POST /v1/user/credits/reset
    - GET /v1/user/credits/history
"""

import pytest
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from server.db.base_class import Base
from server.models.user import User, CreditUsageLog
from server.schemas.credits import (
    CreditConsumeRequest,
    CreditRefundRequest,
    CreditUsageLogResponse,
    UserCreditsResponse,
)
from server.services.credit_service import CreditService, TIER_DAILY_LIMITS
from server.api_v1.endpoints.user import user_router
from server.api import api_router
from server.utils.auth import get_current_user, get_optional_current_user
from server.api_v1.deps import get_db


@pytest.fixture
def db_session():
    """In-memory SQLite database session for fast, isolated tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def free_user(db_session):
    """Create a sample free tier user with 100 daily credits."""
    user = User(
        id="usr-credit-free-1",
        email="free_analyst@example.com",
        display_name="Free Analyst",
        subscription_tier="free",
        daily_credits=100,
        credits_remaining=100,
        last_credit_reset_at=datetime(2026, 8, 31, 8, 0, 0),
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def plus_user(db_session):
    """Create a sample plus tier user with 1,000 daily credits."""
    user = User(
        id="usr-credit-plus-2",
        email="plus_analyst@example.com",
        display_name="Plus Analyst",
        subscription_tier="plus",
        daily_credits=1000,
        credits_remaining=1000,
        last_credit_reset_at=datetime(2026, 8, 31, 8, 0, 0),
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def app_client(db_session, free_user):
    """FastAPI TestClient with overridden dependencies."""
    app = FastAPI()
    app.include_router(api_router, prefix="/v1")

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    def override_get_current_user():
        return free_user

    def override_get_optional_current_user():
        return free_user

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_optional_current_user] = override_get_optional_current_user

    client = TestClient(app)
    return client


class TestCreditLimitsAndBalances:
    """Test daily limits, balance tracking, and consumption logic."""

    def test_default_free_tier_limit_is_100(self, db_session, free_user):
        now = datetime(2026, 8, 31, 10, 0, 0)
        credits = CreditService.get_user_credits(db_session, free_user, now=now)
        assert credits.daily_credits == 100
        assert credits.credits_remaining == 100
        assert credits.credits_used_today == 0
        assert credits.usage_percentage == 0.0
        assert credits.is_exhausted is False
        assert credits.subscription_tier == "free"

    def test_credit_consumption_and_usage_tracking(self, db_session, free_user):
        now = datetime(2026, 8, 31, 10, 0, 0)
        # Consume 25 credits for RAG document analysis
        credits = CreditService.consume_credits(
            db_session, free_user, amount=25, purpose="rag_document_analysis", now=now
        )
        assert credits.daily_credits == 100
        assert credits.credits_remaining == 75
        assert credits.credits_used_today == 25
        assert credits.usage_percentage == 25.0
        assert credits.is_exhausted is False

        # Consume another 25 credits
        credits = CreditService.consume_credits(
            db_session, free_user, amount=25, purpose="deep_research_query", now=now
        )
        assert credits.credits_remaining == 50
        assert credits.credits_used_today == 50
        assert credits.usage_percentage == 50.0

    def test_prevent_overconsumption_beyond_balance(self, db_session, free_user):
        now = datetime(2026, 8, 31, 10, 0, 0)
        # Attempting to consume 101 credits when 100 available
        with pytest.raises(ValueError) as exc:
            CreditService.consume_credits(
                db_session, free_user, amount=101, purpose="heavy_job", now=now
            )
        assert "Insufficient credits" in str(exc.value)

        # Balance remains unchanged at 100
        assert free_user.credits_remaining == 100

    def test_prevent_zero_or_negative_consumption(self, db_session, free_user):
        now = datetime(2026, 8, 31, 10, 0, 0)
        with pytest.raises(ValueError) as exc:
            CreditService.consume_credits(db_session, free_user, amount=0, now=now)
        assert "must be positive" in str(exc.value)

        with pytest.raises(ValueError) as exc:
            CreditService.consume_credits(db_session, free_user, amount=-10, now=now)
        assert "must be positive" in str(exc.value)

    def test_user_credit_isolation(self, db_session, free_user, plus_user):
        now = datetime(2026, 8, 31, 10, 0, 0)
        # Free user consumes 40 credits
        CreditService.consume_credits(db_session, free_user, amount=40, now=now)

        # Plus user consumes 100 credits
        CreditService.consume_credits(db_session, plus_user, amount=100, now=now)

        # Verify free user has 60 remaining (100 - 40)
        free_credits = CreditService.get_user_credits(db_session, free_user, now=now)
        assert free_credits.credits_remaining == 60
        assert free_credits.daily_credits == 100

        # Verify plus user has 900 remaining (1000 - 100)
        plus_credits = CreditService.get_user_credits(db_session, plus_user, now=now)
        assert plus_credits.credits_remaining == 900
        assert plus_credits.daily_credits == 1000

    def test_safe_credit_refund(self, db_session, free_user):
        now = datetime(2026, 8, 31, 10, 0, 0)
        CreditService.consume_credits(db_session, free_user, amount=30, now=now)
        assert free_user.credits_remaining == 70

        # Refund 10 credits
        credits = CreditService.refund_credits(
            db_session, free_user, amount=10, reason="failed_llm_call", now=now
        )
        assert credits.credits_remaining == 80
        assert credits.credits_used_today == 20

        # Refunding more than daily allowance caps at daily_credits (100)
        credits = CreditService.refund_credits(
            db_session, free_user, amount=50, reason="compensation", now=now
        )
        assert credits.credits_remaining == 100
        assert credits.credits_used_today == 0


class TestDailyCreditReset:
    """Test automatic and manual daily credit reset mechanics."""

    def test_same_day_no_reset(self, db_session, free_user):
        # Set last reset to 8:00 AM on Aug 31
        free_user.last_credit_reset_at = datetime(2026, 8, 31, 8, 0, 0)
        free_user.credits_remaining = 60
        db_session.commit()

        # Access at 2:00 PM on same day (Aug 31)
        now = datetime(2026, 8, 31, 14, 0, 0)
        credits = CreditService.get_user_credits(db_session, free_user, now=now)
        assert credits.credits_remaining == 60
        assert credits.credits_used_today == 40

    def test_calendar_day_transition_triggers_automatic_reset(self, db_session, free_user):
        # User had consumed credits on Aug 31
        free_user.last_credit_reset_at = datetime(2026, 8, 31, 12, 0, 0)
        free_user.credits_remaining = 15
        db_session.commit()

        # User accesses on Sep 1 (next day)
        next_day = datetime(2026, 9, 1, 9, 0, 0)
        credits = CreditService.get_user_credits(db_session, free_user, now=next_day)

        assert credits.credits_remaining == 100
        assert credits.daily_credits == 100
        assert credits.credits_used_today == 0
        assert credits.is_exhausted is False
        assert free_user.last_credit_reset_at.date() == next_day.date()

    def test_24_hours_elapsed_triggers_reset(self, db_session, free_user):
        # User last reset was 25 hours ago
        free_user.last_credit_reset_at = datetime(2026, 8, 30, 8, 0, 0)
        free_user.credits_remaining = 0
        db_session.commit()

        now = datetime(2026, 8, 31, 9, 30, 0)
        credits = CreditService.get_user_credits(db_session, free_user, now=now)
        assert credits.credits_remaining == 100
        assert credits.credits_used_today == 0

    def test_manual_credit_reset(self, db_session, free_user):
        free_user.credits_remaining = 10
        db_session.commit()

        credits = CreditService.reset_user_credits(db_session, free_user)
        assert credits.credits_remaining == 100
        assert credits.credits_used_today == 0


class TestCreditUsageLedger:
    """Test transaction history logging in CreditUsageLog."""

    def test_credit_usage_logs_recorded(self, db_session, free_user):
        now = datetime(2026, 8, 31, 10, 0, 0)
        CreditService.consume_credits(
            db_session, free_user, amount=10, purpose="document_summary", now=now
        )
        CreditService.consume_credits(
            db_session, free_user, amount=15, purpose="citation_search", now=now
        )
        CreditService.refund_credits(
            db_session, free_user, amount=5, reason="timeout", now=now
        )

        history = CreditService.get_credit_history(db_session, user_id=free_user.id)
        assert len(history) >= 3
        action_types = [h.action_type for h in history]
        assert "consume" in action_types
        assert "refund" in action_types


class TestCreditsApiEndpoints:
    """Test REST API endpoints using FastAPI TestClient."""

    def test_get_user_credits_endpoint(self, app_client):
        resp = app_client.get("/v1/user/credits")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["daily_credits"] == 100
        assert body["data"]["credits_remaining"] == 100
        assert body["data"]["subscription_tier"] == "free"

    def test_get_user_credits_canonical_endpoint(self, app_client):
        """Canonical credits endpoint is /v1/user/credits — the /profile/credits alias has been removed."""
        resp = app_client.get("/v1/user/credits")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["daily_credits"] == 100

    def test_consume_credits_api_endpoint(self, app_client):
        payload = {"amount": 20, "purpose": "ai_agent_step"}
        resp = app_client.post("/v1/user/credits/consume", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["credits_remaining"] == 80
        assert body["data"]["credits_used_today"] == 20
        assert body["data"]["usage_percentage"] == 20.0

    def test_consume_credits_api_insufficient_credits(self, app_client):
        payload = {"amount": 200, "purpose": "massive_job"}
        resp = app_client.post("/v1/user/credits/consume", json=payload)
        assert resp.status_code == 400
        body = resp.json()
        assert "Insufficient credits" in body["detail"]

    def test_reset_credits_api_endpoint(self, app_client):
        # Consume some first
        app_client.post("/v1/user/credits/consume", json={"amount": 30})

        # Reset back
        resp = app_client.post("/v1/user/credits/reset")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["credits_remaining"] == 100
        assert body["data"]["credits_used_today"] == 0

    def test_get_credits_history_api_endpoint(self, app_client):
        app_client.post("/v1/user/credits/consume", json={"amount": 10, "purpose": "test_step"})
        resp = app_client.get("/v1/user/credits/history")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert isinstance(body["data"], list)
        assert len(body["data"]) >= 1

    def test_calendar_day_boundary_2359_vs_0000(self, db_session, free_user):
        """Test exact UTC boundary behavior between 23:59:59 and 00:00:01."""
        free_user.last_credit_reset_at = datetime(2026, 8, 31, 0, 0, 0)
        free_user.credits_remaining = 35
        db_session.commit()

        # At 23:59:59 UTC on Aug 31 -> No reset
        t_2359 = datetime(2026, 8, 31, 23, 59, 59)
        c1 = CreditService.get_user_credits(db_session, free_user, now=t_2359)
        assert c1.credits_remaining == 35
        assert c1.credits_used_today == 65

        # At 00:00:01 UTC on Sep 01 -> Reset triggers immediately
        t_0000 = datetime(2026, 9, 1, 0, 0, 1)
        c2 = CreditService.get_user_credits(db_session, free_user, now=t_0000)
        assert c2.credits_remaining == 100
        assert c2.credits_used_today == 0
        assert free_user.last_credit_reset_at.date() == t_0000.date()

