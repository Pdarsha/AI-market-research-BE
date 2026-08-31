"""Focused unit tests for Profile -> Upgrade Pricing Data and Subscription Management.

Tests cover:
1. Retrieval of tier-wise pricing specifications and plan details (Free, Plus, Pro, Enterprise).
2. Calculation of annual billing discounts and monthly equivalents.
3. Daily credit limits (100 for Free, 1,000 for Plus, 5,000 for Pro, 20,000 for Enterprise).
4. Identification of current active user tier and personalized `is_current` indicators.
5. User subscription retrieval, credits balance, and daily usage tracking.
6. User subscription upgrade flow with database persistence and credit adjustments.
7. Input validation for plan IDs and billing cycles (handling invalid requests safely).
8. Full API endpoint verification via FastAPI TestClient for /v1/user/plans, /v1/user/pricing,
   /v1/plans, /v1/pricing, /v1/user/subscription, /v1/user/subscription/upgrade, and /v1/user/credits.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from server.db.base_class import Base
from server.models.user import User
from server.schemas.subscription import (
    PlanPrice,
    PlanTier,
    PricingPlansResponse,
    UpgradePlanRequest,
    UserCreditsResponse,
    UserSubscriptionResponse,
)
from server.services.pricing_service import PricingService, PLAN_DEFINITIONS, PLANS_BY_ID
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
def sample_user(db_session):
    """Create a sample user with default free tier subscription."""
    user = User(
        id="usr-test-12345",
        email="analyst@example.com",
        display_name="Test Analyst",
        subscription_tier="free",
        daily_credits=100,
        credits_remaining=100,
        subscription_status="active",
        subscription_period="monthly",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def app_client(db_session, sample_user):
    """FastAPI TestClient with overridden dependencies."""
    app = FastAPI()
    app.include_router(api_router, prefix="/v1")

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    def override_get_current_user():
        return sample_user

    def override_get_optional_current_user():
        return sample_user

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_optional_current_user] = override_get_optional_current_user

    client = TestClient(app)
    return client


class TestPricingTierDefinitions:
    """Validate core plan definitions and pricing data integrity."""

    def test_all_tiers_present(self):
        plan_ids = [p["id"] for p in PLAN_DEFINITIONS]
        assert "free" in plan_ids
        assert "plus" in plan_ids
        assert "pro" in plan_ids
        assert "enterprise" in plan_ids
        assert len(plan_ids) == 4

    def test_free_tier_specifications(self):
        free = PLANS_BY_ID["free"]
        assert free["name"] == "Free"
        assert free["price"]["monthly"] == 0.0
        assert free["price"]["yearly"] == 0.0
        assert free["daily_credits"] == 100
        assert free["limits"]["workspaces"] == 3
        assert free["is_popular"] is False
        assert any("100 daily AI credits" in f for f in free["features"])
        assert any("Workspaces" in f for f in free["features"])

    def test_plus_tier_specifications(self):
        plus = PLANS_BY_ID["plus"]
        assert plus["name"] == "Plus"
        assert plus["price"]["monthly"] == 20.0
        assert plus["price"]["yearly"] == 192.0
        assert plus["price"]["yearly_monthly_equivalent"] == 16.0
        assert plus["daily_credits"] == 1000
        assert plus["badge"] == "Most Popular"
        assert plus["is_popular"] is True
        assert any("1,000 daily AI credits" in f for f in plus["features"])
        assert any("Deep Research" in f for f in plus["features"])
        assert any("Scheduled Agent Workflow" in f for f in plus["features"])

    def test_pro_tier_specifications(self):
        pro = PLANS_BY_ID["pro"]
        assert pro["name"] == "Pro"
        assert pro["price"]["monthly"] == 50.0
        assert pro["price"]["yearly"] == 480.0
        assert pro["price"]["yearly_monthly_equivalent"] == 40.0
        assert pro["daily_credits"] == 5000
        assert pro["badge"] == "Professional"
        assert any("5,000 daily AI credits" in f for f in pro["features"])
        assert any("SEC EDGAR" in f for f in pro["features"])
        assert any("Custom Skill Generation" in f for f in pro["features"])

    def test_enterprise_tier_specifications(self):
        ent = PLANS_BY_ID["enterprise"]
        assert ent["name"] == "Enterprise"
        assert ent["price"]["monthly"] == 200.0
        assert ent["daily_credits"] == 20000
        assert ent["badge"] == "Enterprise"
        assert any("Single-Tenant" in f for f in ent["features"])
        assert any("SSO" in f for f in ent["features"])


class TestPricingServiceLogic:
    """Test PricingService business logic, discounts, and upgrade workflows."""

    def test_get_all_plans_without_user(self):
        plans = PricingService.get_all_plans()
        assert len(plans) == 4
        # Default user tier is free, so free is current
        assert plans[0].id == "free"
        assert plans[0].is_current is True
        assert plans[1].id == "plus"
        assert plans[1].is_current is False

    def test_get_all_plans_with_plus_user(self):
        plans = PricingService.get_all_plans(current_user_tier="plus")
        plus_plan = next(p for p in plans if p.id == "plus")
        free_plan = next(p for p in plans if p.id == "free")
        assert plus_plan.is_current is True
        assert plus_plan.button_text == "Current Plan"
        assert plus_plan.button_variant == "outline"
        assert free_plan.is_current is False

    def test_get_pricing_overview(self):
        overview = PricingService.get_pricing_overview(current_user_tier="free")
        assert isinstance(overview, PricingPlansResponse)
        assert len(overview.plans) == 4
        assert overview.current_tier == "free"
        assert overview.currency == "USD"
        assert overview.discount_yearly_percent == 20
        assert "monthly" in overview.billing_cycles
        assert "yearly" in overview.billing_cycles

    def test_get_user_subscription(self, sample_user):
        sub = PricingService.get_user_subscription(sample_user)
        assert isinstance(sub, UserSubscriptionResponse)
        assert sub.email == "analyst@example.com"
        assert sub.subscription_tier == "free"
        assert sub.daily_credits == 100
        assert sub.credits_remaining == 100
        assert sub.credits_used_today == 0
        assert sub.current_plan.name == "Free"

    def test_get_user_credits(self, sample_user):
        credits = PricingService.get_user_credits(sample_user)
        assert isinstance(credits, UserCreditsResponse)
        assert credits.daily_credits == 100
        assert credits.credits_remaining == 100
        assert credits.credits_used_today == 0
        assert credits.subscription_tier == "free"
        assert credits.reset_in_hours == 24

    def test_upgrade_user_plan_to_plus_monthly(self, db_session, sample_user):
        updated = PricingService.upgrade_user_plan(
            db=db_session,
            user=sample_user,
            plan_id="plus",
            billing_cycle="monthly",
        )
        assert updated.subscription_tier == "plus"
        assert updated.subscription_period == "monthly"
        assert updated.daily_credits == 1000
        assert updated.credits_remaining == 1000
        assert sample_user.subscription_tier == "plus"
        assert sample_user.daily_credits == 1000

    def test_upgrade_user_plan_to_pro_yearly(self, db_session, sample_user):
        updated = PricingService.upgrade_user_plan(
            db=db_session,
            user=sample_user,
            plan_id="pro",
            billing_cycle="yearly",
        )
        assert updated.subscription_tier == "pro"
        assert updated.subscription_period == "yearly"
        assert updated.daily_credits == 5000
        assert updated.credits_remaining == 5000

    def test_upgrade_user_plan_invalid_id(self, db_session, sample_user):
        with pytest.raises(ValueError) as exc:
            PricingService.upgrade_user_plan(
                db=db_session,
                user=sample_user,
                plan_id="nonexistent_plan",
            )
        assert "Unknown plan" in str(exc.value)

    def test_upgrade_user_plan_invalid_cycle(self, db_session, sample_user):
        with pytest.raises(ValueError) as exc:
            PricingService.upgrade_user_plan(
                db=db_session,
                user=sample_user,
                plan_id="plus",
                billing_cycle="weekly",
            )
        assert "Invalid billing cycle" in str(exc.value)


class TestPricingSchemaValidation:
    """Validate Pydantic schema validation constraints and edge cases."""

    def test_upgrade_request_validation_success(self):
        req = UpgradePlanRequest(plan_id="Plus", billing_cycle="YEARLY")
        assert req.plan_id == "plus"
        assert req.billing_cycle == "yearly"

    def test_upgrade_request_validation_invalid_plan(self):
        with pytest.raises(ValueError):
            UpgradePlanRequest(plan_id="diamond")

    def test_upgrade_request_validation_invalid_cycle(self):
        with pytest.raises(ValueError):
            UpgradePlanRequest(plan_id="plus", billing_cycle="biweekly")


class TestPricingApiEndpoints:
    """Test API routes using FastAPI TestClient."""

    def test_get_user_plans_endpoint(self, app_client):
        resp = app_client.get("/v1/user/plans")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert "data" in body
        assert len(body["data"]["plans"]) == 4
        assert body["data"]["currency"] == "USD"
        assert body["data"]["discount_yearly_percent"] == 20

    def test_get_user_plans_canonical_endpoint(self, app_client):
        """Canonical pricing endpoint is /v1/user/plans. Aliases /user/pricing, /v1/plans, /v1/pricing have been removed."""
        resp = app_client.get("/v1/user/plans")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert len(body["data"]["plans"]) == 4

    def test_get_user_subscription_endpoint(self, app_client):
        resp = app_client.get("/v1/user/subscription")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["subscription_tier"] == "free"
        assert body["data"]["daily_credits"] == 100
        assert "current_plan" in body["data"]

    def test_get_user_credits_endpoint(self, app_client):
        resp = app_client.get("/v1/user/credits")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["daily_credits"] == 100
        assert body["data"]["credits_remaining"] == 100
        assert body["data"]["subscription_tier"] == "free"

    def test_upgrade_subscription_endpoint_success(self, app_client):
        payload = {"plan_id": "plus", "billing_cycle": "yearly"}
        resp = app_client.post("/v1/user/subscription/upgrade", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["subscription_tier"] == "plus"
        assert body["data"]["subscription_period"] == "yearly"
        assert body["data"]["daily_credits"] == 1000

    def test_upgrade_subscription_endpoint_invalid_plan(self, app_client):
        payload = {"plan_id": "super_vip", "billing_cycle": "monthly"}
        resp = app_client.post("/v1/user/subscription/upgrade", json=payload)
        assert resp.status_code == 422  # Pydantic validation failure

    def test_upgrade_subscription_endpoint_invalid_cycle(self, app_client):
        payload = {"plan_id": "pro", "billing_cycle": "hourly"}
        resp = app_client.post("/v1/user/subscription/upgrade", json=payload)
        assert resp.status_code == 422  # Pydantic validation failure

    def test_upgrade_cross_user_isolation_and_ledger_recording(self, db_session, sample_user):
        """Verify upgrade updates ONLY target user and logs transaction."""
        user_two = User(
            id="usr-other-999",
            email="other@example.com",
            subscription_tier="free",
            daily_credits=100,
            credits_remaining=100,
        )
        db_session.add(user_two)
        db_session.commit()

        # Upgrade sample_user
        PricingService.upgrade_user_plan(db_session, sample_user, plan_id="enterprise", billing_cycle="yearly")

        # Verify sample_user upgraded
        assert sample_user.subscription_tier == "enterprise"
        assert sample_user.daily_credits == 20000
        assert sample_user.credits_remaining == 20000

        # Verify user_two unchanged
        assert user_two.subscription_tier == "free"
        assert user_two.daily_credits == 100
        assert user_two.credits_remaining == 100

        # Verify credit transaction ledger
        from server.models.user import CreditUsageLog
        logs = db_session.query(CreditUsageLog).filter(CreditUsageLog.user_id == sample_user.id).all()
        assert any(l.action_type == "upgrade" and l.amount == 20000 for l in logs)

