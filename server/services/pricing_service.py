"""Pricing and Subscription Tier Management Service.

Provides tier-wise pricing specifications, feature matrices, usage limits,
and subscription upgrade management for the CapitalEdge platform.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from server.models.user import User
from server.schemas.subscription import (
    PlanPrice,
    PlanTier,
    PricingPlansResponse,
    UserCreditsResponse,
    UserSubscriptionResponse,
)
from server import crud

logger = logging.getLogger(__name__)

# Source of truth for all subscription plan tiers and pricing structures.
PLAN_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "id": "free",
        "name": "Free",
        "tagline": "For individual researchers exploring AI-powered workflows",
        "badge": None,
        "is_popular": False,
        "price": {
            "monthly": 0.0,
            "yearly": 0.0,
            "yearly_monthly_equivalent": 0.0,
            "currency": "USD",
            "currency_symbol": "$",
        },
        "daily_credits": 100,
        "features": [
            "100 daily AI credits",
            "Up to 3 active Workspaces",
            "Standard Document Upload (up to 25 MB/file)",
            "RAG Chat & Semantic Document Search",
            "Section-level Source Citations & PDF Excerpts",
            "Export Results to PDF & Excel",
            "Community Support",
        ],
        "limitations": [
            "Limited daily credits (100/day)",
            "Standard execution priority",
            "No custom skill publishing",
            "No financial MCP connectors",
        ],
        "limits": {
            "workspaces": 3,
            "max_file_size_mb": 25,
            "daily_credits": 100,
            "scheduled_agents": 2,
            "concurrent_chats": 2,
            "ocr_support": True,
            "export_formats": ["pdf", "xlsx", "docx"],
        },
        "button_text": "Current Plan",
        "button_variant": "outline",
    },
    {
        "id": "plus",
        "name": "Plus",
        "tagline": "For active analysts who need deep research and automated workflows",
        "badge": "Most Popular",
        "is_popular": True,
        "price": {
            "monthly": 20.0,
            "yearly": 192.0,
            "yearly_monthly_equivalent": 16.0,
            "currency": "USD",
            "currency_symbol": "$",
        },
        "daily_credits": 1000,
        "features": [
            "1,000 daily AI credits",
            "Unlimited Workspaces & Document Folders",
            "Large Document Upload (up to 100 MB/file)",
            "Deep Research Mode & 'Think Longer' Reasoning Flow",
            "Precise Citation Engine with High-confidence Source Verification",
            "Document Takeaways Grid & Workspace Analytics Mode",
            "Scheduled Agent Workflow Execution with Direct Email Delivery",
            "Priority OCR & Document Parsing Queue",
            "Standard Email Support",
        ],
        "limitations": [
            "Custom skill library publishing is limited to team workspace",
            "Shared API connector rate limits",
        ],
        "limits": {
            "workspaces": -1,
            "max_file_size_mb": 100,
            "daily_credits": 1000,
            "scheduled_agents": 20,
            "concurrent_chats": 10,
            "ocr_support": True,
            "export_formats": ["pdf", "xlsx", "docx", "csv", "shareable_link"],
        },
        "button_text": "Upgrade to Plus",
        "button_variant": "primary",
    },
    {
        "id": "pro",
        "name": "Pro",
        "tagline": "For power users and teams requiring advanced connectors and custom skills",
        "badge": "Professional",
        "is_popular": False,
        "price": {
            "monthly": 50.0,
            "yearly": 480.0,
            "yearly_monthly_equivalent": 40.0,
            "currency": "USD",
            "currency_symbol": "$",
        },
        "daily_credits": 5000,
        "features": [
            "5,000 daily AI credits",
            "Everything in Plus",
            "Custom Skill Generation & Skill Library Publishing",
            "Real-time Financial Connectors (SEC EDGAR, Alpha Vantage, Roic.ai)",
            "Multi-Agent Orchestration & High Concurrency Pipelines",
            "Custom Persona Prompts & Instructions for CapitalEdge",
            "Full Export Suite (PDF, DOCX, XLSX, JSON, Webhook)",
            "Dedicated Support & 99.9% Uptime SLA",
        ],
        "limitations": None,
        "limits": {
            "workspaces": -1,
            "max_file_size_mb": 250,
            "daily_credits": 5000,
            "scheduled_agents": 100,
            "concurrent_chats": 50,
            "ocr_support": True,
            "export_formats": ["pdf", "xlsx", "docx", "csv", "json", "shareable_link"],
        },
        "button_text": "Upgrade to Pro",
        "button_variant": "primary",
    },
    {
        "id": "enterprise",
        "name": "Enterprise",
        "tagline": "For organizations demanding custom security, integrations, and governance",
        "badge": "Enterprise",
        "is_popular": False,
        "price": {
            "monthly": 200.0,
            "yearly": 1920.0,
            "yearly_monthly_equivalent": 160.0,
            "currency": "USD",
            "currency_symbol": "$",
        },
        "daily_credits": 20000,
        "features": [
            "Custom & High-volume Daily AI Credits",
            "Everything in Pro",
            "Dedicated Single-Tenant Instance & DB Isolation",
            "Custom Integrations (Slack, Discord, Teams, GitHub)",
            "Enterprise SSO, SAML & Granular Role-Based Access Control",
            "Comprehensive Audit Logging & Compliance Controls",
            "Custom LLM Fine-Tuning & Self-Hosted Model Adapters",
            "Dedicated Account Manager & 24/7 Priority SLA",
        ],
        "limitations": None,
        "limits": {
            "workspaces": -1,
            "max_file_size_mb": 1024,
            "daily_credits": 20000,
            "scheduled_agents": -1,
            "concurrent_chats": -1,
            "ocr_support": True,
            "export_formats": ["pdf", "xlsx", "docx", "csv", "json", "webhook", "shareable_link"],
        },
        "button_text": "Contact Sales",
        "button_variant": "secondary",
    },
]

PLANS_BY_ID: Dict[str, Dict[str, Any]] = {p["id"]: p for p in PLAN_DEFINITIONS}


class PricingService:
    """Service handling pricing plans, subscription lifecycle, and tier management."""

    @classmethod
    def get_all_plans(cls, current_user_tier: Optional[str] = None) -> List[PlanTier]:
        """Return all available plans, marking `is_current` based on the user's active tier."""
        tier_normalized = (current_user_tier or "free").strip().lower()
        result: List[PlanTier] = []

        for p in PLAN_DEFINITIONS:
            is_current = p["id"] == tier_normalized
            # Create a shallow copy to adjust button text if current
            plan_data = p.copy()
            plan_data["is_current"] = is_current
            if is_current:
                plan_data["button_text"] = "Current Plan"
                plan_data["button_variant"] = "outline"

            result.append(
                PlanTier(
                    id=plan_data["id"],
                    name=plan_data["name"],
                    tagline=plan_data["tagline"],
                    badge=plan_data["badge"],
                    is_popular=plan_data["is_popular"],
                    is_current=plan_data["is_current"],
                    price=PlanPrice(**plan_data["price"]),
                    daily_credits=plan_data["daily_credits"],
                    features=plan_data["features"],
                    limitations=plan_data.get("limitations"),
                    limits=plan_data["limits"],
                    button_text=plan_data["button_text"],
                    button_variant=plan_data["button_variant"],
                )
            )
        return result

    @classmethod
    def get_plan_by_id(cls, plan_id: str, is_current: bool = False) -> Optional[PlanTier]:
        """Fetch a specific plan tier by its slug ID."""
        cleaned_id = (plan_id or "").strip().lower()
        raw = PLANS_BY_ID.get(cleaned_id)
        if not raw:
            return None
        data = raw.copy()
        data["is_current"] = is_current
        return PlanTier(
            id=data["id"],
            name=data["name"],
            tagline=data["tagline"],
            badge=data["badge"],
            is_popular=data["is_popular"],
            is_current=data["is_current"],
            price=PlanPrice(**data["price"]),
            daily_credits=data["daily_credits"],
            features=data["features"],
            limitations=data.get("limitations"),
            limits=data["limits"],
            button_text=data["button_text"],
            button_variant=data["button_variant"],
        )

    @classmethod
    def get_pricing_overview(cls, current_user_tier: Optional[str] = None) -> PricingPlansResponse:
        """Return the complete pricing overview response for UI display."""
        plans = cls.get_all_plans(current_user_tier=current_user_tier)
        return PricingPlansResponse(
            plans=plans,
            current_tier=current_user_tier or "free",
            currency="USD",
            billing_cycles=["monthly", "yearly"],
            discount_yearly_percent=20,
        )

    @classmethod
    def get_user_subscription(cls, user: User) -> UserSubscriptionResponse:
        """Retrieve user's complete subscription status and details."""
        tier = (getattr(user, "subscription_tier", None) or "free").strip().lower()
        status = getattr(user, "subscription_status", None) or "active"
        period = getattr(user, "subscription_period", None) or "monthly"
        daily_credits = getattr(user, "daily_credits", None) or 100
        credits_remaining = getattr(user, "credits_remaining", None)
        if credits_remaining is None:
            credits_remaining = daily_credits
        credits_used = max(0, daily_credits - credits_remaining)

        current_plan = cls.get_plan_by_id(tier, is_current=True) or cls.get_plan_by_id("free", is_current=True)
        available_plans = cls.get_all_plans(current_user_tier=tier)

        return UserSubscriptionResponse(
            id=str(user.id),
            email=user.email or "",
            subscription_tier=tier,
            subscription_status=status,
            subscription_period=period,
            daily_credits=daily_credits,
            credits_remaining=credits_remaining,
            credits_used_today=credits_used,
            current_plan=current_plan,
            available_plans=available_plans,
        )

    @classmethod
    def get_user_credits(cls, user: User) -> UserCreditsResponse:
        """Retrieve daily credit balance and quota for a user."""
        tier = (getattr(user, "subscription_tier", None) or "free").strip().lower()
        daily_credits = getattr(user, "daily_credits", None) or 100
        credits_remaining = getattr(user, "credits_remaining", None)
        if credits_remaining is None:
            credits_remaining = daily_credits
        credits_used = max(0, daily_credits - credits_remaining)
        usage_pct = round((credits_used / daily_credits) * 100, 1) if daily_credits > 0 else 0.0

        return UserCreditsResponse(
            daily_credits=daily_credits,
            credits_remaining=credits_remaining,
            credits_used_today=credits_used,
            subscription_tier=tier,
            reset_in_hours=24,
            usage_percentage=usage_pct,
            is_exhausted=(credits_remaining <= 0),
        )


    @classmethod
    def upgrade_user_plan(
        cls, db: Session, user: User, plan_id: str, billing_cycle: str = "monthly"
    ) -> UserSubscriptionResponse:
        """Upgrade user to target plan tier, adjust credit allowance and persist."""
        target_id = plan_id.strip().lower()
        cycle = billing_cycle.strip().lower()

        if target_id not in PLANS_BY_ID:
            raise ValueError(f"Unknown plan '{plan_id}'. Valid plans: {', '.join(sorted(PLANS_BY_ID.keys()))}")

        if cycle not in {"monthly", "yearly"}:
            raise ValueError(f"Invalid billing cycle '{billing_cycle}'. Must be 'monthly' or 'yearly'.")

        plan = PLANS_BY_ID[target_id]
        new_daily_credits = plan["daily_credits"]

        # Update user attributes
        user.subscription_tier = target_id
        user.subscription_period = cycle
        user.subscription_status = "active"
        user.daily_credits = new_daily_credits
        # Refill credits to new tier daily allotment
        user.credits_remaining = new_daily_credits

        # Record upgrade in the credit ledger via CRUD (no commit here — committed below)
        crud.credit_usage_log.add(
            db,
            user_id=user.id,
            amount=new_daily_credits,
            action_type="upgrade",
            description=f"Subscription upgrade to {target_id.capitalize()} ({cycle})",
            balance_after=new_daily_credits,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )

        # Single commit at service boundary
        crud.user.save(db, db_obj=user, commit=True)

        logger.info(
            "User %s successfully upgraded to tier '%s' (%s)",
            user.id,
            target_id,
            cycle,
        )

        return cls.get_user_subscription(user)
