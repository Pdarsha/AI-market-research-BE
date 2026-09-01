from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class PlanPrice(BaseModel):
    monthly: float = Field(..., description="Monthly price in specified currency")
    yearly: float = Field(..., description="Yearly price in specified currency (billed annually)")
    yearly_monthly_equivalent: float = Field(..., description="Effective monthly price when billed annually")
    currency: str = Field(default="USD", description="Three-letter currency code")
    currency_symbol: str = Field(default="$", description="Currency symbol")


class PlanTier(BaseModel):
    id: str = Field(..., description="Unique plan identifier: free, plus, pro, enterprise")
    name: str = Field(..., description="Human-readable plan name")
    tagline: str = Field(..., description="Brief one-line summary of target audience")
    badge: Optional[str] = Field(None, description="Optional badge, e.g., 'Most Popular', 'Professional'")
    is_popular: bool = Field(default=False, description="Flag indicating if plan is highlighted as popular")
    is_current: bool = Field(default=False, description="Whether this is the user's currently active plan")
    price: PlanPrice = Field(..., description="Pricing structure across billing intervals")
    daily_credits: int = Field(..., description="Number of AI credits refreshed daily")
    features: List[str] = Field(default_factory=list, description="List of key features included in this plan")
    limitations: Optional[List[str]] = Field(default=None, description="List of limitations or missing features")
    limits: Dict[str, Any] = Field(default_factory=dict, description="Numerical/capability limits for this tier")
    button_text: str = Field(..., description="Call-to-action button label")
    button_variant: str = Field(default="primary", description="Button UI variant style")


class PricingPlansResponse(BaseModel):
    plans: List[PlanTier] = Field(..., description="List of all available subscription tiers")
    current_tier: Optional[str] = Field(default="free", description="User's currently active subscription tier")
    currency: str = Field(default="USD", description="Default currency code")
    billing_cycles: List[str] = Field(default=["monthly", "yearly"], description="Supported billing intervals")
    discount_yearly_percent: int = Field(default=20, description="Percentage discount for annual subscriptions")


class UpgradePlanRequest(BaseModel):
    plan_id: str = Field(..., description="Target plan ID: free, plus, pro, enterprise")
    billing_cycle: Optional[str] = Field(default="monthly", description="Billing cycle: monthly or yearly")

    @field_validator("plan_id")
    @classmethod
    def validate_plan_id(cls, v: str) -> str:
        cleaned = (v or "").strip().lower()
        valid_plans = {"free", "plus", "pro", "enterprise"}
        if cleaned not in valid_plans:
            raise ValueError(f"Invalid plan_id '{v}'. Must be one of: {', '.join(sorted(valid_plans))}")
        return cleaned

    @field_validator("billing_cycle")
    @classmethod
    def validate_billing_cycle(cls, v: Optional[str]) -> str:
        if not v:
            return "monthly"
        cleaned = v.strip().lower()
        valid_cycles = {"monthly", "yearly"}
        if cleaned not in valid_cycles:
            raise ValueError(f"Invalid billing_cycle '{v}'. Must be one of: {', '.join(sorted(valid_cycles))}")
        return cleaned


class UserSubscriptionResponse(BaseModel):
    id: str = Field(..., description="User unique ID")
    email: str = Field(..., description="User email address")
    subscription_tier: str = Field(..., description="Active subscription tier: free, plus, pro, enterprise")
    subscription_status: str = Field(default="active", description="Status of subscription: active, trialing, canceled")
    subscription_period: str = Field(default="monthly", description="Billing interval: monthly or yearly")
    daily_credits: int = Field(default=100, description="Daily credit allowance")
    credits_remaining: int = Field(default=100, description="Credits remaining for today")
    credits_used_today: int = Field(default=0, description="Credits consumed today")
    current_plan: PlanTier = Field(..., description="Detailed plan tier object for current subscription")
    available_plans: List[PlanTier] = Field(default_factory=list, description="Available upgrade plan tiers")


class UserCreditsResponse(BaseModel):
    daily_credits: int = Field(..., description="Daily limit of credits")
    credits_remaining: int = Field(..., description="Remaining credits for the day")
    credits_used_today: int = Field(default=0, description="Credits used today")
    subscription_tier: str = Field(..., description="Active subscription tier name")
    reset_in_hours: int = Field(default=24, description="Hours until daily credit balance resets")
