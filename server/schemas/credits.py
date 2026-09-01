from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class UserCreditsResponse(BaseModel):
    daily_credits: int = Field(default=100, description="Daily credit allowance limit")
    credits_remaining: int = Field(default=100, description="Remaining credits for today")
    credits_used_today: int = Field(default=0, description="Credits consumed today")
    subscription_tier: str = Field(default="free", description="Active subscription tier name")
    reset_in_hours: int = Field(default=24, description="Hours until daily credit balance resets")
    last_reset_at: Optional[datetime] = Field(default=None, description="Timestamp of last reset")
    next_reset_at: Optional[datetime] = Field(default=None, description="Timestamp of next scheduled reset")
    usage_percentage: float = Field(default=0.0, description="Percentage of daily credits consumed")
    is_exhausted: bool = Field(default=False, description="True if no credits remain today")

    class Config:
        from_attributes = True


class CreditConsumeRequest(BaseModel):
    amount: int = Field(default=1, gt=0, description="Number of credits to consume (positive integer)")
    purpose: Optional[str] = Field(default="ai_generation", description="Purpose of deduction")

    @field_validator("amount")
    @classmethod
    def validate_positive_amount(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Credit consumption amount must be a positive integer greater than 0.")
        return v


class CreditRefundRequest(BaseModel):
    amount: int = Field(default=1, gt=0, description="Number of credits to refund (positive integer)")
    reason: Optional[str] = Field(default="error_compensation", description="Reason for credit refund")

    @field_validator("amount")
    @classmethod
    def validate_positive_refund(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Credit refund amount must be a positive integer greater than 0.")
        return v


class CreditUsageLogResponse(BaseModel):
    id: str = Field(..., description="Unique log ID")
    amount: int = Field(..., description="Credit delta (negative for consumption, positive for reset/refund)")
    action_type: str = Field(..., description="Action type: consume, reset, refund, bonus")
    description: Optional[str] = Field(default=None, description="Detail/purpose of credit movement")
    balance_after: int = Field(..., description="Credit balance after this transaction")
    created_at: datetime = Field(..., description="Timestamp of transaction")

    class Config:
        from_attributes = True
