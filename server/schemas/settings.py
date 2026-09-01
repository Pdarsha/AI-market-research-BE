from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator, field_validator


SUPPORTED_LANGUAGES = {
    "English",
    "Spanish",
    "French",
    "German",
    "Japanese",
    "Chinese",
    "Hindi",
    "Portuguese",
    "Italian",
    "Dutch",
    "Korean",
    "Russian",
    "Arabic",
}

DEFAULT_AI_TRAITS = [
    "Chatty",
    "Witty",
    "Straight Shooting",
    "Encouraging",
    "Gen Z",
    "Skeptical",
    "Traditional",
    "Forward Thinking",
    "Poetic",
]


class UserSettings(BaseModel):
    # General Tab
    display_language: str = Field(default="English", description="UI display language")
    display_name: Optional[str] = Field(default=None, description="Display name / full name")
    full_name: Optional[str] = Field(default=None, description="Alias for display name")
    avatar: Optional[str] = Field(default=None, description="Profile avatar URL or empty")
    email: Optional[str] = Field(default=None, description="User email address")
    role: Optional[str] = Field(default="member", description="User role in the system")

    # Personalization Tab
    nickname: Optional[str] = Field(default=None, description="What CapitalEdge should call the user")
    occupation: Optional[str] = Field(default=None, description="User profession / occupation")
    bio: Optional[str] = Field(default=None, description="Short user profile description")
    ai_traits: List[str] = Field(default_factory=list, description="Selected AI personality traits")
    custom_instructions: Optional[str] = Field(default=None, description="Custom prompt instructions for CapitalEdge")

    # Connectors & Integrations Tabs
    github_connected: bool = Field(default=True, description="Whether GitHub connector is active")
    slack_connected: bool = Field(default=False, description="Whether Slack integration is active")
    discord_connected: bool = Field(default=False, description="Whether Discord integration is active")
    line_connected: bool = Field(default=False, description="Whether Line integration is active")
    telegram_connected: bool = Field(default=False, description="Whether Telegram integration is active")

    # Data Controls & Privacy
    ai_data_retention: bool = Field(default=True, description="Allow CapitalEdge searches to improve models")
    shared_tasks_count: int = Field(default=0, description="Total shared tasks")
    deployed_agents_count: int = Field(default=0, description="Total active deployed agents")
    purchased_domains: List[str] = Field(default_factory=list, description="Domains purchased by user")

    # Subscription & Credits
    subscription_tier: str = Field(default="free", description="Active subscription plan tier")
    subscription_status: str = Field(default="active", description="Subscription state")
    daily_credits: int = Field(default=100, description="Daily credit limit")
    credits_remaining: int = Field(default=100, description="Credits left for today")
    credits_used_today: int = Field(default=0, description="Credits consumed today")

    # Company & Navigation
    about_us_url: str = Field(default="https://www.ojasyalabs.com", description="Company about us website link")
    support_email: str = Field(default="hello@ojasyalabs.com", description="Designated company support email")
    platform_name: str = Field(default="CapitalEdge", description="Platform name")

    class Config:
        from_attributes = True


class UserSettingsUpdate(BaseModel):
    display_language: Optional[str] = Field(default=None, description="UI display language")
    display_name: Optional[str] = Field(default=None, description="Display name / full name")
    name: Optional[str] = Field(default=None, description="Alias for display_name")
    avatar: Optional[str] = Field(default=None, description="Avatar image URL")
    role: Optional[str] = Field(default=None, description="Role")

    # Personalization
    nickname: Optional[str] = Field(default=None, description="Nickname")
    occupation: Optional[str] = Field(default=None, description="Occupation")
    bio: Optional[str] = Field(default=None, description="Profile bio")
    ai_traits: Optional[List[str]] = Field(default=None, description="AI personality traits list")
    custom_instructions: Optional[str] = Field(default=None, description="Custom instructions for AI")

    # Privacy & Controls
    ai_data_retention: Optional[bool] = Field(default=None, description="AI data retention toggle")

    # Connectors & Integrations
    github_connected: Optional[bool] = Field(default=None)
    slack_connected: Optional[bool] = Field(default=None)
    discord_connected: Optional[bool] = Field(default=None)
    line_connected: Optional[bool] = Field(default=None)
    telegram_connected: Optional[bool] = Field(default=None)

    @field_validator("display_language")
    @classmethod
    def validate_language(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return v
        cleaned = v.strip().capitalize()
        if cleaned not in SUPPORTED_LANGUAGES:
            # Normalize or allow if valid string
            return cleaned
        return cleaned

    @field_validator("avatar")
    @classmethod
    def validate_avatar(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        cleaned = v.strip()
        if not cleaned:
            return ""
        if not (
            cleaned.startswith("http://")
            or cleaned.startswith("https://")
            or cleaned.startswith("data:image/")
            or cleaned.startswith("/")
        ):
            raise ValueError("Avatar must be a valid http/https URL, path, or data URI.")
        return cleaned


class HelpQueryCreate(BaseModel):
    subject: Optional[str] = Field(default=None, max_length=255, description="Subject / Query Title")
    query: Optional[str] = Field(default=None, max_length=255, description="Alias for subject")
    description: Optional[str] = Field(default=None, max_length=10000, description="Detailed problem description or inquiry")
    message: Optional[str] = Field(default=None, max_length=10000, description="Alias for description")
    category: Optional[str] = Field(default="general", max_length=50, description="Category: general, billing, bug, feature")
    email: Optional[str] = Field(default=None, max_length=255, description="User email (optional override if logged in)")
    name: Optional[str] = Field(default=None, max_length=255, description="User name")

    @model_validator(mode="after")
    def validate_and_normalize(self):
        # Resolve subject
        resolved_subject = (self.subject or self.query or "").strip()
        if not resolved_subject:
            raise ValueError("Subject / Query is required and cannot be empty.")
        if len(resolved_subject) < 2:
            raise ValueError("Subject / Query must be at least 2 characters long.")
        if len(resolved_subject) > 255:
            raise ValueError("Subject / Query cannot exceed 255 characters.")
        self.subject = resolved_subject

        # Resolve description
        resolved_desc = (self.description or self.message or "").strip()
        if not resolved_desc:
            raise ValueError("Description / Message is required and cannot be empty.")
        if len(resolved_desc) < 3:
            raise ValueError("Description must be at least 3 characters long.")
        if len(resolved_desc) > 10000:
            raise ValueError("Description cannot exceed 10,000 characters.")
        self.description = resolved_desc

        # Resolve category
        self.category = (self.category or "general").strip().lower()
        return self



class HelpQueryResponse(BaseModel):
    id: str = Field(..., description="Help query ticket ID")
    user_id: Optional[str] = Field(default=None, description="User ID if submitted by logged-in user")
    email: str = Field(..., description="Contact email address")
    name: Optional[str] = Field(default=None, description="Contact name")
    subject: str = Field(..., description="Query subject")
    description: str = Field(..., description="Query description")
    category: str = Field(default="general", description="Query category")
    status: str = Field(default="pending", description="Ticket status: pending, in_review, resolved")
    created_at: datetime = Field(..., description="Submission timestamp")

    class Config:
        from_attributes = True


class AboutUsResponse(BaseModel):
    platform_name: str = Field(default="CapitalEdge", description="Platform name")
    company_name: str = Field(default="Ojasya Labs", description="Company / parent organization name")
    website_url: str = Field(default="https://www.ojasyalabs.com", description="Official website URL")
    contact_email: str = Field(default="hello@ojasyalabs.com", description="Designated company contact email")
    description: str = Field(
        default="Conversational AI Workspace & Agentic Workflow Platform",
        description="Platform tagline / description",
    )
    features: List[str] = Field(
        default_factory=lambda: [
            "Conversational RAG Search & Citations",
            "Multi-Agent Workflow Engine & Schedulers",
            "Deep Research & Document Analytics",
            "Custom Skill Generation & Publishing",
        ]
    )
