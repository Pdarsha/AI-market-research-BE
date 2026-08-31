"""User Profile & Settings Management Service.

Provides complete handling for:
1. Reading and updating user profile settings across General, Personalization, Connectors,
   Integrations, Data Controls, and Account sections.
2. Managing help query support tickets with email dispatch to hello@ojasyalabs.com and
   acknowledgment to the user.
3. Providing company details (About Us -> https://www.ojasyalabs.com) and credits balance.
"""

import json
import logging
from typing import Any, Dict, Optional
from sqlalchemy.orm import Session

from server.models.user import User
from server.schemas.settings import (
    AboutUsResponse,
    DEFAULT_AI_TRAITS,
    HelpQueryCreate,
    HelpQueryResponse,
    UserSettings,
    UserSettingsUpdate,
)
from server.utils.mail import send_help_support_email, is_valid_email
from server.services.audit_logger import AuditLogger
from server import crud


logger = logging.getLogger(__name__)


class SettingsService:
    """Service handling user settings, profile preferences, and support tickets."""

    @classmethod
    def get_user_settings(cls, user: User) -> UserSettings:
        """Construct full UserSettings schema for the given User."""
        # Parse traits
        traits = DEFAULT_AI_TRAITS.copy()
        if user.ai_traits:
            try:
                parsed = json.loads(user.ai_traits)
                if isinstance(parsed, list):
                    traits = parsed
            except Exception:
                traits = [t.strip() for t in user.ai_traits.split(",") if t.strip()]

        # Parse settings_json if stored
        extra_settings: Dict[str, Any] = {}
        if user.settings_json:
            try:
                extra_settings = json.loads(user.settings_json)
            except Exception:
                pass

        disp_name = (
            user.display_name
            or getattr(user, "username", None)
            or (user.email.split("@")[0] if user.email else "User")
        )

        daily_credits = getattr(user, "daily_credits", None) or 100
        credits_remaining = getattr(user, "credits_remaining", None)
        if credits_remaining is None:
            credits_remaining = daily_credits
        credits_used = max(0, daily_credits - credits_remaining)

        return UserSettings(
            display_language=user.display_language or "English",
            display_name=disp_name,
            full_name=user.display_name or disp_name,
            avatar=user.avatar or "",
            email=user.email or "",
            role=str(user.role) if user.role is not None else "member",
            nickname=user.nickname or "",
            occupation=user.occupation or "",
            bio=user.bio or "",
            ai_traits=traits,
            custom_instructions=user.custom_instructions or "",
            github_connected=extra_settings.get("github_connected", True),
            slack_connected=extra_settings.get("slack_connected", False),
            discord_connected=extra_settings.get("discord_connected", False),
            line_connected=extra_settings.get("line_connected", False),
            telegram_connected=extra_settings.get("telegram_connected", False),
            ai_data_retention=user.ai_data_retention if user.ai_data_retention is not None else True,
            shared_tasks_count=extra_settings.get("shared_tasks_count", 0),
            deployed_agents_count=extra_settings.get("deployed_agents_count", 0),
            purchased_domains=extra_settings.get("purchased_domains", []),
            subscription_tier=user.subscription_tier or "free",
            subscription_status=user.subscription_status or "active",
            daily_credits=daily_credits,
            credits_remaining=credits_remaining,
            credits_used_today=credits_used,
            about_us_url="https://www.ojasyalabs.com",
            support_email="hello@ojasyalabs.com",
            platform_name="CapitalEdge",
        )

    @classmethod
    def update_user_settings(
        cls, db: Session, user: User, update_data: UserSettingsUpdate
    ) -> UserSettings:
        """Update settings fields for the user and persist to the database."""
        # Update General
        if update_data.display_language is not None:
            user.display_language = update_data.display_language.strip()

        if update_data.display_name is not None:
            user.display_name = update_data.display_name.strip()
        elif update_data.name is not None:
            user.display_name = update_data.name.strip()

        if update_data.avatar is not None:
            user.avatar = update_data.avatar.strip()

        if update_data.role is not None:
            user.role = update_data.role.strip()

        # Update Personalization
        if update_data.nickname is not None:
            user.nickname = update_data.nickname.strip()

        if update_data.occupation is not None:
            user.occupation = update_data.occupation.strip()

        if update_data.bio is not None:
            user.bio = update_data.bio.strip()

        if update_data.ai_traits is not None:
            user.ai_traits = json.dumps(update_data.ai_traits)

        if update_data.custom_instructions is not None:
            user.custom_instructions = update_data.custom_instructions.strip()

        # Update Privacy & Data controls
        if update_data.ai_data_retention is not None:
            user.ai_data_retention = bool(update_data.ai_data_retention)

        # Update Connectors / Integrations inside settings_json
        extra_settings: Dict[str, Any] = {}
        if user.settings_json:
            try:
                extra_settings = json.loads(user.settings_json)
            except Exception:
                pass

        if update_data.github_connected is not None:
            extra_settings["github_connected"] = bool(update_data.github_connected)
        if update_data.slack_connected is not None:
            extra_settings["slack_connected"] = bool(update_data.slack_connected)
        if update_data.discord_connected is not None:
            extra_settings["discord_connected"] = bool(update_data.discord_connected)
        if update_data.line_connected is not None:
            extra_settings["line_connected"] = bool(update_data.line_connected)
        if update_data.telegram_connected is not None:
            extra_settings["telegram_connected"] = bool(update_data.telegram_connected)

        user.settings_json = json.dumps(extra_settings)

        crud.user.save(db, db_obj=user, commit=True)

        logger.info("User %s settings updated successfully.", user.id)
        return cls.get_user_settings(user)

    @classmethod
    async def submit_help_query(
        cls,
        db: Session,
        query_in: HelpQueryCreate,
        current_user: Optional[User] = None,
    ) -> HelpQueryResponse:
        """Save a new user support ticket and dispatch email notification and acknowledgment."""
        user_id = str(current_user.id) if current_user else None
        email = (
            (current_user.email if current_user and current_user.email else None)
            or query_in.email
            or ""
        ).strip().lower()

        if not email:
            raise ValueError("Contact email is required to submit a help query.")

        if not is_valid_email(email):
            raise ValueError(f"Invalid email address '{email}'.")

        name = (
            query_in.name
            or (current_user.display_name if current_user else None)
            or (email.split("@")[0])
        )

        subject = (query_in.subject or query_in.query or "").strip()
        description = (query_in.description or query_in.message or "").strip()
        category = (query_in.category or "general").strip().lower()

        ticket = crud.help_query.create_query(
            db=db,
            user_id=user_id,
            email=email,
            name=name,
            subject=subject,
            description=description,
            category=category,
        )

        logger.info(
            "Created support ticket #%s from user %s <%s>: %s",
            ticket.id,
            name,
            email,
            subject,
        )

        AuditLogger.log(
            action="USER_SUBMIT_HELP_QUERY",
            user_id=user_id,
            endpoint="/v1/user/help",
            http_method="POST",
            status="200",
            metadata={
                "ticket_id": ticket.id,
                "email": email,
                "subject": subject,
                "category": category,
            },
        )

        # Dispatch async notification emails (non-blocking / resilient)
        try:
            await send_help_support_email(
                user_email=email,
                user_name=name,
                subject=subject,
                description=description,
                ticket_id=ticket.id,
                category=category,
            )
        except Exception as e:
            logger.warning("Error during support email dispatch for ticket #%s: %s", ticket.id, e)

        return HelpQueryResponse.model_validate(ticket)

    @classmethod
    def get_about_us(cls) -> AboutUsResponse:
        """Return company and platform information."""
        return AboutUsResponse(
            platform_name="CapitalEdge",
            company_name="Ojasya Labs",
            website_url="https://www.ojasyalabs.com",
            contact_email="hello@ojasyalabs.com",
            description="Conversational AI Workspace & Agentic Workflow Platform",
            features=[
                "Conversational RAG Search & Citations",
                "Multi-Agent Workflow Engine & Schedulers",
                "Deep Research & Document Analytics",
                "Custom Skill Generation & Publishing",
            ],
        )
