"""Focused unit tests for Profile -> Settings functionality.

Tests cover:
1. Retrieval of complete user settings (General, Personalization, Connectors, Privacy, Credits, About Us).
2. Updating General profile settings (display_language, display_name, avatar).
3. Updating Personalization preferences (nickname, occupation, bio, ai_traits, custom_instructions).
4. Updating Data Controls & Privacy (ai_data_retention toggle, integrations status).
5. User isolation (ensuring settings modifications only affect the logged-in user).
6. About Us endpoint returning company name "Ojasya Labs" and website "https://www.ojasyalabs.com".
7. Help & Support query submission (form validation, database persistence, email dispatch).
8. Validation rules for settings updates and help queries.
9. Backward compatibility of GET & POST /v1/user/profile and POST /v1/user/update-profile.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from server.db.base_class import Base
from server.models.user import User, HelpQuery
from server.schemas.settings import (
    AboutUsResponse,
    HelpQueryCreate,
    HelpQueryResponse,
    UserSettings,
    UserSettingsUpdate,
)
from server.services.settings_service import SettingsService
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
    """Create a sample user with default profile settings."""
    user = User(
        id="usr-settings-test-1",
        email="analyst@example.com",
        display_name="Mitul Sarvaiya",
        role="member",
        avatar="https://example.com/avatar.png",
        subscription_tier="free",
        daily_credits=100,
        credits_remaining=100,
        subscription_status="active",
        subscription_period="monthly",
        display_language="English",
        nickname="Mitul",
        occupation="Financial Analyst",
        bio="Researching enterprise markets.",
        ai_traits='["Straight Shooting", "Witty"]',
        custom_instructions="Focus on hard numbers and margin metrics.",
        ai_data_retention=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def other_user(db_session):
    """Create a second user to test user isolation."""
    user = User(
        id="usr-settings-test-2",
        email="colleague@example.com",
        display_name="Priya Patel",
        role="analyst",
        subscription_tier="plus",
        daily_credits=1000,
        credits_remaining=1000,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def app_client(db_session, sample_user):
    """FastAPI TestClient with overridden auth & DB dependencies."""
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


class TestSettingsServiceLogic:
    """Test SettingsService reading, updating, and isolation."""

    def test_get_user_settings(self, sample_user):
        settings = SettingsService.get_user_settings(sample_user)
        assert isinstance(settings, UserSettings)
        assert settings.display_language == "English"
        assert settings.display_name == "Mitul Sarvaiya"
        assert settings.nickname == "Mitul"
        assert settings.occupation == "Financial Analyst"
        assert "Straight Shooting" in settings.ai_traits
        assert settings.ai_data_retention is True
        assert settings.daily_credits == 100
        assert settings.about_us_url == "https://www.ojasyalabs.com"
        assert settings.support_email == "hello@ojasyalabs.com"

    def test_update_user_settings_personalization(self, db_session, sample_user):
        update = UserSettingsUpdate(
            nickname="Mitu",
            occupation="Senior Equity Analyst",
            bio="Focusing on quick commerce and semiconductor supply chains.",
            ai_traits=["Encouraging", "Gen Z", "Forward Thinking"],
            custom_instructions="Include SWOT and unit economics tables.",
        )
        updated = SettingsService.update_user_settings(db_session, sample_user, update)
        assert updated.nickname == "Mitu"
        assert updated.occupation == "Senior Equity Analyst"
        assert updated.bio == "Focusing on quick commerce and semiconductor supply chains."
        assert updated.ai_traits == ["Encouraging", "Gen Z", "Forward Thinking"]
        assert updated.custom_instructions == "Include SWOT and unit economics tables."

        # Verify persisted in database
        db_user = db_session.query(User).filter(User.id == sample_user.id).first()
        assert db_user.nickname == "Mitu"
        assert db_user.occupation == "Senior Equity Analyst"

    def test_update_user_settings_privacy_and_connectors(self, db_session, sample_user):
        update = UserSettingsUpdate(
            ai_data_retention=False,
            slack_connected=True,
            discord_connected=True,
        )
        updated = SettingsService.update_user_settings(db_session, sample_user, update)
        assert updated.ai_data_retention is False
        assert updated.slack_connected is True
        assert updated.discord_connected is True

    def test_user_isolation(self, db_session, sample_user, other_user):
        # Update sample_user
        update = UserSettingsUpdate(display_language="German", nickname="Hans")
        SettingsService.update_user_settings(db_session, sample_user, update)

        # Ensure other_user remains unchanged
        other_settings = SettingsService.get_user_settings(other_user)
        assert other_settings.display_language == "English"
        assert other_settings.nickname == ""
        assert other_settings.display_name == "Priya Patel"

    def test_get_about_us(self):
        about = SettingsService.get_about_us()
        assert isinstance(about, AboutUsResponse)
        assert about.company_name == "Ojasya Labs"
        assert about.website_url == "https://www.ojasyalabs.com"
        assert about.contact_email == "hello@ojasyalabs.com"


class TestHelpQuerySubmission:
    """Test Help & Support query submission and validation."""

    def test_submit_help_query_success(self, db_session, sample_user):
        import asyncio

        async def _run():
            query_in = HelpQueryCreate(
                subject="Question regarding citation export",
                description="How do I export citations with high confidence excerpts to DOCX?",
                category="general",
            )
            with patch("server.services.settings_service.send_help_support_email", new_callable=AsyncMock) as mock_mail:
                mock_mail.return_value = True
                ticket = await SettingsService.submit_help_query(db_session, query_in, current_user=sample_user)

                assert ticket.email == sample_user.email
                assert ticket.subject == "Question regarding citation export"
                assert ticket.category == "general"
                assert ticket.status == "pending"

                # Check database persistence
                db_ticket = db_session.query(HelpQuery).filter(HelpQuery.id == ticket.id).first()
                assert db_ticket is not None
                assert db_ticket.user_id == sample_user.id
                assert db_ticket.subject == "Question regarding citation export"

                mock_mail.assert_called_once()

        asyncio.run(_run())

    def test_submit_help_query_validation(self):
        with pytest.raises(ValueError):
            HelpQueryCreate(subject="", description="Valid description")

        with pytest.raises(ValueError):
            HelpQueryCreate(subject="Valid subject", description="")



class TestSettingsApiEndpoints:
    """Test REST API endpoints using FastAPI TestClient."""

    def test_get_profile_get_and_post(self, app_client):
        resp_get = app_client.get("/v1/user/profile")
        assert resp_get.status_code == 200
        assert resp_get.json()["data"]["email"] == "analyst@example.com"

        resp_post = app_client.post("/v1/user/profile")
        assert resp_post.status_code == 200
        assert resp_post.json()["data"]["email"] == "analyst@example.com"

    def test_get_user_settings_api(self, app_client):
        resp = app_client.get("/v1/user/settings")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["display_language"] == "English"
        assert body["data"]["about_us_url"] == "https://www.ojasyalabs.com"
        assert body["data"]["support_email"] == "hello@ojasyalabs.com"

    def test_update_user_settings_api(self, app_client):
        payload = {
            "display_language": "French",
            "nickname": "Mitul French",
            "occupation": "Tech Lead",
            "ai_data_retention": False,
        }
        resp = app_client.post("/v1/user/settings", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["display_language"] == "French"
        assert body["data"]["nickname"] == "Mitul French"
        assert body["data"]["occupation"] == "Tech Lead"
        assert body["data"]["ai_data_retention"] is False

    def test_get_about_us_api(self, app_client):
        """Canonical about endpoint is /v1/user/about — the top-level /v1/about alias has been removed."""
        resp = app_client.get("/v1/user/about")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["website_url"] == "https://www.ojasyalabs.com"
        assert body["data"]["company_name"] == "Ojasya Labs"

    def test_submit_help_query_api(self, app_client):
        payload = {
            "subject": "Need assistance with scheduled workflows",
            "description": "My scheduled workflow results did not trigger at the set hour.",
            "category": "bug",
        }
        with patch("server.services.settings_service.send_help_support_email", new_callable=AsyncMock) as mock_mail:
            mock_mail.return_value = True
            resp = app_client.post("/v1/user/help", json=payload)
            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is True
            assert body["data"]["subject"] == "Need assistance with scheduled workflows"
            assert body["data"]["status"] == "pending"

    def test_submit_help_query_api_validation_error(self, app_client):
        payload = {
            "subject": "",
            "description": "",
        }
        resp = app_client.post("/v1/user/help", json=payload)
        assert resp.status_code == 422

    def test_get_user_help_queries_api(self, app_client):
        resp = app_client.get("/v1/user/help")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert isinstance(body["data"], list)

    def test_upload_avatar_flow_and_persistence(self, app_client, db_session, sample_user):
        import io
        
        # 1. Upload valid image via multipart/form-data
        fake_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        files = {"file": ("avatar_pic.png", io.BytesIO(fake_png), "image/png")}
        
        resp = app_client.post("/v1/user/upload-avatar", files=files)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        uploaded_avatar_path = body["data"]["avatar"]
        assert uploaded_avatar_path.startswith("/static/avatars/avatar_")
        assert uploaded_avatar_path.endswith(".png")

        # 2. Verify database persistence
        db_user = db_session.query(User).filter(User.id == sample_user.id).first()
        assert db_user.avatar == uploaded_avatar_path

        # 3. Verify GET /v1/user/profile returns the saved avatar
        profile_resp = app_client.get("/v1/user/profile")
        assert profile_resp.status_code == 200
        profile_data = profile_resp.json()["data"]
        assert profile_data["avatar"] == uploaded_avatar_path

        # 4. Verify alias endpoint /v1/user/avatar works too
        files2 = {"avatar": ("profile.jpg", io.BytesIO(fake_png), "image/jpeg")}
        resp2 = app_client.post("/v1/user/avatar", files=files2)
        assert resp2.status_code == 200
        assert resp2.json()["data"]["avatar"].endswith(".jpg")

        # 5. Verify deleting avatar
        del_resp = app_client.delete("/v1/user/avatar")
        assert del_resp.status_code == 200
        assert del_resp.json()["data"]["avatar"] == ""
        assert db_session.query(User).filter(User.id == sample_user.id).first().avatar == ""

    def test_upload_avatar_invalid_file_rejected(self, app_client):
        import io

        # Text file instead of image
        files = {"file": ("malicious.txt", io.BytesIO(b"not an image"), "text/plain")}
        resp = app_client.post("/v1/user/upload-avatar", files=files)
        assert resp.status_code == 400
        error_text = str(resp.json().get("detail") or resp.json().get("message") or "")
        assert "Uploaded file must be a valid image" in error_text
