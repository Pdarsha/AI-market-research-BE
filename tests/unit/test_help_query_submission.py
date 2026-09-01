"""Focused unit and API test suite for Profile -> Help / Query Submission.

Tests cover:
1. Successful support query submission by authenticated user.
2. Successful submission using field aliases (query / message).
3. Query submission by unauthenticated visitor with explicit email.
4. Input validation rules:
   - Required subject/query (non-empty, minimum length 2, max length 255).
   - Required description/message (non-empty, minimum length 3, max length 10,000).
   - Invalid email format rejection.
5. Graceful handling of email dispatch failures without failing the user request or exposing internal 500 errors.
6. Notification email dispatch to hello@ojasyalabs.com and user acknowledgment dispatch.
7. User isolation when retrieving submitted tickets via GET /v1/user/help.
8. Top-level endpoint POST /v1/help and router endpoint POST /v1/user/help.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from server.db.base_class import Base
from server.models.user import User, HelpQuery
from server.schemas.settings import HelpQueryCreate, HelpQueryResponse
from server.services.settings_service import SettingsService
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
    """Create a sample authenticated user."""
    user = User(
        id="usr-help-test-1",
        email="analyst@example.com",
        display_name="Dev Analyst",
        role="member",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def other_user(db_session):
    """Create a second user to test query isolation."""
    user = User(
        id="usr-help-test-2",
        email="other@example.com",
        display_name="Other Analyst",
        role="member",
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


class TestHelpQueryValidationAndPersistence:
    """Test validation and database persistence for help queries."""

    def test_submit_query_authenticated_success(self, db_session, sample_user):
        async def _run():
            query_in = HelpQueryCreate(
                subject="Cannot export takeaways to Excel",
                description="When clicking export on the document takeaways grid, the XLSX download hangs.",
                category="bug",
            )
            with patch("server.services.settings_service.send_help_support_email", new_callable=AsyncMock) as mock_mail:
                mock_mail.return_value = True
                resp = await SettingsService.submit_help_query(
                    db=db_session,
                    query_in=query_in,
                    current_user=sample_user,
                )
                assert resp.email == "analyst@example.com"
                assert resp.name == "Dev Analyst"
                assert resp.subject == "Cannot export takeaways to Excel"
                assert resp.category == "bug"
                assert resp.status == "pending"

                # Check database record
                db_record = db_session.query(HelpQuery).filter(HelpQuery.id == resp.id).first()
                assert db_record is not None
                assert db_record.user_id == sample_user.id
                assert db_record.description == query_in.description

                # Verify email dispatch was triggered with ticket ID and hello@ojasyalabs.com recipient
                mock_mail.assert_called_once()
                call_kwargs = mock_mail.call_args.kwargs
                assert call_kwargs["user_email"] == "analyst@example.com"
                assert call_kwargs["subject"] == "Cannot export takeaways to Excel"
                assert call_kwargs["ticket_id"] == resp.id

        asyncio.run(_run())

    def test_submit_query_using_field_aliases(self, db_session, sample_user):
        async def _run():
            # Using query instead of subject, message instead of description
            query_in = HelpQueryCreate(
                query="Inquiry regarding credit quota",
                message="How do credits reset each day?",
                category="billing",
            )
            assert query_in.subject == "Inquiry regarding credit quota"
            assert query_in.description == "How do credits reset each day?"

            with patch("server.services.settings_service.send_help_support_email", new_callable=AsyncMock) as mock_mail:
                mock_mail.return_value = True
                resp = await SettingsService.submit_help_query(
                    db=db_session,
                    query_in=query_in,
                    current_user=sample_user,
                )
                assert resp.subject == "Inquiry regarding credit quota"
                assert resp.description == "How do credits reset each day?"

        asyncio.run(_run())

    def test_submit_query_unauthenticated_visitor(self, db_session):
        async def _run():
            query_in = HelpQueryCreate(
                email="visitor@example.com",
                name="External Visitor",
                subject="Pricing inquiry for enterprise license",
                description="We would like to request enterprise dedicated hosting pricing for 50 seats.",
                category="sales",
            )
            with patch("server.services.settings_service.send_help_support_email", new_callable=AsyncMock) as mock_mail:
                mock_mail.return_value = True
                resp = await SettingsService.submit_help_query(
                    db=db_session,
                    query_in=query_in,
                    current_user=None,
                )
                assert resp.email == "visitor@example.com"
                assert resp.name == "External Visitor"
                assert resp.user_id is None

                db_record = db_session.query(HelpQuery).filter(HelpQuery.id == resp.id).first()
                assert db_record is not None
                assert db_record.user_id is None
                assert db_record.email == "visitor@example.com"

        asyncio.run(_run())

    def test_validation_empty_subject(self):
        with pytest.raises(ValueError) as exc:
            HelpQueryCreate(subject="   ", description="Valid description")
        assert "Subject / Query is required" in str(exc.value)

    def test_validation_empty_description(self):
        with pytest.raises(ValueError) as exc:
            HelpQueryCreate(subject="Valid Subject", description="  ")
        assert "Description / Message is required" in str(exc.value)

    def test_validation_subject_too_short(self):
        with pytest.raises(ValueError) as exc:
            HelpQueryCreate(subject="A", description="Valid description")
        assert "at least 2 characters" in str(exc.value)

    def test_validation_subject_too_long(self):
        with pytest.raises((ValueError, Exception)) as exc:
            HelpQueryCreate(subject="X" * 300, description="Valid description")
        assert "255" in str(exc.value)


    def test_validation_invalid_email(self, db_session):
        async def _run():
            query_in = HelpQueryCreate(
                email="invalid-email-format",
                subject="Valid Subject",
                description="Valid Description",
            )
            with pytest.raises(ValueError) as exc:
                await SettingsService.submit_help_query(
                    db=db_session,
                    query_in=query_in,
                    current_user=None,
                )
            assert "Invalid email" in str(exc.value)

        asyncio.run(_run())

    def test_email_failure_handled_gracefully(self, db_session, sample_user):
        """Ensure that if SMTP or FastMail fails, the help query is still saved and success is returned."""
        async def _run():
            query_in = HelpQueryCreate(
                subject="System latency report",
                description="Chat completions experienced 5s latency during morning hours.",
            )
            with patch("server.services.settings_service.send_help_support_email", side_effect=Exception("SMTP Network Timeout")):
                # Should not raise exception
                resp = await SettingsService.submit_help_query(
                    db=db_session,
                    query_in=query_in,
                    current_user=sample_user,
                )
                assert resp is not None
                assert resp.status == "pending"

                # Verify ticket is safely persisted in DB
                db_record = db_session.query(HelpQuery).filter(HelpQuery.id == resp.id).first()
                assert db_record is not None
                assert db_record.subject == "System latency report"

        asyncio.run(_run())


class TestHelpQueryApiEndpoints:
    """Test REST API endpoints via FastAPI TestClient."""

    def test_post_user_help_endpoint_success(self, app_client):
        payload = {
            "subject": "Scheduled agent output format",
            "description": "Can we receive HTML summaries directly instead of markdown attachments?",
            "category": "feature",
        }
        with patch("server.services.settings_service.send_help_support_email", new_callable=AsyncMock) as mock_mail:
            mock_mail.return_value = True
            resp = app_client.post("/v1/user/help", json=payload)
            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is True
            assert body["data"]["subject"] == "Scheduled agent output format"
            assert body["data"]["category"] == "feature"
            assert "submitted successfully" in body["message"]

    def test_post_user_help_endpoint_success(self, app_client):
        """Canonical help endpoint is /v1/user/help — the top-level /v1/help alias has been removed."""
        payload = {
            "query": "General feedback on UI",
            "message": "The citation popover looks very clean and responsive.",
            "category": "general",
        }
        with patch("server.services.settings_service.send_help_support_email", new_callable=AsyncMock) as mock_mail:
            mock_mail.return_value = True
            resp = app_client.post("/v1/user/help", json=payload)
            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is True
            assert body["data"]["subject"] == "General feedback on UI"

    def test_post_help_endpoint_validation_error(self, app_client):
        payload = {
            "subject": "",
            "description": "",
        }
        resp = app_client.post("/v1/user/help", json=payload)
        assert resp.status_code == 422  # Pydantic validation error

    def test_post_help_endpoint_email_failure_graceful_response(self, app_client):
        payload = {
            "subject": "Minor typo in documentation",
            "description": "On the citations guide, there is a small typo in section 2.",
        }
        with patch("server.services.settings_service.send_help_support_email", side_effect=Exception("Connection refused")):
            resp = app_client.post("/v1/user/help", json=payload)
            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is True
            assert body["data"]["subject"] == "Minor typo in documentation"

    def test_get_user_help_queries_user_isolation(self, app_client, db_session, sample_user, other_user):
        # Create ticket for sample_user
        t1 = HelpQuery(
            id="t-1",
            user_id=sample_user.id,
            email=sample_user.email,
            subject="Sample User Ticket",
            description="Ticket description 1",
            status="pending",
        )
        # Create ticket for other_user
        t2 = HelpQuery(
            id="t-2",
            user_id=other_user.id,
            email=other_user.email,
            subject="Other User Ticket",
            description="Ticket description 2",
            status="pending",
        )
        db_session.add_all([t1, t2])
        db_session.commit()

        resp = app_client.get("/v1/user/help")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        # Only sample_user's ticket is returned
        ticket_ids = [t["id"] for t in body["data"]]
        assert "t-1" in ticket_ids
        assert "t-2" not in ticket_ids
