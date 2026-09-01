"""Focused unit tests for scheduled Agent Workflow email delivery.

Tests cover:
1. Agent workflow creation with automatic logged-in user email capture.
2. Agent workflow creation with explicit recipient_email / user_email.
3. Agent workflow schema normalization and validation for email fields.
4. Email template rendering and formatting with markdown and sources.
5. Email syntax validation and invalid email safe handling.
6. Scheduled execution triggering email dispatch to the associated user's email.
7. Resilient execution when email dispatch encounters errors or missing emails.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from server.db.base_class import Base
from server.models.agent_workflow import (
    AgentStatus,
    AgentWorkflow,
    AgentWorkflowExecution,
    ExecutionStatus,
    TriggerType,
)
from server.models.user import User
from server.schemas.agent_workflow import (
    AgentWorkflowCreate,
    AgentWorkflowResponse,
    AgentWorkflowUpdate,
    AgentWorkflowListItem,
    AgentWorkflowExecutionResponse,
)
from server.crud.crud_agent_workflow import CRUDAgentWorkflow
from server.services.agent_execution_service import AgentExecutionService
from server.utils.mail import (
    get_agent_result_template,
    is_valid_email,
    send_agent_workflow_email,
)


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


@pytest.fixture
def test_user(db_session):
    """Create a mock registered user in DB."""
    user = User(
        id="user-12345",
        email="analyst@capitaledge.ai",
        display_name="Senior Analyst",
        is_verified=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


class TestAgentEmailValidationAndTemplates:
    """Test email validation and HTML template rendering."""

    def test_is_valid_email(self):
        assert is_valid_email("user@example.com") is True
        assert is_valid_email("first.last+test@sub.domain.co.uk") is True
        assert is_valid_email("invalid-email") is False
        assert is_valid_email("@missinguser.com") is False
        assert is_valid_email("missingdomain@") is False
        assert is_valid_email("") is False
        assert is_valid_email(None) is False
        assert is_valid_email(123) is False

    def test_get_agent_result_template_rendering(self):
        html_body = get_agent_result_template(
            agent_name="Daily Market Summary",
            result_content="## Market Update\n\n- **S&P 500**: +1.2%\n- **NASDAQ**: +1.8%\n\nAll segments positive.",
            execution_time=datetime(2026, 8, 31, 10, 0, 0, tzinfo=timezone.utc),
            sources=["Q3_Report.pdf", "Market_Brief.docx"],
        )

        assert "CapitalEdge" in html_body
        assert "Daily Market Summary" in html_body
        assert "Market Update" in html_body
        assert "S&amp;P 500" in html_body or "S&P 500" in html_body
        assert "Q3_Report.pdf" in html_body
        assert "Market_Brief.docx" in html_body
        assert "2026-08-31 10:00:00 UTC" in html_body

    def test_get_agent_result_template_strips_system_markers(self):
        raw = "Analysis complete.\n\n<<<FOLLOW_UP_QUESTIONS>>>\n[\"Q1?\", \"Q2?\"]"
        html_body = get_agent_result_template(
            agent_name="Tech Report",
            result_content=raw,
        )
        assert "Analysis complete." in html_body
        assert "<<<FOLLOW_UP_QUESTIONS>>>" not in html_body
        assert "[\"Q1?\"" not in html_body


class TestAgentSchemaEmailAssociation:
    """Test schema normalization and email field synchronization."""

    def test_schema_sync_user_email_to_recipient_email(self):
        data = {
            "name": "Earnings Tracker",
            "instructions": "Track earnings",
            "user_email": "custom@capitaledge.ai",
        }
        create_schema = AgentWorkflowCreate(**data)
        assert create_schema.recipient_email == "custom@capitaledge.ai"
        assert create_schema.user_email == "custom@capitaledge.ai"

    def test_schema_sync_recipient_email_to_user_email(self):
        data = {
            "name": "Earnings Tracker",
            "instructions": "Track earnings",
            "recipient_email": "recipient@capitaledge.ai",
        }
        create_schema = AgentWorkflowCreate(**data)
        assert create_schema.recipient_email == "recipient@capitaledge.ai"
        assert create_schema.user_email == "recipient@capitaledge.ai"

    def test_update_schema_email_fields(self):
        update_schema = AgentWorkflowUpdate(user_email="updated@capitaledge.ai")
        assert update_schema.recipient_email == "updated@capitaledge.ai"
        assert update_schema.user_email == "updated@capitaledge.ai"

    def test_list_item_and_response_schema_contain_email(self):
        response_schema = AgentWorkflowResponse(
            id="agent-1",
            user_id="user-1",
            name="Test Agent",
            trigger_type=TriggerType.SCHEDULED,
            recipient_email="test@capitaledge.ai",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        assert response_schema.recipient_email == "test@capitaledge.ai"

        list_item = AgentWorkflowListItem(
            id="agent-1",
            name="Test Agent",
            trigger_type=TriggerType.SCHEDULED,
            execution_mode="auto",
            status=AgentStatus.ACTIVE,
            version="1.0.0",
            recipient_email="test@capitaledge.ai",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        assert list_item.recipient_email == "test@capitaledge.ai"


class TestAgentWorkflowCreationEmailCapture:
    """Test capturing logged-in user email on agent creation in CRUD and model."""

    def test_create_agent_captures_user_email_automatically(self, db_session, test_user):
        crud_agent = CRUDAgentWorkflow(AgentWorkflow)
        create_in = AgentWorkflowCreate(
            name="Morning Digest Agent",
            instructions="Summarize top morning news",
            trigger_type=TriggerType.SCHEDULED,
            schedule_frequency="daily",
            schedule_time="08:00",
        )

        agent = crud_agent.create_agent(
            db=db_session,
            user_id=test_user.id,
            obj_in=create_in,
            status=AgentStatus.ACTIVE.value,
        )

        assert agent.recipient_email == test_user.email
        assert agent.effective_recipient_email == test_user.email
        assert agent.user_email == test_user.email

    def test_create_agent_with_explicit_recipient_email(self, db_session, test_user):
        crud_agent = CRUDAgentWorkflow(AgentWorkflow)
        create_in = AgentWorkflowCreate(
            name="Executive Briefing",
            instructions="Prepare executive summary",
            trigger_type=TriggerType.SCHEDULED,
            schedule_frequency="weekly",
            schedule_day="monday",
            schedule_time="09:00",
            recipient_email="executive-team@capitaledge.ai",
        )

        agent = crud_agent.create_agent(
            db=db_session,
            user_id=test_user.id,
            obj_in=create_in,
            status=AgentStatus.ACTIVE.value,
        )

        assert agent.recipient_email == "executive-team@capitaledge.ai"
        assert agent.effective_recipient_email == "executive-team@capitaledge.ai"


class TestScheduledExecutionEmailDelivery:
    """Test that scheduled executions dispatch result emails to associated user."""

    def test_send_agent_workflow_email_success(self):
        async def _test():
            with patch("server.utils.mail.FastMail.send_message", new_callable=AsyncMock) as mock_send:
                success = await send_agent_workflow_email(
                    email="analyst@capitaledge.ai",
                    agent_name="Morning Digest",
                    result_content="Today's financial news is positive.",
                )
                assert success is True
                mock_send.assert_called_once()
                call_args = mock_send.call_args[0]
                message = call_args[0]
                assert any("analyst@capitaledge.ai" in str(r) for r in message.recipients)
                assert "Morning Digest" in message.subject

        asyncio.run(_test())

    def test_send_agent_workflow_email_invalid_email_safe(self):
        async def _test():
            with patch("server.utils.mail.FastMail.send_message", new_callable=AsyncMock) as mock_send:
                success = await send_agent_workflow_email(
                    email="not-an-email",
                    agent_name="Morning Digest",
                    result_content="Content",
                )
                assert success is False
                mock_send.assert_not_called()

        asyncio.run(_test())

    def test_scheduled_execution_triggers_email_dispatch(self, db_session, test_user):
        async def _test():
            crud_agent = CRUDAgentWorkflow(AgentWorkflow)
            agent = crud_agent.create_agent(
                db=db_session,
                user_id=test_user.id,
                obj_in=AgentWorkflowCreate(
                    name="Automated Analyst",
                    instructions="Analyze market filings",
                    trigger_type=TriggerType.SCHEDULED,
                    schedule_frequency="daily",
                    schedule_time="09:00",
                    recipient_email=test_user.email,
                ),
                status=AgentStatus.ACTIVE.value,
            )

            mock_engine = MagicMock()
            mock_response = AgentWorkflowExecutionResponse(
                id="exec-123",
                workflow_id=agent.id,
                agent_id=agent.id,
                agent_name=agent.name,
                workflow_name=agent.name,
                user_id=test_user.id,
                trigger_source="scheduled",
                query_or_goal="Analyze market filings",
                status=ExecutionStatus.COMPLETED.value,
                final_response="## Report\nMarket analysis concluded successfully with positive outlook.",
                source_documents=["Filing_10K.pdf"],
                citations=[],
                skills_applied=[],
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                created_at=datetime.utcnow(),
                execution_time_ms=1200,
            )
            mock_engine.execute_workflow = AsyncMock(return_value=mock_response)

            exec_service = AgentExecutionService(workflow_engine=mock_engine)

            with patch("server.utils.mail.send_agent_workflow_email", new_callable=AsyncMock) as mock_email:
                mock_email.return_value = True

                result = await exec_service.execute(
                    agent_id=agent.id,
                    trigger_source="scheduled",
                    db=db_session,
                )

                assert result is not None
                assert result.status == ExecutionStatus.COMPLETED.value
                mock_email.assert_called_once()
                _, kwargs = mock_email.call_args
                assert kwargs["email"] == test_user.email
                assert kwargs["agent_name"] == agent.name
                assert "Market analysis concluded successfully" in kwargs["result_content"]
                assert kwargs["sources"] == ["Filing_10K.pdf"]

        asyncio.run(_test())

    def test_scheduled_execution_resilient_to_email_failure(self, db_session, test_user):
        """Even if SMTP / email fails, agent execution completes successfully."""
        async def _test():
            crud_agent = CRUDAgentWorkflow(AgentWorkflow)
            agent = crud_agent.create_agent(
                db=db_session,
                user_id=test_user.id,
                obj_in=AgentWorkflowCreate(
                    name="Resilient Agent",
                    instructions="Process data",
                    trigger_type=TriggerType.SCHEDULED,
                    recipient_email="valid@capitaledge.ai",
                ),
                status=AgentStatus.ACTIVE.value,
            )

            mock_engine = MagicMock()
            mock_response = AgentWorkflowExecutionResponse(
                id="exec-456",
                workflow_id=agent.id,
                agent_id=agent.id,
                agent_name=agent.name,
                workflow_name=agent.name,
                user_id=test_user.id,
                trigger_source="scheduled",
                query_or_goal="Process data",
                status=ExecutionStatus.COMPLETED.value,
                final_response="Data processed successfully.",
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                created_at=datetime.utcnow(),
                execution_time_ms=800,
            )
            mock_engine.execute_workflow = AsyncMock(return_value=mock_response)
            exec_service = AgentExecutionService(workflow_engine=mock_engine)

            with patch("server.utils.mail.send_agent_workflow_email", new_callable=AsyncMock) as mock_email:
                mock_email.side_effect = Exception("SMTP Server Connection Timeout")

                result = await exec_service.execute(
                    agent_id=agent.id,
                    trigger_source="scheduled",
                    db=db_session,
                )

                # Execution must still succeed and return result
                assert result is not None
                assert result.status == ExecutionStatus.COMPLETED.value
                assert result.final_response == "Data processed successfully."
                mock_email.assert_called_once()

        asyncio.run(_test())
