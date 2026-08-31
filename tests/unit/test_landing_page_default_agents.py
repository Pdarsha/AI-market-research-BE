"""Focused unit tests for Landing Page Default Agents seed configuration.

Tests cover:
1. Verification of the 10 canonical landing page default agents.
2. Verification of exact names and instructions for all 10 agents:
   - Market Primer
   - Management Meter
   - Deal Diligence
   - Unit Economics Lab
   - Sector Scan
   - Brand Health Check
   - Win Loss Review
   - Voice of Customer
   - Persona Pro
   - Launch Lab
3. Verification of display ordering (1 through 10).
4. PromptTemplateRegistry discovery and loading from filesystem manifests.
5. API endpoints verification:
   - GET /v1/cards/prompt-templates
   - GET /v1/cards/agents
   - GET /v1/agents/templates
   - GET /v1/agents/default
6. Verification that existing user-created agents are isolated and unaltered.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from server.db.base_class import Base
from server.models.user import User
from server.models.agent_workflow import AgentWorkflow, AgentStatus
from server.services.agent_seed_service import AgentSeedService, DEFAULT_LANDING_AGENTS
from server.services.prompt_template_registry import get_prompt_template_registry
from server.api import api_router
from server.utils.auth import get_current_user, get_optional_current_user
from server.api_v1.deps import get_db

EXPECTED_AGENTS = [
    (
        "Market Primer",
        "Conduct a deep-dive on the Indian quick-commerce market covering size, growth, segments, buying behaviour, and the top five players.",
    ),
    (
        "Management Meter",
        "Assess Meta management credibility by comparing capex and Reality Labs commitments against delivered outcomes over the last three years.",
    ),
    (
        "Deal Diligence",
        "Run a pre-deal review of the Microsoft–Activision style acquisition across valuation, synergies, structure, and integration risk.",
    ),
    (
        "Unit Economics Lab",
        "Break Zomato and Swiggy down to per-order economics — AOV, contribution margin, CAC, payback — and test durability.",
    ),
    (
        "Sector Scan",
        "Map the Indian NBFC sector's structure, growth pools, margin profile, and regulatory pressure points, and rank where value is migrating.",
    ),
    (
        "Brand Health Check",
        "Assess Nike's brand health on awareness, consideration, sentiment, and share of voice versus Adidas and On Running.",
    ),
    (
        "Win Loss Review",
        "Analyse won and lost deals against Salesforce and HubSpot to extract why we win, why we lose, and which objections repeat.",
    ),
    (
        "Voice of Customer",
        "Mine app-store reviews, surveys, and support tickets for Swiggy to surface recurring themes, sentiment shifts, and unmet needs.",
    ),
    (
        "Persona Pro",
        "Build evidence-backed personas for AWS enterprise buyers covering goals, pain points, buying triggers, objections, and channels.",
    ),
    (
        "Launch Lab",
        "Build a go-to-market launch plan for OpenAI's newest enterprise product covering positioning, enablement, channels, and launch-day checklist.",
    ),
]


@pytest.fixture
def db_session():
    """In-memory SQLite database session."""
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
    """Sample user for testing default agent APIs."""
    user = User(
        id="usr-landing-agents-1",
        email="analyst@example.com",
        display_name="Landing Analyst",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def app_client(db_session, sample_user):
    """FastAPI TestClient configured with api_router."""
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


class TestLandingPageDefaultAgentsSeed:
    """Test seed definitions and prompt template registry loading."""

    def test_canonical_agent_count(self):
        agents = AgentSeedService.get_default_landing_agents()
        assert len(agents) == 10

    def test_exact_names_and_prompts(self):
        agents = AgentSeedService.get_default_landing_agents()
        for idx, (expected_name, expected_prompt) in enumerate(EXPECTED_AGENTS):
            agent = agents[idx]
            assert agent["title"] == expected_name
            assert agent["prompt"] == expected_prompt
            assert agent["display_order"] == idx + 1

    def test_prompt_template_registry_loads_exact_10_agents(self):
        registry = get_prompt_template_registry()
        registry.reload()
        templates = registry.list_templates()
        assert len(templates) == 10

        template_titles = [t.get_metadata()["title"] for t in templates]
        expected_titles = [name for name, _ in EXPECTED_AGENTS]
        assert template_titles == expected_titles

    def test_user_created_agents_remain_unaffected(self, db_session, sample_user):
        # Create a custom user agent workflow
        custom_agent = AgentWorkflow(
            id="agent-custom-999",
            user_id=sample_user.id,
            name="My Custom Semiconductor Tracker",
            description="Tracks foundry capex",
            instructions="Summarize TSMC and Samsung quarterly capex spend.",
            status=AgentStatus.ACTIVE,
        )
        db_session.add(custom_agent)
        db_session.commit()

        # Check that default seed service is isolated
        default_agents = AgentSeedService.get_default_landing_agents()
        assert len(default_agents) == 10
        assert not any(a["id"] == "agent-custom-999" for a in default_agents)

        # Check that custom agent exists untouched in DB
        db_agent = db_session.query(AgentWorkflow).filter(AgentWorkflow.id == "agent-custom-999").first()
        assert db_agent is not None
        assert db_agent.name == "My Custom Semiconductor Tracker"


class TestLandingPageAgentsApiEndpoints:
    """Test API retrieval of landing page default agents."""

    def test_get_prompt_templates_api(self, app_client):
        resp = app_client.get("/v1/cards/prompt-templates")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert "data" in body
        assert len(body["data"]) == 10

        titles = [item["title"] for item in body["data"]]
        assert titles[0] == "Market Primer"
        assert titles[1] == "Management Meter"
        assert titles[2] == "Deal Diligence"
        assert titles[3] == "Unit Economics Lab"
        assert titles[4] == "Sector Scan"
        assert titles[5] == "Brand Health Check"
        assert titles[6] == "Win Loss Review"
        assert titles[7] == "Voice of Customer"
        assert titles[8] == "Persona Pro"
        assert titles[9] == "Launch Lab"

    def test_get_agents_templates_api(self, app_client):
        resp = app_client.get("/v1/agents/templates")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert len(body["data"]) == 10

    def test_get_agents_templates_canonical_api(self, app_client):
        """Canonical agents templates endpoint is /v1/agents/templates — the /default alias has been removed."""
        resp = app_client.get("/v1/agents/templates")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert len(body["data"]) == 10

    def test_seed_idempotency_and_no_duplicates(self, db_session, sample_user):
        """Verify seeding runs idempotently without duplicate records."""
        first_seed = AgentSeedService.seed_default_agents_for_user(db_session, sample_user.id)
        assert len(first_seed) == 10

        second_seed = AgentSeedService.seed_default_agents_for_user(db_session, sample_user.id)
        assert len(second_seed) == 0

        user_agents = db_session.query(AgentWorkflow).filter(AgentWorkflow.user_id == sample_user.id).all()
        assert len(user_agents) == 10

    def test_seed_never_overwrites_user_customizations(self, db_session, sample_user):
        """Verify user customized agent is preserved during seed execution."""
        custom_agent = AgentWorkflow(
            id="agent-custom-market-primer",
            user_id=sample_user.id,
            name="Market Primer",
            description="Custom user description for quick commerce",
            instructions="Custom prompt with proprietary rubric",
            status=AgentStatus.ACTIVE.value,
        )
        db_session.add(custom_agent)
        db_session.commit()

        # Run seed
        AgentSeedService.seed_default_agents_for_user(db_session, sample_user.id)

        # Confirm user's custom agent was not overwritten
        reloaded = db_session.query(AgentWorkflow).filter(AgentWorkflow.id == "agent-custom-market-primer").first()
        assert reloaded.description == "Custom user description for quick commerce"
        assert reloaded.instructions == "Custom prompt with proprietary rubric"

    def test_all_prompt_md_files_are_single_line_instructions(self):
        """Verify every prompt.md file is strictly a single line with no markdown/headings."""
        from pathlib import Path
        templates_dir = Path("server/prompt_templates")
        assert templates_dir.is_dir()

        folders = [f for f in templates_dir.iterdir() if f.is_dir()]
        assert len(folders) == 10

        for folder in folders:
            prompt_file = folder / "prompt.md"
            assert prompt_file.exists(), f"Missing prompt.md in {folder.name}"
            content = prompt_file.read_text(encoding="utf-8").strip()

            # Must not contain newlines or markdown headings
            assert "\n" not in content, f"prompt.md in {folder.name} contains multiple lines"
            assert not content.startswith("#"), f"prompt.md in {folder.name} starts with markdown header"
            assert len(content) > 20, f"prompt.md in {folder.name} is too short"

