"""Landing Page Default Agents Seed Service.

Provides the canonical seed and default template configurations for CapitalEdge Landing Page Agents.
Guarantees exact agent names, instructions, descriptions, tags, and icon mappings.
"""

from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

DEFAULT_LANDING_AGENTS: List[Dict[str, Any]] = [
    {
        "id": "market_primer",
        "name": "Market Primer",
        "title": "Market Primer",
        "description": "Conduct a deep-dive on the Indian quick-commerce market covering size, growth, segments, buying behaviour, and the top five players.",
        "prompt": "Conduct a deep-dive on the Indian quick-commerce market covering size, growth, segments, buying behaviour, and the top five players.",
        "instructions": "Conduct a deep-dive on the Indian quick-commerce market covering size, growth, segments, buying behaviour, and the top five players.",
        "tag": "Market Research",
        "icon": "trending-up",
        "category": "market_research",
        "display_order": 1,
        "is_default": True,
        "is_system_template": True,
    },
    {
        "id": "management_meter",
        "name": "Management Meter",
        "title": "Management Meter",
        "description": "Assess Meta management credibility by comparing capex and Reality Labs commitments against delivered outcomes over the last three years.",
        "prompt": "Assess Meta management credibility by comparing capex and Reality Labs commitments against delivered outcomes over the last three years.",
        "instructions": "Assess Meta management credibility by comparing capex and Reality Labs commitments against delivered outcomes over the last three years.",
        "tag": "Governance",
        "icon": "users",
        "category": "governance",
        "display_order": 2,
        "is_default": True,
        "is_system_template": True,
    },
    {
        "id": "deal_diligence",
        "name": "Deal Diligence",
        "title": "Deal Diligence",
        "description": "Run a pre-deal review of the Microsoft–Activision style acquisition across valuation, synergies, structure, and integration risk.",
        "prompt": "Run a pre-deal review of the Microsoft–Activision style acquisition across valuation, synergies, structure, and integration risk.",
        "instructions": "Run a pre-deal review of the Microsoft–Activision style acquisition across valuation, synergies, structure, and integration risk.",
        "tag": "M&A Diligence",
        "icon": "briefcase",
        "category": "ma_diligence",
        "display_order": 3,
        "is_default": True,
        "is_system_template": True,
    },
    {
        "id": "unit_economics_lab",
        "name": "Unit Economics Lab",
        "title": "Unit Economics Lab",
        "description": "Break Zomato and Swiggy down to per-order economics — AOV, contribution margin, CAC, payback — and test durability.",
        "prompt": "Break Zomato and Swiggy down to per-order economics — AOV, contribution margin, CAC, payback — and test durability.",
        "instructions": "Break Zomato and Swiggy down to per-order economics — AOV, contribution margin, CAC, payback — and test durability.",
        "tag": "Unit Economics",
        "icon": "pie-chart",
        "category": "financial_analysis",
        "display_order": 4,
        "is_default": True,
        "is_system_template": True,
    },
    {
        "id": "sector_scan",
        "name": "Sector Scan",
        "title": "Sector Scan",
        "description": "Map the Indian NBFC sector's structure, growth pools, margin profile, and regulatory pressure points, and rank where value is migrating.",
        "prompt": "Map the Indian NBFC sector's structure, growth pools, margin profile, and regulatory pressure points, and rank where value is migrating.",
        "instructions": "Map the Indian NBFC sector's structure, growth pools, margin profile, and regulatory pressure points, and rank where value is migrating.",
        "tag": "Sector Mapping",
        "icon": "compass",
        "category": "sector_analysis",
        "display_order": 5,
        "is_default": True,
        "is_system_template": True,
    },
    {
        "id": "brand_health_check",
        "name": "Brand Health Check",
        "title": "Brand Health Check",
        "description": "Assess Nike's brand health on awareness, consideration, sentiment, and share of voice versus Adidas and On Running.",
        "prompt": "Assess Nike's brand health on awareness, consideration, sentiment, and share of voice versus Adidas and On Running.",
        "instructions": "Assess Nike's brand health on awareness, consideration, sentiment, and share of voice versus Adidas and On Running.",
        "tag": "Brand Analytics",
        "icon": "activity",
        "category": "brand_analytics",
        "display_order": 6,
        "is_default": True,
        "is_system_template": True,
    },
    {
        "id": "win_loss_review",
        "name": "Win Loss Review",
        "title": "Win Loss Review",
        "description": "Analyse won and lost deals against Salesforce and HubSpot to extract why we win, why we lose, and which objections repeat.",
        "prompt": "Analyse won and lost deals against Salesforce and HubSpot to extract why we win, why we lose, and which objections repeat.",
        "instructions": "Analyse won and lost deals against Salesforce and HubSpot to extract why we win, why we lose, and which objections repeat.",
        "tag": "Sales Intelligence",
        "icon": "award",
        "category": "sales_intelligence",
        "display_order": 7,
        "is_default": True,
        "is_system_template": True,
    },
    {
        "id": "voice_of_customer",
        "name": "Voice of Customer",
        "title": "Voice of Customer",
        "description": "Mine app-store reviews, surveys, and support tickets for Swiggy to surface recurring themes, sentiment shifts, and unmet needs.",
        "prompt": "Mine app-store reviews, surveys, and support tickets for Swiggy to surface recurring themes, sentiment shifts, and unmet needs.",
        "instructions": "Mine app-store reviews, surveys, and support tickets for Swiggy to surface recurring themes, sentiment shifts, and unmet needs.",
        "tag": "Customer Insights",
        "icon": "message-square",
        "category": "customer_research",
        "display_order": 8,
        "is_default": True,
        "is_system_template": True,
    },
    {
        "id": "persona_pro",
        "name": "Persona Pro",
        "title": "Persona Pro",
        "description": "Build evidence-backed personas for AWS enterprise buyers covering goals, pain points, buying triggers, objections, and channels.",
        "prompt": "Build evidence-backed personas for AWS enterprise buyers covering goals, pain points, buying triggers, objections, and channels.",
        "instructions": "Build evidence-backed personas for AWS enterprise buyers covering goals, pain points, buying triggers, objections, and channels.",
        "tag": "Persona Building",
        "icon": "user-check",
        "category": "persona_building",
        "display_order": 9,
        "is_default": True,
        "is_system_template": True,
    },
    {
        "id": "launch_lab",
        "name": "Launch Lab",
        "title": "Launch Lab",
        "description": "Build a go-to-market launch plan for OpenAI's newest enterprise product covering positioning, enablement, channels, and launch-day checklist.",
        "prompt": "Build a go-to-market launch plan for OpenAI's newest enterprise product covering positioning, enablement, channels, and launch-day checklist.",
        "instructions": "Build a go-to-market launch plan for OpenAI's newest enterprise product covering positioning, enablement, channels, and launch-day checklist.",
        "tag": "GTM Launch",
        "icon": "rocket",
        "category": "gtm_strategy",
        "display_order": 10,
        "is_default": True,
        "is_system_template": True,
    },
]

DEFAULT_AGENTS_BY_ID = {agent["id"]: agent for agent in DEFAULT_LANDING_AGENTS}


class AgentSeedService:
    """Service to supply default landing page agents and workflow templates."""

    @classmethod
    def get_default_landing_agents(cls) -> List[Dict[str, Any]]:
        """Return the 10 canonical Landing Page default agents sorted by display order.

        Dynamically synchronizes with PromptTemplateRegistry (manifest.json + prompt.md)
        as the single canonical source of truth.
        """
        try:
            from server.services.prompt_template_registry import get_prompt_template_registry
            registry = get_prompt_template_registry()
            templates = registry.get_all()
            if templates and len(templates) == 10:
                dynamic_agents = []
                for t in templates:
                    dynamic_agents.append({
                        "id": t.id,
                        "name": t.title,
                        "title": t.title,
                        "description": t.description,
                        "prompt": t.prompt,
                        "instructions": t.prompt,
                        "tag": t.tag,
                        "icon": t.icon,
                        "category": t.manifest.get("category", t.id),
                        "display_order": t.manifest.get("display_order", 999),
                        "is_default": True,
                        "is_system_template": True,
                    })
                return sorted(dynamic_agents, key=lambda a: a.get("display_order", 999))
        except Exception:
            pass

        return sorted(DEFAULT_LANDING_AGENTS, key=lambda a: a.get("display_order", 999))

    @classmethod
    def get_agent_by_id(cls, agent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific default agent by ID."""
        agents = cls.get_default_landing_agents()
        target_id = agent_id.strip().lower()
        for a in agents:
            if a["id"] == target_id:
                return a
        return DEFAULT_AGENTS_BY_ID.get(target_id)


    @classmethod
    def seed_default_agents_for_user(cls, db, user_id: str) -> List[Any]:
        """Idempotently seed default agents for a user.

        Ensures:
        - Never overwrites existing user agents.
        - Never creates duplicate agents when executed multiple times.
        - Preserves canonical display_order (1-10).
        """
        from server import crud
        from server.models.agent_workflow import AgentStatus, TriggerType

        created_agents = []
        for agent_data in cls.get_default_landing_agents():
            existing = crud.agent_workflow.get_by_user_and_name(
                db=db,
                user_id=str(user_id),
                name=agent_data["name"],
            )
            if existing:
                continue

            agent_obj = crud.agent_workflow.create_agent(
                db=db,
                user_id=str(user_id),
                obj_in={
                    "name": agent_data["name"],
                    "description": agent_data["description"],
                    "instructions": agent_data["instructions"],
                    "status": AgentStatus.DRAFT.value,
                    "trigger_type": TriggerType.AUTOMATIC.value,
                    "execution_mode": "auto",
                    "is_system_template": True,
                },
            )
            created_agents.append(agent_obj)

        return created_agents


