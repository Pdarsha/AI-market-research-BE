import logging
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from server.services.prompt_template_registry import (
    PromptTemplateRegistry,
    get_prompt_template_registry,
)
from server.services.agent_seed_service import AgentSeedService
from server.services.audit_logger import AuditLogger

logger = logging.getLogger(__name__)

prompt_templates_router = APIRouter(prefix="/cards", tags=["Prompt Templates"])


@prompt_templates_router.get("/prompt-templates")
def list_prompt_templates(
    registry: PromptTemplateRegistry = Depends(get_prompt_template_registry),
):
    """Retrieve all active, valid landing page default agent prompt templates."""
    templates = registry.list_templates()
    if templates:
        data = [t.get_metadata() for t in templates]
    else:
        # Fallback to canonical seed definitions if filesystem cache was empty
        data = AgentSeedService.get_default_landing_agents()

    AuditLogger.log(
        action="PROMPT_TEMPLATES_LIST",
        endpoint="/v1/cards/prompt-templates",
        http_method="GET",
        status="200",
        metadata={"count": len(data)},
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": data,
            "message": "Prompt templates fetched successfully.",
        },
    )

