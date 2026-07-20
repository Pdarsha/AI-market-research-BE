from fastapi import APIRouter

from server.api_v1.endpoints.documents import citations_router, documents_router
from server.api_v1.endpoints.genai import ai_router
from server.api_v1.endpoints.user import user_router
from server.api_v1.endpoints.workspace import workspace_router
from server.api_v1.endpoints.prompt_templates import prompt_templates_router

api_router = APIRouter()

api_router.include_router(user_router, include_in_schema=True)
api_router.include_router(ai_router, include_in_schema=True)
api_router.include_router(workspace_router, include_in_schema=True)
api_router.include_router(documents_router, include_in_schema=True)
api_router.include_router(citations_router, include_in_schema=True)
api_router.include_router(prompt_templates_router, include_in_schema=True)
