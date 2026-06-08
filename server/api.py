from fastapi import APIRouter

from server.api_v1.endpoints.genai import ai_router
from server.api_v1.endpoints.user import user_router

api_router = APIRouter()

api_router.include_router(user_router, include_in_schema=True)
api_router.include_router(ai_router, include_in_schema=True)
