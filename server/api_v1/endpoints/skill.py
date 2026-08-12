import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from httpx2 import query

from server import crud, schemas
from server.api_v1.deps import get_db
from server.services.audit_logger import AuditLogger
from server.services.skill_generation_service import (
    SkillGenerationError,
    skill_generation_service,
)
from server.services.skill_validation_service import (
    SkillValidationError,
    skill_validation_service,
)
from server.services.skill_suggestion_registry import (
    SkillSuggestionRegistry,
    get_skill_suggestion_registry,
)
from server.utils.auth import get_current_user, get_optional_current_user

logger = logging.getLogger(__name__)

skill_router = APIRouter(prefix="/skills", tags=["Skills"])


@skill_router.get("/suggestions")
def get_skill_suggestions(
    registry: SkillSuggestionRegistry = Depends(get_skill_suggestion_registry),
    current_user=Depends(get_optional_current_user),
):
    """Retrieve all active skill generation prompt suggestions.

    Returns suggestions ordered by display_order.
    """
    user_id = str(current_user.id) if current_user else "anonymous"
    suggestions = registry.list_suggestions()
    data = [s.get_metadata() for s in suggestions]

    AuditLogger.log(
        action="SKILL_SUGGESTIONS_LIST",
        user_id=user_id,
        endpoint="/v1/skills/suggestions",
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
            "message": "Skill suggestions fetched successfully.",
        },
    )


@skill_router.post("/generate")
def generate_skill_draft(
    request: schemas.SkillDraftRequest,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Generate a Skill draft from a natural language request.

    This endpoint uses AI to convert a user's natural language description
    into a structured Skill draft containing:
    - name: Concise skill name
    - description: What the skill does
    - trigger_description: When the skill should activate
    - instructions: Detailed operational steps

    The draft is returned but NOT persisted to the Skill Library.
    The user can review and then save it via POST /skills.
    """
    user_id = str(current_user.id)

    try:
        draft = skill_generation_service.generate_draft(
            request.request,
            user_id=user_id,
            db=db,
        )

        AuditLogger.log(
            action="SKILL_DRAFT_GENERATE",
            user_id=user_id,
            endpoint="/v1/skills/generate",
            http_method="POST",
            status="200",
            metadata={
                "request_length": len(request.request),
                "generated_name": draft.get("name"),
            },
        )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "error": None,
                "data": draft,
                "message": "Skill draft generated successfully.",
            },
        )

    except SkillGenerationError as e:
        logger.warning(
            f"Skill generation failed for user {user_id}: {e.message} (code: {e.code})"
        )

        AuditLogger.log(
            action="SKILL_DRAFT_GENERATE",
            user_id=user_id,
            endpoint="/v1/skills/generate",
            http_method="POST",
            status=(
                "400"
                if e.code in ("EMPTY_REQUEST", "REQUEST_TOO_LONG", "VALIDATION_FAILED")
                else "500"
            ),
            metadata={
                "error_code": e.code,
                "error_message": e.message,
            },
        )

        # Map error codes to HTTP status codes
        status_code_map = {
            "EMPTY_REQUEST": status.HTTP_400_BAD_REQUEST,
            "REQUEST_TOO_LONG": status.HTTP_400_BAD_REQUEST,
            "VALIDATION_FAILED": status.HTTP_400_BAD_REQUEST,
            "MALFORMED_RESPONSE": status.HTTP_502_BAD_GATEWAY,
            "EMPTY_RESPONSE": status.HTTP_502_BAD_GATEWAY,
            "LLM_ERROR": status.HTTP_502_BAD_GATEWAY,
            "GENERATION_FAILED": status.HTTP_500_INTERNAL_SERVER_ERROR,
        }

        http_status = status_code_map.get(e.code, status.HTTP_500_INTERNAL_SERVER_ERROR)

        return JSONResponse(
            status_code=http_status,
            content={
                "success": False,
                "error": {
                    "code": e.code,
                    "message": e.message,
                    "details": e.details,
                },
                "data": None,
                "message": e.message,
            },
        )

    except Exception as e:
        logger.exception(
            f"Unexpected error in skill draft generation for user {user_id}: {e}"
        )

        AuditLogger.log(
            action="SKILL_DRAFT_GENERATE",
            user_id=user_id,
            endpoint="/v1/skills/generate",
            http_method="POST",
            status="500",
            metadata={
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred during skill draft generation",
                    "details": {"error_type": type(e).__name__},
                },
                "data": None,
                "message": "Internal server error",
            },
        )


@skill_router.post("/validate")
def validate_skill_draft(
    request: schemas.SkillValidationRequest,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Validate a Skill draft before saving to the Skill Library.

    Performs deterministic structural validation including:
    - Required fields present and non-empty
    - Field length limits
    - Trigger description specificity
    - Instructions quality and structure
    - Name uniqueness (if check_uniqueness=true, excluding current draft ID if editing)

    Optionally includes AI-based quality assessment (if include_ai_quality=true).

    Does NOT save or publish the skill. Returns validation result with errors and warnings.
    """
    user_id = str(current_user.id)

    try:
        # Convert draft to dict for the service
        draft_dict = request.draft.model_dump()
        draft_id = draft_dict.get("id")

        validation_result = skill_validation_service.validate_draft(
            draft=draft_dict,
            user_id=user_id if request.check_uniqueness else None,
            check_uniqueness=request.check_uniqueness,
            include_ai_quality=request.include_ai_quality,
            exclude_skill_id=draft_id,
        )

        AuditLogger.log(
            action="SKILL_DRAFT_VALIDATE",
            user_id=user_id,
            endpoint="/v1/skills/validate",
            http_method="POST",
            status="200",
            metadata={
                "valid": validation_result.is_valid,
                "error_count": len(validation_result.errors),
                "warning_count": len(validation_result.warnings),
                "check_uniqueness": request.check_uniqueness,
                "include_ai_quality": request.include_ai_quality,
            },
        )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "error": None,
                "data": validation_result.to_dict(),
                "message": (
                    "Validation complete."
                    if validation_result.is_valid
                    else "Validation failed."
                ),
            },
        )

    except SkillValidationError as e:
        logger.warning(
            f"Skill validation failed for user {user_id}: {e.message} (code: {e.code})"
        )

        AuditLogger.log(
            action="SKILL_DRAFT_VALIDATE",
            user_id=user_id,
            endpoint="/v1/skills/validate",
            http_method="POST",
            status="400",
            metadata={
                "error_code": e.code,
                "error_message": e.message,
            },
        )

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "code": e.code,
                    "message": e.message,
                    "details": e.details,
                },
                "data": None,
                "message": e.message,
            },
        )

    except Exception as e:
        logger.exception(
            f"Unexpected error in skill draft validation for user {user_id}: {e}"
        )

        AuditLogger.log(
            action="SKILL_DRAFT_VALIDATE",
            user_id=user_id,
            endpoint="/v1/skills/validate",
            http_method="POST",
            status="500",
            metadata={
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred during skill draft validation",
                    "details": {"error_type": type(e).__name__},
                },
                "data": None,
                "message": "Internal server error",
            },
        )


@skill_router.post("")
def save_skill(
    request: schemas.SkillSaveRequest,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Save a validated Skill draft to the Skill Library.

    Creates a new Skill record with the provided draft content if no existing ID is supplied.
    If an existing Skill UUID is provided, updates that exact record preserving its UUID.

    Flow: Draft → Validate → Save (status=draft)
    """
    user_id = str(current_user.id)

    try:
        draft_dict = request.draft.model_dump()
        draft_id = draft_dict.get("id")

        # Check if updating an existing skill record by ID
        existing_skill = None
        if draft_id:
            existing_skill = crud.skill.get_by_user_and_id(
                db, user_id=user_id, skill_id=draft_id
            )

        # For new skill saves (without existing skill ID), ensure unique name
        if not existing_skill and request.draft.name:
            unique_name = crud.skill.get_unique_skill_name(
                db, user_id=user_id, base_name=request.draft.name
            )
            draft_dict["name"] = unique_name
            request.draft.name = unique_name

        # Validate the draft, excluding the existing skill ID from uniqueness check if updating
        validation_result = skill_validation_service.validate_draft(
            draft=draft_dict,
            user_id=user_id,
            check_uniqueness=True,
            include_ai_quality=False,
            exclude_skill_id=existing_skill.id if existing_skill else draft_id,
        )

        if not validation_result.is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error": {
                        "code": "VALIDATION_FAILED",
                        "message": "Skill draft validation failed",
                        "details": {"errors": validation_result.errors},
                    },
                    "data": None,
                    "message": "Cannot save invalid skill draft",
                },
            )

        if existing_skill:
            # Update existing skill record (preserving UUID)
            skill_update = schemas.SkillUpdate(
                name=request.draft.name,
                description=request.draft.description,
                trigger_description=request.draft.trigger_description,
                instructions=request.draft.instructions,
            )
            skill = crud.skill.update(db, db_obj=existing_skill, obj_in=skill_update)
            msg = "Skill draft updated successfully."
            res_status = status.HTTP_200_OK
        else:
            # Create new skill record
            skill_create = schemas.SkillCreate(
                name=request.draft.name,
                description=request.draft.description,
                trigger_description=request.draft.trigger_description,
                instructions=request.draft.instructions,
            )
            skill = crud.skill.create_skill(
                db=db,
                user_id=user_id,
                obj_in=skill_create,
                status=request.status or "draft",
            )
            msg = "Skill saved successfully."
            res_status = status.HTTP_201_CREATED

        AuditLogger.log(
            action="SKILL_SAVE",
            user_id=user_id,
            endpoint="/v1/skills",
            http_method="POST",
            status=str(res_status),
            metadata={
                "skill_id": skill.id,
                "skill_name": skill.name,
                "status": skill.status,
            },
        )

        return JSONResponse(
            status_code=res_status,
            content={
                "success": True,
                "error": None,
                "data": schemas.SkillResponse.model_validate(skill).model_dump(
                    mode="json"
                ),
                "message": msg,
            },
        )

    except ValueError as e:
        logger.warning(f"Skill save failed for user {user_id}: {e}")

        AuditLogger.log(
            action="SKILL_SAVE",
            user_id=user_id,
            endpoint="/v1/skills",
            http_method="POST",
            status="400",
            metadata={"error_message": str(e)},
        )

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "code": "DUPLICATE_NAME",
                    "message": str(e),
                },
                "data": None,
                "message": str(e),
            },
        )

    except Exception as e:
        logger.exception(f"Unexpected error in skill save for user {user_id}: {e}")

        AuditLogger.log(
            action="SKILL_SAVE",
            user_id=user_id,
            endpoint="/v1/skills",
            http_method="POST",
            status="500",
            metadata={
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred during skill save",
                    "details": {"error_type": type(e).__name__},
                },
                "data": None,
                "message": "Internal server error",
            },
        )


@skill_router.get("")
def get_skills(
    skip: int = 0,
    limit: int = 100,
    status: str = None,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Get all saved Skills belonging to the authenticated user.

    Returns both draft and published skills.
    Does not return temporary generated drafts (they don't exist in the database).
    """
    user_id = str(current_user.id)

    skills = crud.skill.get_multi_by_user(
        db, user_id=user_id, skip=skip, limit=limit, status=status
    )
    serialized_skills = [
        schemas.SkillResponse.model_validate(s).model_dump(mode="json") for s in skills
    ]

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": serialized_skills,
            "message": "Skills retrieved successfully.",
        },
    )


@skill_router.get("/{skill_id}")
def get_skill(
    skill_id: str,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Get a specific saved Skill by ID.

    Verifies ownership - returns only the user's Skill.
    Returns complete saved Skill data.
    Returns 404 when missing or not owned.

    This endpoint works ONLY with database-backed Skills.
    It must never attempt to retrieve an unsaved generated draft.
    """
    user_id = str(current_user.id)

    skill = crud.skill.get(db, skill_id)

    if not skill or skill.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Skill not found"
        )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": schemas.SkillResponse.model_validate(skill).model_dump(mode="json"),
            "message": "Skill retrieved successfully.",
        },
    )


@skill_router.patch("/{skill_id}")
def update_skill(
    skill_id: str,
    request: schemas.SkillUpdate,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Update a saved Skill.

    Allows updating: name, description, trigger_description, instructions
    - Authenticates user
    - Verifies ownership
    - Updates only allowed fields
    - Updates updated_at
    - Preserves owner
    - Does not automatically publish
    - Returns updated Skill

    This endpoint is ONLY for database-backed Skills.
    """
    user_id = str(current_user.id)

    skill = crud.skill.get(db, skill_id)

    if not skill or skill.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Skill not found"
        )

    # Check for duplicate name if name is being updated
    if request.name and request.name != skill.name:
        existing = crud.skill.get_by_user_and_name(
            db, user_id=user_id, name=request.name
        )
        if existing:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error": {
                        "code": "DUPLICATE_NAME",
                        "message": f"Skill with name '{request.name}' already exists",
                    },
                    "data": None,
                    "message": f"Skill with name '{request.name}' already exists",
                },
            )

    # Update the skill using base CRUD update
    updated_skill = crud.skill.update(db, db_obj=skill, obj_in=request)

    AuditLogger.log(
        action="SKILL_UPDATE",
        user_id=user_id,
        endpoint=f"/v1/skills/{skill_id}",
        http_method="PATCH",
        status="200",
        metadata={
            "skill_id": skill.id,
            "skill_name": skill.name,
            "updated_fields": list(request.model_dump(exclude_unset=True).keys()),
        },
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": schemas.SkillResponse.model_validate(updated_skill).model_dump(
                mode="json"
            ),
            "message": "Skill updated successfully.",
        },
    )


@skill_router.post("/{skill_id}/publish")
def publish_skill(
    skill_id: str,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Publish a saved Skill.

    Changes the skill status from draft to published.
    Sets published_at timestamp and increments major version.
    Validates ownership and current state before publishing.

    Flow: Saved Skill (draft) → Validate → Publish → Published Skill
    """
    user_id = str(current_user.id)

    try:
        skill = crud.skill.publish_skill(
            db=db,
            skill_id=skill_id,
            user_id=user_id,
        )

        if not skill:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Skill not found or access denied",
            )

        AuditLogger.log(
            action="SKILL_PUBLISH",
            user_id=user_id,
            endpoint=f"/v1/skills/{skill_id}/publish",
            http_method="POST",
            status="200",
            metadata={
                "skill_id": skill.id,
                "skill_name": skill.name,
                "version": skill.version,
            },
        )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "error": None,
                "data": schemas.SkillResponse.model_validate(skill).model_dump(
                    mode="json"
                ),
                "message": "Skill published successfully.",
            },
        )

    except ValueError as e:
        logger.warning(f"Skill publish failed for user {user_id}: {e}")

        AuditLogger.log(
            action="SKILL_PUBLISH",
            user_id=user_id,
            endpoint=f"/v1/skills/{skill_id}/publish",
            http_method="POST",
            status="400",
            metadata={"error_message": str(e)},
        )

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "code": "INVALID_STATE_TRANSITION",
                    "message": str(e),
                },
                "data": None,
                "message": str(e),
            },
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"Unexpected error in skill publish for user {user_id}: {e}")

        AuditLogger.log(
            action="SKILL_PUBLISH",
            user_id=user_id,
            endpoint=f"/v1/skills/{skill_id}/publish",
            http_method="POST",
            status="500",
            metadata={
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred during skill publish",
                    "details": {"error_type": type(e).__name__},
                },
                "data": None,
                "message": "Internal server error",
            },
        )


@skill_router.delete("/{skill_id}")
def delete_skill(
    skill_id: str,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Delete a Skill.

    The user can delete both draft and published skills.
    - Authenticates user
    - Verifies ownership
    - Deletes database record
    - Returns appropriate response
    - Handles missing Skill

    After deletion, the Skill must no longer appear in GET or GET ALL.
    """
    user_id = str(current_user.id)

    skill = crud.skill.get(db, skill_id)

    if not skill or skill.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Skill not found"
        )

    crud.skill.remove(db, id=skill_id)

    AuditLogger.log(
        action="SKILL_DELETE",
        user_id=user_id,
        endpoint=f"/v1/skills/{skill_id}",
        http_method="DELETE",
        status="200",
        metadata={
            "skill_id": skill_id,
            "skill_name": skill.name,
            "status": skill.status,
        },
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": None,
            "message": "Skill deleted successfully.",
        },
    )
