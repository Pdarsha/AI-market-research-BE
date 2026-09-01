import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse, StreamingResponse

from server import crud, schemas
from server.api_v1.deps import get_db
from server.models.agent_workflow import AgentStatus, AgentWorkflow
from server.services.agent_scheduler_service import agent_scheduler_service
from server.services.agent_tool_registry import agent_tool_registry
from server.services.agent_validation_service import agent_validation_service

from server.services.agent_workflow_engine import agent_workflow_engine
from server.services.agent_seed_service import AgentSeedService
from server.services.audit_logger import AuditLogger
from server.utils.auth import get_current_user


logger = logging.getLogger(__name__)

agent_workflow_router = APIRouter(prefix="/agents", tags=["Agents"])


@agent_workflow_router.post("/validate")
def validate_agent(
    request: schemas.AgentValidationRequest,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Validate an Agent configuration against structural and business rules."""
    user_id = str(current_user.id)

    validation_result = agent_validation_service.validate_agent_config(
        agent=request.agent,
        user_id=user_id,
        db=db,
        check_uniqueness=request.check_uniqueness,
        exclude_agent_id=request.target_agent_id,
    )

    AuditLogger.log(
        action="AGENT_VALIDATE",
        user_id=user_id,
        endpoint="/v1/agents/validate",
        http_method="POST",
        status="200",
        metadata={
            "valid": validation_result.valid,
            "error_count": len(validation_result.errors),
            "warning_count": len(validation_result.warnings),
        },
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": validation_result.model_dump(mode="json"),
            "message": "Validation complete." if validation_result.valid else "Validation failed.",
        },
    )


@agent_workflow_router.post("")
def create_agent(
    request: schemas.AgentWorkflowCreate,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Create a new Agent Workflow in draft or active state."""
    user_id = str(current_user.id)

    # 1. Validate configuration
    validation_res = agent_validation_service.validate_agent_config(
        agent=request,
        user_id=user_id,
        db=db,
        check_uniqueness=True,
    )

    if not validation_res.valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "code": "VALIDATION_FAILED",
                    "message": "Agent configuration validation failed",
                    "details": [e.model_dump() for e in validation_res.errors],
                },
                "data": None,
                "message": "Cannot save invalid agent configuration.",
            },
        )

    request.recipient_email = getattr(current_user, "email", None)
    request.user_email = request.recipient_email

    try:
        agent = crud.agent_workflow.create_agent(
            db=db,
            user_id=user_id,
            obj_in=request,
        )

        AuditLogger.log(
            action="AGENT_CREATE",
            user_id=user_id,
            workspace_id=agent.workspace_id,
            endpoint="/v1/agents",
            http_method="POST",
            status="201",
            metadata={
                "agent_id": str(agent.id),
                "agent_name": agent.name,
                "trigger_type": agent.trigger_type,
                "status": agent.status,
            },
        )

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "success": True,
                "error": None,
                "data": schemas.AgentWorkflowResponse.model_validate(agent).model_dump(
                    mode="json"
                ),
                "message": "Agent created successfully.",
            },
        )

    except ValueError as e:
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


@agent_workflow_router.get("")
def list_agents(
    workspace_id: Optional[str] = Query(None, description="Filter by workspace ID"),
    trigger_type: Optional[str] = Query(None, description="Filter by trigger type"),
    status: Optional[str] = Query(None, description="Filter by agent status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=200),
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """List all Agent Workflows belonging to the authenticated user."""
    user_id = str(current_user.id)

    agents = crud.agent_workflow.get_multi_by_user(
        db,
        user_id=user_id,
        skip=skip,
        limit=limit,
        workspace_id=workspace_id,
        trigger_type=trigger_type,
        status=status,
    )

    serialized_list = []
    for item in agents:
        doc_count = len(item.selected_document_ids or [])
        list_item = schemas.AgentWorkflowListItem(
            id=str(item.id),
            name=item.name,
            description=item.description,
            trigger_type=item.trigger_type,
            schedule_frequency=item.schedule_frequency,
            schedule_day=item.schedule_day,
            schedule_time=item.schedule_time,
            schedule_start_time=item.schedule_start_time,
            schedule_end_time=item.schedule_end_time,
            schedule_timezone=item.schedule_timezone,
            execution_mode=item.execution_mode,
            status=item.status,
            version=item.version,
            workspace_id=item.workspace_id,
            limit_docs_past_7_days=item.limit_docs_past_7_days,
            limit_documents_to_past_7_days=item.limit_documents_to_past_7_days,
            selected_document_count=doc_count,
            selected_sources=item.selected_sources or [],
            created_at=item.created_at,
            updated_at=item.updated_at,
            activated_at=item.activated_at,
            last_run_at=item.last_run_at,
        )
        serialized_list.append(list_item.model_dump(mode="json"))

    AuditLogger.log(
        action="AGENT_LIST",
        user_id=user_id,
        workspace_id=workspace_id,
        endpoint="/v1/agents",
        http_method="GET",
        status="200",
        metadata={"count": len(serialized_list)},
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": serialized_list,
            "message": "Agents fetched successfully.",
        },
    )


@agent_workflow_router.get("/scheduler/status")
def get_scheduler_status(
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Retrieve scheduler engine status and active scheduled agents."""
    user_id = str(current_user.id)
    agents = agent_scheduler_service.get_active_scheduled_agents(db, user_id=user_id)

    agent_schedules = []
    for a in agents:
        next_run = agent_scheduler_service.calculate_next_run_time(a)
        agent_schedules.append({
            "agent_id": a.id,
            "agent_name": a.name,
            "frequency": a.schedule_frequency,
            "day": a.schedule_day,
            "time": a.schedule_time,
            "start_time": a.schedule_start_time,
            "end_time": a.schedule_end_time,
            "timezone": a.schedule_timezone,
            "cron": a.schedule_cron,
            "last_run_at": a.last_run_at.isoformat() if a.last_run_at else None,
            "next_run_at": next_run.isoformat() if next_run else None,
            "scheduled_count_limit": a.scheduled_count_limit,
            "limit_docs_past_7_days": a.limit_docs_past_7_days,
            "limit_documents_to_past_7_days": a.limit_documents_to_past_7_days,
        })

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": {
                "is_running": agent_scheduler_service.is_running,
                "scheduled_agents_count": len(agents),
                "scheduled_agents": agent_schedules,
            },
            "message": "Scheduler status retrieved successfully.",
        },
    )


@agent_workflow_router.post("/scheduler/tick")
async def trigger_scheduler_tick(
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Trigger a scheduler tick manually to evaluate and execute due scheduled agents."""
    tick_result = await agent_scheduler_service.tick(db=db)

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": tick_result,
            "message": f"Scheduler tick completed. {tick_result['triggered_count']} jobs triggered.",
        },
    )


@agent_workflow_router.get("/tools")
def list_available_agent_tools(
    current_user=Depends(get_current_user),
):
    """List all available built-in tools (RAG, SEC, Roic, News, Companies House, Web Search)."""
    tools = agent_tool_registry.list_tools()
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": tools,
            "message": "Tools retrieved successfully.",
        },
    )


@agent_workflow_router.get("/templates")
def list_default_agent_templates(
    current_user=Depends(get_current_user),
):
    """List the 10 canonical landing page default agents and workflow seed templates."""
    agents = AgentSeedService.get_default_landing_agents()
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": agents,
            "message": "Default landing page agent templates retrieved successfully.",
        },
    )


@agent_workflow_router.post("/run")
async def run_adhoc_workflow(
    request: schemas.AgentExecutionRequest,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Execute an ad-hoc Agent task without requiring a pre-saved agent workflow entity."""
    user_id = str(current_user.id)

    execution_response = await agent_workflow_engine.execute_workflow(
        db=db,
        workflow=None,
        user_id=user_id,
        query=request.query,
        document_ids=request.document_ids,
        filter_past_7_days=request.filter_past_7_days,
        trigger_source=request.trigger_source or "manual",
        parameters=request.parameters,
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": execution_response.model_dump(mode="json"),
            "message": "Workflow executed successfully.",
        },
    )


@agent_workflow_router.get("/runs/{run_id}")
def get_execution_run_detail(
    run_id: str,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Get complete details, state snapshot, and citations of an execution run."""
    user_id = str(current_user.id)

    execution = crud.agent_workflow.get_execution_by_id(
        db, user_id=user_id, execution_id=run_id
    )
    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Execution run not found"
        )

    step_logs_db = crud.agent_workflow.get_step_logs_by_execution(db, execution_id=run_id)
    step_log_responses = [
        schemas.AgentWorkflowStepLogResponse.model_validate(sl) for sl in step_logs_db
    ]

    response_obj = schemas.AgentWorkflowExecutionResponse.model_validate(execution)
    response_obj.step_logs = step_log_responses

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": response_obj.model_dump(mode="json"),
            "message": "Execution run retrieved successfully.",
        },
    )


@agent_workflow_router.get("/runs/{run_id}/steps")
def get_execution_step_logs(
    run_id: str,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Get the granular step execution logs and traces for an execution run."""
    user_id = str(current_user.id)

    execution = crud.agent_workflow.get_execution_by_id(
        db, user_id=user_id, execution_id=run_id
    )
    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Execution run not found"
        )

    step_logs_db = crud.agent_workflow.get_step_logs_by_execution(db, execution_id=run_id)

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": [
                schemas.AgentWorkflowStepLogResponse.model_validate(sl).model_dump(mode="json")
                for sl in step_logs_db
            ],
            "message": "Step logs retrieved successfully.",
        },
    )


@agent_workflow_router.get("/{agent_id}")
def get_agent(
    agent_id: str,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Retrieve full configuration and details of an Agent Workflow."""
    user_id = str(current_user.id)

    agent = crud.agent_workflow.get_by_user_and_id(db, user_id=user_id, agent_id=agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found"
        )

    AuditLogger.log(
        action="AGENT_GET",
        user_id=user_id,
        workspace_id=agent.workspace_id,
        endpoint=f"/v1/agents/{agent_id}",
        http_method="GET",
        status="200",
        metadata={"agent_id": agent_id},
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": schemas.AgentWorkflowResponse.model_validate(agent).model_dump(mode="json"),
            "message": "Agent details retrieved successfully.",
        },
    )


@agent_workflow_router.patch("/{agent_id}")
def update_agent(
    agent_id: str,
    request: schemas.AgentWorkflowUpdate,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Update an existing Agent Workflow configuration."""
    user_id = str(current_user.id)

    agent = crud.agent_workflow.get_by_user_and_id(db, user_id=user_id, agent_id=agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found"
        )

    request.model_fields_set.discard("recipient_email")
    request.model_fields_set.discard("user_email")

    # Prepare merged state for validation
    current_data = schemas.AgentWorkflowResponse.model_validate(agent).model_dump()
    update_data = request.model_dump(exclude_unset=True)

    # For partial updates (PATCH), ignore None or empty string for name/instructions so existing values are preserved
    non_null_updates = {
        k: v for k, v in update_data.items()
        if v is not None and (k != "name" or (isinstance(v, str) and v.strip()))
    }
    merged_data = {**current_data, **non_null_updates}

    validation_res = agent_validation_service.validate_agent_config(
        agent=merged_data,
        user_id=user_id,
        db=db,
        check_uniqueness=bool(
            non_null_updates.get("name") and non_null_updates["name"] != agent.name
        ),
        exclude_agent_id=agent_id,
    )


    if not validation_res.valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "code": "VALIDATION_FAILED",
                    "message": "Agent configuration validation failed",
                    "details": [e.model_dump() for e in validation_res.errors],
                },
                "data": None,
                "message": "Cannot update agent with invalid configuration.",
            },
        )

    updated_agent = crud.agent_workflow.update_agent(
        db=db,
        db_obj=agent,
        obj_in=request,
    )

    AuditLogger.log(
        action="AGENT_UPDATE",
        user_id=user_id,
        workspace_id=updated_agent.workspace_id,
        endpoint=f"/v1/agents/{agent_id}",
        http_method="PATCH",
        status="200",
        metadata={
            "agent_id": agent_id,
            "updated_fields": list(update_data.keys()),
        },
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": schemas.AgentWorkflowResponse.model_validate(updated_agent).model_dump(
                mode="json"
            ),
            "message": "Agent updated successfully.",
        },
    )


@agent_workflow_router.post("/{agent_id}/activate")
def activate_agent(
    agent_id: str,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Activate an Agent Workflow. Strictly validates instructions and schedule configuration."""
    user_id = str(current_user.id)

    agent = crud.agent_workflow.get_by_user_and_id(db, user_id=user_id, agent_id=agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found"
        )

    # Perform active-mode validation
    agent_dict = schemas.AgentWorkflowResponse.model_validate(agent).model_dump()
    agent_dict["status"] = AgentStatus.ACTIVE.value

    validation_res = agent_validation_service.validate_agent_config(
        agent=agent_dict,
        user_id=user_id,
        db=db,
        check_uniqueness=False,
    )

    if not validation_res.valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "code": "ACTIVATION_FAILED",
                    "message": "Agent cannot be activated because required fields are missing or invalid",
                    "details": [e.model_dump() for e in validation_res.errors],
                },
                "data": None,
                "message": "Cannot activate agent with incomplete configuration.",
            },
        )

    activated_agent = crud.agent_workflow.activate_agent(db=db, db_obj=agent)

    AuditLogger.log(
        action="AGENT_ACTIVATE",
        user_id=user_id,
        workspace_id=activated_agent.workspace_id,
        endpoint=f"/v1/agents/{agent_id}/activate",
        http_method="POST",
        status="200",
        metadata={"agent_id": agent_id, "status": activated_agent.status},
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": schemas.AgentWorkflowResponse.model_validate(activated_agent).model_dump(
                mode="json"
            ),
            "message": "Agent activated successfully.",
        },
    )


@agent_workflow_router.post("/{agent_id}/pause")
def pause_agent(
    agent_id: str,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Pause an active Agent Workflow."""
    user_id = str(current_user.id)

    agent = crud.agent_workflow.get_by_user_and_id(db, user_id=user_id, agent_id=agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found"
        )

    paused_agent = crud.agent_workflow.pause_agent(db=db, db_obj=agent)

    AuditLogger.log(
        action="AGENT_PAUSE",
        user_id=user_id,
        workspace_id=paused_agent.workspace_id,
        endpoint=f"/v1/agents/{agent_id}/pause",
        http_method="POST",
        status="200",
        metadata={"agent_id": agent_id, "status": paused_agent.status},
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": schemas.AgentWorkflowResponse.model_validate(paused_agent).model_dump(
                mode="json"
            ),
            "message": "Agent paused successfully.",
        },
    )


@agent_workflow_router.post("/{agent_id}/deactivate")
def deactivate_agent(
    agent_id: str,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Deactivate/pause an active Agent Workflow."""
    user_id = str(current_user.id)

    agent = crud.agent_workflow.get_by_user_and_id(db, user_id=user_id, agent_id=agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found"
        )

    deactivated_agent = crud.agent_workflow.deactivate_agent(db=db, db_obj=agent)

    AuditLogger.log(
        action="AGENT_DEACTIVATE",
        user_id=user_id,
        workspace_id=deactivated_agent.workspace_id,
        endpoint=f"/v1/agents/{agent_id}/deactivate",
        http_method="POST",
        status="200",
        metadata={"agent_id": agent_id, "status": deactivated_agent.status},
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": schemas.AgentWorkflowResponse.model_validate(deactivated_agent).model_dump(
                mode="json"
            ),
            "message": "Agent deactivated successfully.",
        },
    )


@agent_workflow_router.delete("/{agent_id}")
def delete_agent(
    agent_id: str,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Delete an Agent Workflow."""
    user_id = str(current_user.id)

    agent = crud.agent_workflow.get_by_user_and_id(db, user_id=user_id, agent_id=agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found"
        )

    workspace_id = agent.workspace_id
    agent_name = agent.name

    crud.agent_workflow.remove(db, id=agent.id)

    AuditLogger.log(
        action="AGENT_DELETE",
        user_id=user_id,
        workspace_id=workspace_id,
        endpoint=f"/v1/agents/{agent_id}",
        http_method="DELETE",
        status="200",
        metadata={"agent_id": agent_id, "agent_name": agent_name},
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": None,
            "message": "Agent deleted successfully.",
        },
    )


# --- Execution Endpoints ---


@agent_workflow_router.post("/{agent_id}/run")
async def run_agent_workflow(
    agent_id: str,
    request: schemas.AgentExecutionRequest = schemas.AgentExecutionRequest(),
    current_user=Depends(get_current_user),
):
    """Execute a configured Agent Workflow through the isolated agent execution service."""
    from server.services.agent_execution_service import agent_execution_service

    user_id = str(current_user.id)

    if request.stream:
        return StreamingResponse(
            agent_execution_service.execute_stream(
                agent_id=agent_id,
                user_id=user_id,
                request=request,
            ),
            media_type="text/event-stream",
        )

    execution_response = await agent_execution_service.execute(
        agent_id=agent_id,
        user_id=user_id,
        query=request.query,
        workspace_id=request.workspace_id,
        document_ids=request.document_ids,
        filter_past_7_days=request.filter_past_7_days,
        trigger_source=request.trigger_source or "manual",
        parameters=request.parameters,
        chat_id=request.chat_id,
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": execution_response.status == "completed",
            "error": execution_response.error_message if execution_response.status != "completed" else None,
            "data": execution_response.model_dump(mode="json"),
            "message": (
                "Agent execution completed successfully."
                if execution_response.status == "completed"
                else f"Agent execution failed: {execution_response.error_message}"
            ),
        },
    )




@agent_workflow_router.post("/{agent_id}/trigger-schedule")
async def trigger_agent_schedule_manually(
    agent_id: str,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Manually trigger the scheduled execution for an agent workflow."""
    user_id = str(current_user.id)
    agent = crud.agent_workflow.get_by_user_and_id(db, user_id=user_id, agent_id=agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found"
        )

    execution_response = await agent_scheduler_service.execute_scheduled_agent(
        agent_id=agent.id, db=db
    )

    if not execution_response:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to execute scheduled agent (agent might not be active or already running)",
        )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": execution_response.model_dump(mode="json"),
            "message": f"Scheduled execution for Agent '{agent.name}' completed.",
        },
    )



@agent_workflow_router.get("/{agent_id}/runs")
def list_agent_runs(
    agent_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    """List execution history and runs for a specific agent workflow."""
    user_id = str(current_user.id)

    agent = crud.agent_workflow.get_by_user_and_id(db, user_id=user_id, agent_id=agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found"
        )

    executions = crud.agent_workflow.list_executions_by_workflow(
        db, user_id=user_id, workflow_id=agent_id, skip=skip, limit=limit
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": [
                schemas.AgentWorkflowExecutionListItem(
                    id=e.id,
                    workflow_id=e.workflow_id,
                    trigger_source=e.trigger_source,
                    status=e.status,
                    query_or_goal=e.query_or_goal,
                    execution_time_ms=e.execution_time_ms,
                    started_at=e.started_at,
                    completed_at=e.completed_at,
                    step_count=len(e.plan_steps or []),
                ).model_dump(mode="json")
                for e in executions
            ],
            "message": "Execution history retrieved successfully.",
        },
    )
