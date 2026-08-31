"""Agent Execution Service.

Provides an isolated, non-blocking execution service for Agent Workflows.
Decouples agent execution lifecycle from HTTP request handlers and scheduler ticks:
  - Manages independent database session per execution
  - Orchestrates chat conversation creation and message persistence
  - Manages execution state lifecycle: PENDING -> RUNNING -> COMPLETED / FAILED / CANCELLED
  - Executes workflow asynchronously without blocking the event loop
  - Updates last_run_at and next_run_at according to scheduling rules
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from server import crud, schemas
from server.db.session import SessionLocal
from server.models.agent_workflow import (
    AgentStatus,
    AgentWorkflow,
    AgentWorkflowExecution,
    AIExecutionMode,
    ExecutionStatus,
    TriggerType,
)
from server.models.genai import ChatMessage, ChatSession
from server.schemas.agent_workflow import (
    AgentExecutionRequest,
    AgentWorkflowExecutionResponse,
)
from server.services.agent_title_service import generate_agent_chat_title
from server.services.agent_workflow_engine import (
    AgentWorkflowEngine,
    get_agent_workflow_engine,
)
from server.services.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class AgentExecutionService:
    """Isolated execution coordinator for Agent Workflows."""

    def __init__(
        self,
        workflow_engine: Optional[AgentWorkflowEngine] = None,
    ):
        self._workflow_engine = workflow_engine

    @property
    def workflow_engine(self) -> AgentWorkflowEngine:
        if self._workflow_engine is None:
            self._workflow_engine = get_agent_workflow_engine()
        return self._workflow_engine

    async def execute(
        self,
        *,
        agent_id: str,
        user_id: Optional[str] = None,
        query: Optional[str] = None,
        workspace_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        filter_past_7_days: Optional[bool] = None,
        trigger_source: str = "manual",
        parameters: Optional[Dict[str, Any]] = None,
        chat_id: Optional[str] = None,
        status_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
        cancellation_check: Optional[Callable[[], bool]] = None,
        db: Optional[Session] = None,
        triggered_at: Optional[datetime] = None,
    ) -> AgentWorkflowExecutionResponse:
        """Execute an agent workflow with an isolated DB session and full state lifecycle."""
        owns_db = False
        db_session = db
        if db_session is None:
            db_session = SessionLocal()
            owns_db = True

        execution_time_now = triggered_at or datetime.utcnow()
        parameters = dict(parameters or {})

        try:
            # 1. Load agent with user ownership verification
            if user_id:
                agent = crud.agent_workflow.get_by_user_and_id(
                    db_session, user_id=str(user_id), agent_id=agent_id
                )
            else:
                agent = crud.agent_workflow.get_agent_by_id(db_session, agent_id=agent_id)

            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found"
                )

            resolved_user_id = str(agent.user_id)
            resolved_workspace_id = workspace_id or agent.workspace_id
            if resolved_workspace_id:
                parameters["workspace_id"] = resolved_workspace_id

            agent_status = getattr(agent.status, "value", agent.status)
            if agent_status != AgentStatus.ACTIVE.value:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Agent is inactive and cannot be executed.",
                )

            # Check scheduled count limit for scheduled/automatic triggers
            if trigger_source in ("scheduled", "automatic") and agent.scheduled_count_limit is not None:
                exec_count = crud.agent_workflow.get_exec_count(db_session, agent.id) or 0
                if exec_count >= agent.scheduled_count_limit:
                    agent.status = AgentStatus.PAUSED.value
                    agent.next_run_at = None
                    db_session.add(agent)
                    db_session.commit()
                    logger.info(
                        "[AGENT_EXECUTION] Agent '%s' reached limit (%d/%d) and was paused.",
                        agent.name,
                        exec_count,
                        agent.scheduled_count_limit,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Agent has reached its scheduled count limit of {agent.scheduled_count_limit} runs.",
                    )

            # 2. Conversation handling
            target_chat_id = chat_id or parameters.get("chat_id")
            resolved_chat_session: Optional[ChatSession] = None
            effective_query = (
                query.strip()
                if query and query.strip()
                else (agent.instructions or agent.name)
            )

            if target_chat_id:
                # Attach to existing chat session
                resolved_chat_session = crud.chat_session.get(db_session, id=target_chat_id)
                if resolved_chat_session and str(resolved_chat_session.user_id) == resolved_user_id:
                    resolved_chat_session.agent_id = agent.id
                    resolved_chat_session.agent_name = agent.name
                    resolved_chat_session.execution_type = "agent"
                    if not resolved_chat_session.title or resolved_chat_session.title in (
                        "New Task",
                        "Untitled",
                        "New Chat",
                        "Chat",
                        agent.name,
                    ):
                        resolved_chat_session.title = generate_agent_chat_title(
                            agent_name=agent.name,
                            instructions=agent.instructions,
                            query=query,
                            trigger_source=trigger_source,
                            execution_time=execution_time_now,
                        )
                    db_session.add(resolved_chat_session)
                    db_session.commit()

                    # Save user message if provided
                    if query and query.strip():
                        user_msg = ChatMessage(
                            chat_id=target_chat_id,
                            user_id=resolved_user_id,
                            role="user",
                            content=query.strip(),
                            workspace_id=resolved_workspace_id,
                            document_ids=document_ids or agent.selected_document_ids or [],
                        )
                        db_session.add(user_msg)
                        db_session.commit()
            elif trigger_source in ("scheduled", "automatic"):
                # Create dedicated conversation for scheduled run
                sched_title = generate_agent_chat_title(
                    agent_name=agent.name,
                    instructions=agent.instructions,
                    trigger_source=trigger_source,
                    execution_time=execution_time_now,
                )
                resolved_chat_session = crud.chat_session.create(
                    db=db_session,
                    obj_in=schemas.ChatSessionCreate(
                        user_id=resolved_user_id,
                        title=sched_title,
                        workspace_id=resolved_workspace_id,
                        agent_id=str(agent.id),
                        agent_name=agent.name,
                        execution_type="agent",
                        execution_date=execution_time_now.date(),
                        chat_metadata={
                            "trigger_source": trigger_source,
                            "scheduled_at": execution_time_now.isoformat(),
                            "frequency": agent.schedule_frequency,
                        },
                    ),
                )
                target_chat_id = resolved_chat_session.id
                parameters["chat_id"] = target_chat_id

                # Save user prompt message
                user_msg = ChatMessage(
                    chat_id=target_chat_id,
                    user_id=resolved_user_id,
                    role="user",
                    content=effective_query,
                    workspace_id=resolved_workspace_id,
                    document_ids=agent.selected_document_ids or [],
                )
                db_session.add(user_msg)
                db_session.commit()
                db_session.refresh(resolved_chat_session)

                logger.info(
                    "[AGENT_EXECUTION] [CONVERSATION_CREATED] Created chat '%s' (chat_id=%s, agent_id=%s)",
                    resolved_chat_session.title,
                    target_chat_id,
                    agent.id,
                )

            # 3. Execute workflow
            logger.info(
                "[AGENT_EXECUTION] [START] Executing agent '%s' (id=%s, trigger=%s, chat_id=%s)",
                agent.name,
                agent.id,
                trigger_source,
                target_chat_id,
            )

            execution_response = await self.workflow_engine.execute_workflow(
                db=db_session,
                workflow=agent,
                user_id=resolved_user_id,
                query=effective_query,
                document_ids=document_ids,
                filter_past_7_days=filter_past_7_days,
                trigger_source=trigger_source,
                parameters=parameters,
                status_callback=status_callback,
                cancellation_check=cancellation_check,
            )

            # 4. Persist response in chat session if available
            if target_chat_id:
                try:
                    mode_used = (
                        agent.execution_mode.value
                        if hasattr(agent.execution_mode, "value")
                        else str(agent.execution_mode or "auto")
                    )
                    if (
                        execution_response
                        and execution_response.final_response
                        and execution_response.status == ExecutionStatus.COMPLETED.value
                    ):
                        assistant_msg = ChatMessage(
                            chat_id=target_chat_id,
                            user_id=resolved_user_id,
                            role="assistant",
                            content=execution_response.final_response,
                            workspace_id=resolved_workspace_id,
                            document_ids=document_ids or agent.selected_document_ids or [],
                            source_documents=execution_response.source_documents or None,
                            citations=execution_response.citations or None,
                            skills_used=[
                                {
                                    "name": s,
                                    "selection_type": "agent_skill",
                                    "is_auto": True,
                                }
                                for s in (execution_response.skills_applied or [])
                            ],
                            agent_id=agent.id,
                            agent_name=agent.name,
                            mode_used=mode_used,
                        )
                        db_session.add(assistant_msg)
                        if resolved_chat_session:
                            resolved_chat_session.updated_at = datetime.utcnow()
                            db_session.add(resolved_chat_session)
                        db_session.commit()
                        logger.info(
                            "[AGENT_EXECUTION] [RESPONSE_SAVED] Persisted response to chat '%s'",
                            target_chat_id,
                        )
                    elif execution_response and execution_response.status == ExecutionStatus.FAILED.value:
                        err_msg = ChatMessage(
                            chat_id=target_chat_id,
                            user_id=resolved_user_id,
                            role="assistant",
                            content=f"⚠️ **Agent Execution Failed**: {execution_response.error_message or 'Unknown error'}",
                            workspace_id=resolved_workspace_id,
                            document_ids=document_ids or agent.selected_document_ids or [],
                            agent_id=agent.id,
                            agent_name=agent.name,
                        )
                        db_session.add(err_msg)
                        if resolved_chat_session:
                            resolved_chat_session.updated_at = datetime.utcnow()
                            db_session.add(resolved_chat_session)
                        db_session.commit()
                except Exception as ex_chat:
                    logger.warning(
                        "[AGENT_EXECUTION] Failed to persist chat response: %s", ex_chat
                    )

            if execution_response:
                execution_response.agent_id = agent.id
                execution_response.agent_name = agent.name
                execution_response.workflow_name = agent.name

            # 5. Scheduled Agent Email Delivery
            if (
                trigger_source == "scheduled"
                and execution_response
                and execution_response.status == ExecutionStatus.COMPLETED.value
            ):
                try:
                    recipient = agent.recipient_email
                    if not recipient:
                        from server.models.user import User
                        user_obj = agent.user or db_session.query(User).filter(User.id == str(agent.user_id)).first()
                        recipient = user_obj.email if user_obj else None

                    if recipient:
                        from server.utils.mail import send_agent_workflow_email

                        await send_agent_workflow_email(
                            email=recipient,
                            agent_name=agent.name,
                            result_content=execution_response.final_response or "",
                            execution_time=execution_time_now,
                            sources=execution_response.source_documents or [],
                        )
                        logger.info(
                            "[AGENT_EXECUTION] [EMAIL_SENT] Dispatched scheduled result email to %s for agent '%s'",
                            recipient,
                            agent.name,
                        )
                    else:
                        logger.warning(
                            "[AGENT_EXECUTION] [EMAIL_SKIPPED] No recipient email found for agent '%s' (user_id=%s)",
                            agent.name,
                            agent.user_id,
                        )
                except Exception as ex_mail:
                    logger.error(
                        "[AGENT_EXECUTION] [EMAIL_ERROR] Failed sending scheduled agent result email: %s",
                        ex_mail,
                        exc_info=True,
                    )

            # 6. Advance schedule & update last_run_at
            from server.services.agent_scheduler_service import agent_scheduler_service

            agent.last_run_at = datetime.utcnow()
            status_val = getattr(agent.status, "value", agent.status)
            trigger_val = getattr(agent.trigger_type, "value", agent.trigger_type)
            if status_val == AgentStatus.ACTIVE.value and trigger_val in (
                TriggerType.SCHEDULED.value,
                TriggerType.BOTH.value,
                "scheduled",
                "both",
            ):
                agent.next_run_at = agent_scheduler_service.calculate_next_run_time(agent)
                if agent.scheduled_count_limit is not None:
                    total_runs = crud.agent_workflow.get_exec_count(db_session, agent.id) or 0
                    if total_runs >= agent.scheduled_count_limit:
                        agent.status = AgentStatus.PAUSED.value
                        agent.next_run_at = None

            db_session.add(agent)
            db_session.commit()

            return execution_response

        finally:
            if owns_db and db_session is not None:
                db_session.close()

    async def execute_stream(
        self,
        *,
        agent_id: str,
        user_id: str,
        request: AgentExecutionRequest,
        db: Optional[Session] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream execution events and final tokens in Server-Sent Events (SSE) format."""
        queue: asyncio.Queue = asyncio.Queue()

        async def on_status_event(event: dict):
            await queue.put(event)

        owns_db = False
        db_session = db
        if db_session is None:
            db_session = SessionLocal()
            owns_db = True

        try:
            agent = crud.agent_workflow.get_by_user_and_id(
                db_session, user_id=user_id, agent_id=agent_id
            )
            if not agent:
                yield f"data: {json.dumps({'type': 'error', 'error': 'Agent not found'})}\n\n"
                return

            start_payload = {
                "type": "agent_started",
                "status": "agent_started",
                "step": f"Starting agent '{agent.name}'...",
                "agent_id": agent.id,
                "agent_name": agent.name,
                "agents": [{"id": agent.id, "name": agent.name, "isWorkflow": True}],
                "execution_mode": (
                    agent.execution_mode.value
                    if hasattr(agent.execution_mode, "value")
                    else str(agent.execution_mode)
                ),
            }
            yield f"data: {json.dumps(start_payload)}\n\n"

            exec_task = asyncio.create_task(
                self.execute(
                    agent_id=agent_id,
                    user_id=user_id,
                    query=request.query,
                    workspace_id=request.workspace_id,
                    document_ids=request.document_ids,
                    filter_past_7_days=request.filter_past_7_days,
                    trigger_source=request.trigger_source or "manual",
                    parameters=request.parameters,
                    chat_id=request.chat_id,
                    status_callback=on_status_event,
                    db=db_session,
                )
            )

            while not exec_task.done() or not queue.empty():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.05)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    continue

            exec_res = await exec_task
            final_text = exec_res.final_response or ""

            # Stream tokens
            chunk_size = 16
            for i in range(0, len(final_text), chunk_size):
                token_chunk = final_text[i : i + chunk_size]
                yield f"data: {json.dumps({'type': 'token', 'content': token_chunk})}\n\n"

            yield f"data: {json.dumps({'type': 'complete', 'data': exec_res.model_dump(mode='json')})}\n\n"

        finally:
            if owns_db and db_session is not None:
                db_session.close()


agent_execution_service = AgentExecutionService()
