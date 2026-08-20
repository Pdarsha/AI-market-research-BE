"""Agent Trigger Service.

Implements the Automatic Trigger mechanism for Agent Workflows:
  1. Receive the existing user request / chat message.
  2. Determine whether an active Agent is configured for Automatic Trigger.
  3. Identify the relevant Agent based on the user's request and Agent configuration.
  4. If a suitable Agent is found, invoke the Agent Execution Engine (with trigger_source="automatic").
  5. Pass user's request, document context, and workspace scoping to the Agent.
  6. Stream or return the Agent result through the existing chat response flow.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI
from sqlalchemy.orm import Session

from server import crud, schemas
from server.config import settings
from server.models.agent_workflow import AgentStatus, AgentWorkflow, TriggerType
from server.services.agent_workflow_engine import (
    AgentWorkflowEngine,
    get_agent_workflow_engine,
)
from server.utils.chat_helpers import get_llm_kwargs
from server.utils.llm_client import get_openai_client, execute_with_llm_retry
from server.services.audit_logger import AuditLogger
from server.utils.source_attribution import source_attribution_from_citations

logger = logging.getLogger(__name__)


class AgentTriggerService:
    """Service to discover, match, and automatically trigger active Agent Workflows from user queries."""

    def __init__(
        self,
        openai_client: Optional[OpenAI] = None,
        workflow_engine: Optional[AgentWorkflowEngine] = None,
    ):
        self.openai_client = openai_client or get_openai_client()
        self.workflow_engine = workflow_engine or get_agent_workflow_engine()

    def get_active_automatic_agents(
        self, db: Session, *, user_id: str, workspace_id: Optional[str] = None
    ) -> List[AgentWorkflow]:
        """Fetch active agents belonging to user configured with automatic or both trigger types."""
        filters = [
            AgentWorkflow.user_id == str(user_id),
            AgentWorkflow.status.in_([AgentStatus.ACTIVE.value, AgentStatus.ACTIVE]),
            AgentWorkflow.trigger_type.in_([TriggerType.AUTOMATIC.value, TriggerType.BOTH.value]),
        ]
        if workspace_id:
            filters.append(
                (AgentWorkflow.workspace_id == str(workspace_id))
                | (AgentWorkflow.workspace_id.is_(None))
            )
        agents = db.query(AgentWorkflow).filter(*filters).all()
        # In-memory safety filter ensuring inactive/paused/draft agents are never returned
        return [
            a for a in agents
            if a.status in (AgentStatus.ACTIVE.value, AgentStatus.ACTIVE)
        ]


    def match_agent_for_request(
        self,
        db: Session,
        *,
        user_id: str,
        query: str,
        workspace_id: Optional[str] = None,
        explicit_agent_id: Optional[str] = None,
    ) -> Optional[AgentWorkflow]:
        """Determine if an active Automatic Trigger Agent matches the user's request.

        CRITICAL: Inactive agents are strictly excluded before any matching, scoring, or selection.

        Returns:
            Matched active AgentWorkflow or None if no agent is suitable.
        """
        if not query or not query.strip():
            return None

        candidates = self.get_active_automatic_agents(
            db, user_id=user_id, workspace_id=workspace_id
        )
        # Double check candidate active status
        active_candidates = [
            a for a in candidates
            if a.status in (AgentStatus.ACTIVE.value, AgentStatus.ACTIVE)
        ]
        if not active_candidates:
            return None

        # 1. Explicit ID match (e.g. passed in request) - only allowed if agent is active
        if explicit_agent_id:
            for agent in active_candidates:
                if str(agent.id) == str(explicit_agent_id):
                    return agent

        clean_query = query.strip()

        # 2. Check for explicit @mention or agent prefix (e.g. "@Financial Analyst", "agent:Market Research")
        for agent in active_candidates:
            agent_name = agent.name.strip().lower()
            # Check @AgentName
            pattern_at = re.compile(rf"@\b{re.escape(agent_name)}\b", re.IGNORECASE)
            # Check agent:AgentName
            pattern_prefix = re.compile(rf"agent:\s*{re.escape(agent_name)}\b", re.IGNORECASE)
            if pattern_at.search(clean_query) or pattern_prefix.search(clean_query):
                logger.info(
                    "[AGENT_AUTO_TRIGGER] Explicit name match for active agent '%s' (id=%s)",
                    agent.name,
                    agent.id,
                )
                return agent

        # 3. If there is an exact substring match for agent name among active candidates
        for agent in active_candidates:
            if agent.name.lower() in clean_query.lower():
                logger.info(
                    "[AGENT_AUTO_TRIGGER] Exact substring match for active agent '%s' (id=%s)",
                    agent.name,
                    agent.id,
                )
                return agent

        # 4. Semantic / Intent-aware Matching using LLM Dispatcher
        agent_summaries = [
            {
                "id": a.id,
                "name": a.name,
                "description": a.description or "",
                "instructions_preview": (a.instructions or "")[:200],
                "sources": a.selected_sources or [],
            }
            for a in active_candidates
        ]

        system_prompt = (
            "You are an AI Agent Dispatch Router. Your job is to evaluate if a user's question or request "
            "is specifically asking for one of the user's active autonomous AI Agents.\n"
            "Rules:\n"
            "- Only match an agent if the query directly corresponds to that agent's explicit specialty or named purpose.\n"
            "- If the query is a general chat question, greeting, or standard document lookup, return matched_agent_id as null.\n"
            "- Respond strictly in valid JSON: {'matched_agent_id': 'string or null', 'confidence': float, 'reason': 'string'}"
        )

        user_content = (
            f"Active User Agents:\n{json.dumps(agent_summaries, indent=2)}\n\n"
            f"User Query:\n\"{clean_query}\"\n\n"
            "Evaluate match."
        )

        try:
            trigger_kwargs = get_llm_kwargs(
                settings.LLM_MODEL,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            def _do_trigger_eval():
                try:
                    return self.openai_client.chat.completions.create(
                        model=settings.LLM_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                        **trigger_kwargs,
                    )
                except Exception as api_err:
                    if "temperature" in str(api_err).lower() and "temperature" in trigger_kwargs:
                        retry_kwargs = {k: v for k, v in trigger_kwargs.items() if k != "temperature"}
                        return self.openai_client.chat.completions.create(
                            model=settings.LLM_MODEL,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_content},
                            ],
                            **retry_kwargs,
                        )
                    raise api_err

            response = execute_with_llm_retry(
                _do_trigger_eval,
                max_attempts=3,
                initial_delay=1.0,
                operation_name="Agent Auto-Trigger Evaluation",
            )
            parsed = json.loads(response.choices[0].message.content or "{}")
            matched_id = parsed.get("matched_agent_id")
            confidence = float(parsed.get("confidence", 0.0))

            if matched_id and confidence >= 0.7:
                for a in active_candidates:
                    if str(a.id) == str(matched_id):
                        logger.info(
                            "[AGENT_AUTO_TRIGGER] Semantic match: Active Agent '%s' (id=%s, confidence=%.2f, reason=%s)",
                            a.name,
                            a.id,
                            confidence,
                            parsed.get("reason"),
                        )
                        return a
        except Exception as e:
            logger.warning(f"Semantic agent dispatch failed: {str(e)}")

        return None

    def match_agents_for_request(
        self,
        db: Session,
        *,
        user_id: str,
        query: str,
        workspace_id: Optional[str] = None,
        max_agents: int = 3,
    ) -> List[AgentWorkflow]:
        """Match up to max_agents active Automatic Trigger Agents for a user request.

        CRITICAL: Inactive/paused/draft agents are strictly excluded from all matching paths.
        """
        if not query or not query.strip():
            return []

        candidates = self.get_active_automatic_agents(
            db, user_id=user_id, workspace_id=workspace_id
        )
        active_candidates = [
            a for a in candidates
            if a.status in (AgentStatus.ACTIVE.value, AgentStatus.ACTIVE)
        ]
        if not active_candidates:
            return []

        clean_query = query.strip()
        matched: List[AgentWorkflow] = []

        # 1. Exact name / mention / substring matches among active candidates
        for agent in active_candidates:
            agent_name = agent.name.strip().lower()
            pattern_at = re.compile(rf"@\b{re.escape(agent_name)}\b", re.IGNORECASE)
            pattern_prefix = re.compile(rf"agent:\s*{re.escape(agent_name)}\b", re.IGNORECASE)
            if pattern_at.search(clean_query) or pattern_prefix.search(clean_query) or (agent_name in clean_query.lower()):
                if agent not in matched:
                    matched.append(agent)
                    if len(matched) >= max_agents:
                        return matched

        # If none matched by keyword, try single semantic match
        if not matched:
            single = self.match_agent_for_request(
                db, user_id=user_id, query=query, workspace_id=workspace_id
            )
            if single and single.status in (AgentStatus.ACTIVE.value, AgentStatus.ACTIVE):
                matched.append(single)

        return matched[:max_agents]

    def execute_and_stream_chat_response(
        self,
        db: Session,
        *,
        current_user: Any,
        chat_id: str,
        session: Any,
        request: schemas.ChatMessageRequest,
        matched_agent: AgentWorkflow,
        unique_doc_ids: List[str],
        background_tasks: Optional[Any] = None,
    ) -> Any:
        """Invoke Agent Execution Engine and return streaming SSE or non-streaming response."""
        user_id = str(current_user.id)
        workspace_id = session.workspace_id or request.workspace_id

        # Guard: Inactive agent cannot be executed
        if matched_agent.status not in (AgentStatus.ACTIVE.value, AgentStatus.ACTIVE):
            raise HTTPException(
                status_code=400,
                detail="Agent is inactive and cannot be executed.",
            )

        # 1. Update session metadata and canonical title
        try:
            from server.services.agent_title_service import generate_agent_chat_title
            session.agent_id = matched_agent.id
            session.agent_name = matched_agent.name
            session.execution_type = "agent"
            if not session.title or session.title in ("New Task", "Untitled", "New Chat", "Chat"):
                session.title = generate_agent_chat_title(
                    agent_name=matched_agent.name,
                    instructions=matched_agent.instructions,
                    query=request.content,
                    trigger_source="automatic",
                )
            db.add(session)
            db.commit()
        except Exception as ex_sess:
            logger.warning(f"[AGENT_TRIGGER] Failed to update session metadata: {ex_sess}")

        # 2. Create User Message
        user_message = crud.chat_message.create(
            db=db,
            obj_in=schemas.ChatMessageCreate(
                chat_id=chat_id,
                content=request.content,
                role="user",
                user_id=user_id,
                workspace_id=workspace_id,
                document_ids=unique_doc_ids or None,
            ),
        )

        # 2. Async runner for workflow execution with live status event streaming
        async def event_stream():
            queue: asyncio.Queue = asyncio.Queue()

            async def on_status_event(event: Dict[str, Any]):
                await queue.put(event)

            # Initial event: announce automatic agent trigger
            start_payload = {
                "type": "agent_started",
                "status": "agent_started",
                "step": f"Starting agent '{matched_agent.name}'...",
                "agent_id": matched_agent.id,
                "agent_name": matched_agent.name,
                "agents": [{"id": matched_agent.id, "name": matched_agent.name, "isWorkflow": True}],
                "execution_mode": (
                    matched_agent.execution_mode.value
                    if hasattr(matched_agent.execution_mode, "value")
                    else str(matched_agent.execution_mode)
                ),
            }
            yield f"data: {json.dumps(start_payload)}\n\n"

            # Launch execution task concurrently
            workflow_task = asyncio.create_task(
                self.workflow_engine.execute_workflow(
                    db=db,
                    workflow=matched_agent,
                    user_id=user_id,
                    query=request.content,
                    document_ids=unique_doc_ids,
                    filter_past_7_days=matched_agent.limit_docs_past_7_days,
                    trigger_source="automatic",
                    parameters={
                        "chat_id": chat_id,
                        "workspace_id": workspace_id,
                        "user_message_id": user_message.id,
                    },
                    status_callback=on_status_event,
                )
            )

            # Stream live status events in real time as they are emitted
            while not workflow_task.done() or not queue.empty():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.05)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    continue

            exec_res = await workflow_task
            final_text = exec_res.final_response or ""

            # Stream tokens in small chunks
            chunk_size = 16
            for i in range(0, len(final_text), chunk_size):
                token_chunk = final_text[i : i + chunk_size]
                yield f"data: {json.dumps({'type': 'token', 'content': token_chunk})}\n\n"

            # Persist assistant message in chat history
            mode_used = (
                matched_agent.execution_mode.value
                if hasattr(matched_agent.execution_mode, "value")
                else str(matched_agent.execution_mode)
            )
            assistant_message = crud.chat_message.create(
                db=db,
                obj_in=schemas.ChatMessageCreate(
                    chat_id=chat_id,
                    content=final_text,
                    role="assistant",
                    user_id=user_id,
                    workspace_id=workspace_id,
                    document_ids=unique_doc_ids or None,
                    source_documents=exec_res.source_documents or None,
                    citations=exec_res.citations or None,
                    skills_used=[
                        {"name": s, "selection_type": "agent_skill", "is_auto": True}
                        for s in (exec_res.skills_applied or [])
                    ],
                    agent_id=matched_agent.id,
                    agent_name=matched_agent.name,
                    mode_used=mode_used,
                ),
            )

            # Audit log
            AuditLogger.log(
                action="AGENT_AUTOMATIC_TRIGGER_RESPONSE",
                user_id=user_id,
                workspace_id=workspace_id,
                chat_id=chat_id,
                endpoint=f"/v1/ai/chats/{chat_id}/messages",
                http_method="POST",
                status="200",
                processing_time=exec_res.execution_time_ms,
                metadata={
                    "agent_id": matched_agent.id,
                    "agent_name": matched_agent.name,
                    "execution_id": exec_res.id,
                    "citations_count": len(exec_res.citations or []),
                },
            )

            source_attribution = (
                exec_res.source_attribution
                if getattr(exec_res, "source_attribution", None)
                else source_attribution_from_citations(exec_res.citations)
            )
            follow_up_questions = exec_res.follow_up_questions or []

            message_obj = {
                "id": assistant_message.id,
                "chat_id": assistant_message.chat_id,
                "role": assistant_message.role,
                "content": assistant_message.content,
                "workspace_id": assistant_message.workspace_id,
                "document_ids": assistant_message.document_ids,
                "source_documents": exec_res.source_documents,
                "citations": exec_res.citations or [],
                "sources": exec_res.citations or [],
                "source_attribution": source_attribution,
                "follow_up_questions": follow_up_questions,
                "skills_used": assistant_message.skills_used,
                "skills": assistant_message.skills_used,
                "mode_used": mode_used,
                "agent_id": matched_agent.id,
                "agent_name": matched_agent.name,
                "agents": [{"id": matched_agent.id, "name": matched_agent.name, "isWorkflow": True}],
                "execution_id": exec_res.id,
                "created_at": assistant_message.created_at.isoformat()
                if assistant_message.created_at
                else None,
                "updated_at": assistant_message.updated_at.isoformat()
                if assistant_message.updated_at
                else None,
            }

            final_data = {
                "success": True,
                "error": None,
                "data": {
                    "chat_session_id": chat_id,
                    "user_message_id": user_message.id,
                    "assistant_message_id": assistant_message.id,
                    "response": final_text,
                    "content": final_text,
                    "final_response": final_text,
                    "agent_workflow_id": matched_agent.id,
                    "agent_id": matched_agent.id,
                    "agent_name": matched_agent.name,
                    "agents": [{"id": matched_agent.id, "name": matched_agent.name, "isWorkflow": True}],
                    "agent_execution_id": exec_res.id,
                    "document_ids": unique_doc_ids,
                    "active_document_ids": unique_doc_ids,
                    "source_documents": exec_res.source_documents,
                    "citations": exec_res.citations or [],
                    "sources": exec_res.citations or [],
                    "source_attribution": source_attribution,
                    "follow_up_questions": follow_up_questions,
                    "skills_used": assistant_message.skills_used,
                    "skills": assistant_message.skills_used,
                    "mode_used": mode_used,
                    "message": message_obj,
                },
                "message": f"Agent '{matched_agent.name}' automatically triggered and executed.",
            }

            yield f"data: {json.dumps({'type': 'complete', 'data': final_data})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")



# Singleton
agent_trigger_service = AgentTriggerService()


def get_agent_trigger_service() -> AgentTriggerService:
    return agent_trigger_service
