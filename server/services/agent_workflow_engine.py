"""Core Agent Execution Engine.

Implements the decoupled, multi-step Agent cognitive loop:
  Receive request
  → Determine trigger / context
  → Break task into steps (planning)
  → Select next step
  → Check for relevant skill
  → If skill exists, load skill + tools
  → If no skill exists, continue without skill
  → Execute step using skill/tools/context
  → Check whether step succeeded
  → If failed, retry/handle error according to configured retry strategy
  → If successful, update shared state
  → Check for more steps
  → Continue until all steps are complete
  → Final synthesis
  → Final response
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from openai import OpenAI
from sqlalchemy.orm import Session

from server import crud
from server.config import settings
from server.models.agent_workflow import (
    AgentStatus,
    AgentWorkflow,
    AgentWorkflowExecution,
    AgentWorkflowStepLog,
    AIExecutionMode,
    ExecutionStatus,
    StepExecutionStatus,
)
from server.models.skill import Skill
from server.schemas.agent_workflow import (
    AgentStepPlanItem,
    AgentWorkflowExecutionResponse,
    AgentWorkflowStepLogResponse,
)
from server.services.agent_tool_registry import AgentToolRegistry, get_agent_tool_registry
from server.services.audit_logger import AuditLogger
from server.services.citation_service import get_citation_service
from server.services.skill_instruction_service import SkillInstructionService
from server.services.skill_selection_service import SkillSelectionService

logger = logging.getLogger(__name__)


@dataclass
class AgentSharedState:
    """Blackboard maintaining shared memory, facts, outputs, and context across workflow steps."""

    execution_id: str
    workflow_id: Optional[str]
    workflow_name: str
    user_id: str
    workspace_id: Optional[str]
    trigger_source: str
    query_or_goal: str
    agent_instructions: str
    execution_mode: str
    document_ids: List[str] = field(default_factory=list)
    limit_docs_past_7_days: bool = False
    selected_sources: List[str] = field(default_factory=list)
    custom_sources: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Dynamic mutable state accumulated across steps
    step_results: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    source_documents: Set[str] = field(default_factory=set)
    skills_applied: List[str] = field(default_factory=list)
    tools_called: List[str] = field(default_factory=list)
    shared_variables: Dict[str, Any] = field(default_factory=dict)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize shared state for database snapshot."""
        return {
            "execution_id": self.execution_id,
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "trigger_source": self.trigger_source,
            "query_or_goal": self.query_or_goal,
            "execution_mode": self.execution_mode,
            "limit_docs_past_7_days": self.limit_docs_past_7_days,
            "selected_sources": self.selected_sources,
            "skills_applied": list(dict.fromkeys(self.skills_applied)),
            "tools_called": list(dict.fromkeys(self.tools_called)),
            "source_documents": list(self.source_documents),
            "step_count": len(self.step_results),
            "shared_variables": self.shared_variables,
            "errors": self.errors,
        }


class AgentWorkflowEngine:
    """Reusable, trigger-agnostic core execution engine for Agent Workflows."""

    def __init__(
        self,
        openai_client: Optional[OpenAI] = None,
        tool_registry: Optional[AgentToolRegistry] = None,
        skill_selector: Optional[SkillSelectionService] = None,
        skill_instructor: Optional[SkillInstructionService] = None,
    ):
        self.openai_client = openai_client or OpenAI(api_key=settings.OPENAI_API_KEY)
        self.tool_registry = tool_registry or get_agent_tool_registry()
        self.skill_selector = skill_selector or SkillSelectionService(client=self.openai_client)
        self.skill_instructor = skill_instructor or SkillInstructionService()

    # =========================================================================
    # Step 1 & 2: Receive Request & Context Initialization
    # =========================================================================

    def initialize_shared_state(
        self,
        db: Session,
        *,
        workflow: Optional[AgentWorkflow],
        user_id: str,
        query: Optional[str],
        document_ids: Optional[List[str]],
        filter_past_7_days: Optional[bool],
        trigger_source: str,
        parameters: Optional[Dict[str, Any]],
    ) -> Tuple[AgentSharedState, AgentWorkflowExecution]:
        """Determine trigger context, initialize shared state blackboard, and persist execution row."""
        execution_id = str(uuid.uuid4())
        parameters = parameters or {}

        # Resolve query/goal and agent metadata
        if workflow:
            workflow_id = workflow.id
            workflow_name = workflow.name
            workspace_id = workflow.workspace_id
            agent_instructions = workflow.instructions or ""
            execution_mode = (
                workflow.execution_mode.value
                if hasattr(workflow.execution_mode, "value")
                else str(workflow.execution_mode or "auto")
            )
            limit_docs_past_7_days = (
                filter_past_7_days
                if filter_past_7_days is not None
                else bool(workflow.limit_docs_past_7_days)
            )
            selected_sources = list(workflow.selected_sources or [])
            custom_sources = list(workflow.custom_sources or [])
            bound_docs = list(workflow.selected_document_ids or [])
            all_doc_ids = list(dict.fromkeys(bound_docs + (document_ids or [])))
            effective_query = query.strip() if query and query.strip() else (agent_instructions or workflow_name)
        else:
            workflow_id = None
            workflow_name = parameters.get("agent_name", "Ad-hoc Agent")
            workspace_id = parameters.get("workspace_id")
            agent_instructions = parameters.get("instructions", "")
            execution_mode = parameters.get("execution_mode", "auto")
            limit_docs_past_7_days = bool(filter_past_7_days)
            selected_sources = parameters.get("selected_sources", ["workspace_documents", "sec_edgar"])
            custom_sources = parameters.get("custom_sources", [])
            all_doc_ids = document_ids or []
            effective_query = query.strip() if query and query.strip() else agent_instructions

        shared_state = AgentSharedState(
            execution_id=execution_id,
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            user_id=str(user_id),
            workspace_id=workspace_id,
            trigger_source=trigger_source or "manual",
            query_or_goal=effective_query,
            agent_instructions=agent_instructions,
            execution_mode=execution_mode,
            document_ids=all_doc_ids,
            limit_docs_past_7_days=limit_docs_past_7_days,
            selected_sources=selected_sources,
            custom_sources=custom_sources,
            parameters=parameters,
        )

        execution_db = AgentWorkflowExecution(
            id=execution_id,
            workflow_id=workflow_id,
            user_id=str(user_id),
            workspace_id=workspace_id,
            trigger_source=shared_state.trigger_source,
            status=ExecutionStatus.RUNNING.value,
            query_or_goal=shared_state.query_or_goal,
            plan_steps=[],
            started_at=datetime.utcnow(),
            created_at=datetime.utcnow(),
        )

        db.add(execution_db)
        db.commit()
        db.refresh(execution_db)

        return shared_state, execution_db

    # =========================================================================
    # Step 3: Break Task into Steps (Planning)
    # =========================================================================

    def plan_workflow_steps(
        self, shared_state: AgentSharedState, db: Session
    ) -> List[AgentStepPlanItem]:
        """Decompose the agent goal and instructions into sequential executable steps."""
        available_tools = self.tool_registry.list_tools()
        active_tools = [
            t for t in available_tools if t["name"] in shared_state.selected_sources or t["name"] == "rag"
        ]

        system_prompt = (
            "You are an expert AI Workflow Planner. Your job is to break down a business analysis task "
            "into 1 to 4 logical, sequential, atomic execution steps.\n"
            "Respond strictly with valid JSON with the format: {'plan': [{'step_number': 1, 'step_name': '...', 'description': '...', 'tool_hint': '...', 'skill_hint': '...'}]}"
        )

        instructions_line = (
            f"Agent Instructions: {shared_state.agent_instructions}\n"
            if shared_state.agent_instructions
            else "Agent Instructions: None\n"
        )

        user_content = (
            f"Agent Name: {shared_state.workflow_name}\n"
            f"{instructions_line}"
            f"Execution Query/Goal: {shared_state.query_or_goal}\n"
            f"Limit Documents to Past 7 Days: {shared_state.limit_docs_past_7_days}\n"
            f"Active Tools: {json.dumps(active_tools)}\n\n"
            "Generate a sequential plan to fulfill the objective."
        )

        try:
            response = self.openai_client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            parsed = json.loads(response.choices[0].message.content or "{}")
            raw_steps = parsed.get("plan") or parsed.get("steps") or []
            if raw_steps:
                plan_items = []
                for i, s in enumerate(raw_steps, start=1):
                    plan_items.append(
                        AgentStepPlanItem(
                            step_number=i,
                            step_name=s.get("step_name") or f"Step {i}",
                            description=s.get("description") or str(s),
                            tool_hint=s.get("tool_hint"),
                            skill_hint=s.get("skill_hint"),
                        )
                    )
                return plan_items
        except Exception as e:
            logger.warning(f"LLM planning fallback triggered: {str(e)}")

        # Deterministic Fallback Plan
        fallback_steps = []
        step_num = 1
        if "workspace_documents" in shared_state.selected_sources or "rag" in shared_state.selected_sources or shared_state.document_ids:
            fallback_steps.append(
                AgentStepPlanItem(
                    step_number=step_num,
                    step_name="Retrieve Document Context",
                    description="Retrieve relevant text chunks and tables from workspace documents.",
                    tool_hint="rag",
                )
            )
            step_num += 1

        for src in shared_state.selected_sources:
            if src in ("sec_edgar", "roic_ai", "financial_news", "companies_house") and src not in [s.tool_hint for s in fallback_steps]:
                fallback_steps.append(
                    AgentStepPlanItem(
                        step_number=step_num,
                        step_name=f"Query {src.replace('_', ' ').title()}",
                        description=f"Fetch external financial or market data using {src}.",
                        tool_hint=src,
                    )
                )
                step_num += 1

        fallback_steps.append(
            AgentStepPlanItem(
                step_number=step_num,
                step_name="Analyze and Synthesize",
                description="Synthesize gathered findings into an executive report with citations.",
                tool_hint=None,
            )
        )
        return fallback_steps

    # =========================================================================
    # Step 4, 5, 6: Check for Relevant Skill & Load Skill + Tools
    # =========================================================================

    def resolve_skill_for_step(
        self, db: Session, step: AgentStepPlanItem, shared_state: AgentSharedState
    ) -> Tuple[Optional[Skill], str]:
        """Evaluate if a relevant skill exists for the step and load its complete instructions."""
        matched_skill: Optional[Skill] = None

        # 1. Match by explicit skill hint if provided in plan
        if step.skill_hint:
            candidates = self.skill_selector.get_candidate_skills(db, user_id=shared_state.user_id)
            for s in candidates:
                if s.name.lower() == step.skill_hint.lower() or s.id == step.skill_hint:
                    matched_skill = s
                    break

        # 2. Match automatically via SkillSelectionService if not yet matched
        if not matched_skill:
            step_query = f"{step.step_name}: {step.description} (Goal: {shared_state.query_or_goal})"
            selected = self.skill_selector.select_skills(
                db, user_id=shared_state.user_id, query=step_query, max_skills=1
            )
            if selected:
                matched_skill = selected[0]

        # 3. Load instructions if skill found
        if matched_skill:
            instructions = self.skill_instructor.load_skill_instructions(matched_skill)
            if matched_skill.name not in shared_state.skills_applied:
                shared_state.skills_applied.append(matched_skill.name)
            return matched_skill, instructions

        return None, ""

    # =========================================================================
    # Step 7, 8, 9, 10: Execute Step using Skill/Tools/Context + Retry Handling
    # =========================================================================

    async def execute_step_with_retry(
        self,
        db: Session,
        step: AgentStepPlanItem,
        shared_state: AgentSharedState,
        step_log: AgentWorkflowStepLog,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """Execute step using skills, tools, and shared context with configurable retry strategy."""
        # 1. Check and load skill
        skill, skill_instructions = self.resolve_skill_for_step(db, step, shared_state)
        step_log.skill_name = skill.name if skill else None

        # 2. Prepare tool context
        tool_context = {
            "db": db,
            "workspace_id": shared_state.workspace_id,
            "user_id": shared_state.user_id,
            "document_ids": shared_state.document_ids,
            "limit_docs_past_7_days": shared_state.limit_docs_past_7_days,
            "shared_variables": shared_state.shared_variables,
        }

        # 3. Retry loop
        attempt = 0
        last_error = None
        start_time = time.time()

        while attempt <= max_retries:
            try:
                # A. Tool Execution (if tool_hint present or step requires external data)
                tool_output = None
                tool_name = step.tool_hint
                if tool_name:
                    tool = self.tool_registry.get_tool(tool_name)
                    if tool:
                        step_log.tool_name = tool_name
                        tool_output = await tool.execute(
                            query=shared_state.query_or_goal,
                            context=tool_context,
                        )
                        if tool_name not in shared_state.tools_called:
                            shared_state.tools_called.append(tool_name)

                # B. Execute Custom Sources if configured and not yet executed
                custom_outputs = []
                for cs in shared_state.custom_sources:
                    c_res = await self.tool_registry.execute_custom_source(
                        custom_source=cs, query=shared_state.query_or_goal, context=tool_context
                    )
                    custom_outputs.append(c_res)
                    if cs.get("name") and cs.get("name") not in shared_state.tools_called:
                        shared_state.tools_called.append(cs.get("name"))

                # C. LLM Step Reasoning (augmenting step with skill + tool output + shared state)
                step_prompt = (
                    f"Execute Workflow Step {step.step_number}: {step.step_name}\n"
                    f"Objective: {step.description}\n"
                    f"Overall Goal: {shared_state.query_or_goal}\n"
                )
                if shared_state.agent_instructions:
                    step_prompt += f"Agent Instructions: {shared_state.agent_instructions}\n"
                if skill_instructions:
                    step_prompt += f"\nActive Skill Guidance ({skill.name}):\n{skill_instructions}\n"
                if tool_output:
                    step_prompt += f"\nTool Data ({tool_name}):\n{json.dumps(tool_output, default=str)[:3000]}\n"
                if custom_outputs:
                    step_prompt += f"\nCustom Source Data:\n{json.dumps(custom_outputs, default=str)[:2000]}\n"

                llm_response = self.openai_client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an analytical AI workflow step worker. Perform the requested task step accurately and concisely.",
                        },
                        {"role": "user", "content": step_prompt},
                    ],
                    temperature=0.2,
                )
                reasoning_output = llm_response.choices[0].message.content or ""

                duration_ms = int((time.time() - start_time) * 1000)

                # Assemble step outcome
                step_result = {
                    "step_number": step.step_number,
                    "step_name": step.step_name,
                    "status": "completed",
                    "skill_used": skill.name if skill else None,
                    "tool_used": tool_name,
                    "tool_output": tool_output,
                    "reasoning_output": reasoning_output,
                    "duration_ms": duration_ms,
                }

                # Update database step log
                step_log.status = StepExecutionStatus.COMPLETED.value
                step_log.output_result = step_result
                step_log.latency_ms = duration_ms
                step_log.retry_count = attempt
                db.add(step_log)
                db.commit()

                return step_result

            except Exception as e:
                attempt += 1
                last_error = str(e)
                logger.warning(
                    f"Step {step.step_number} ({step.step_name}) attempt {attempt} failed: {last_error}"
                )
                step_log.retry_count = attempt
                step_log.status = StepExecutionStatus.RETRYING.value if attempt <= max_retries else StepExecutionStatus.FAILED.value
                db.add(step_log)
                db.commit()

                if attempt <= max_retries:
                    # Exponential backoff delay: 0.5s, 1.0s, etc.
                    await asyncio.sleep(0.5 * (2 ** (attempt - 1)))

        # All retries exhausted
        duration_ms = int((time.time() - start_time) * 1000)
        step_log.status = StepExecutionStatus.FAILED.value
        step_log.error_message = last_error
        step_log.latency_ms = duration_ms
        db.add(step_log)
        db.commit()

        return {
            "step_number": step.step_number,
            "step_name": step.step_name,
            "status": "failed",
            "error": last_error,
            "duration_ms": duration_ms,
        }

    # =========================================================================
    # Step 11: Update Shared State
    # =========================================================================

    def update_shared_state_after_step(
        self, shared_state: AgentSharedState, step_result: Dict[str, Any]
    ) -> None:
        """Merge step outcomes, retrieved documents, tool facts, and insights into shared blackboard."""
        step_num = step_result.get("step_number", len(shared_state.step_results) + 1)
        shared_state.step_results[step_num] = step_result

        # If step produced tool output, incorporate into state
        tool_output = step_result.get("tool_output")
        if isinstance(tool_output, dict):
            # Incorporate RAG chunks & source documents
            if tool_output.get("chunks"):
                shared_state.retrieved_chunks.extend(tool_output.get("chunks", []))
            if tool_output.get("source_documents"):
                for doc in tool_output.get("source_documents", []):
                    shared_state.source_documents.add(doc)

            # Incorporate financial variables
            if tool_output.get("tool") == "sec_edgar" and "data" in tool_output:
                shared_state.shared_variables["sec_facts"] = tool_output["data"]
            if tool_output.get("tool") == "roic_ai" and "data" in tool_output:
                shared_state.shared_variables["roic_data"] = tool_output["data"]

        # Record step reasoning in shared variables
        if step_result.get("reasoning_output"):
            shared_state.shared_variables[f"step_{step_num}_summary"] = step_result["reasoning_output"]

        # Record failure warning if failed
        if step_result.get("status") == "failed":
            shared_state.errors.append({
                "step_number": step_num,
                "step_name": step_result.get("step_name"),
                "error": step_result.get("error"),
            })

    # =========================================================================
    # Step 13 & 14: Final Synthesis & Response
    # =========================================================================

    def synthesize_final_response(
        self, shared_state: AgentSharedState
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Synthesize accumulated findings into a structured response with citations."""
        # Determine model and reasoning based on execution_mode
        mode = shared_state.execution_mode
        if mode == AIExecutionMode.THINK_LONGER.value:
            model = settings.THINK_LONGER_MODEL
            kwargs = {}
        elif mode == AIExecutionMode.DEEP_RESEARCH.value:
            model = settings.DEEP_RESEARCH_MODEL
            kwargs = {}
        else:
            model = settings.LLM_MODEL
            kwargs = {}

        # Compile step summaries and evidence
        steps_summary = []
        for num, res in sorted(shared_state.step_results.items()):
            steps_summary.append(
                f"### Step {num}: {res.get('step_name')}\n"
                f"- Outcome: {res.get('status')}\n"
                f"- Findings: {res.get('reasoning_output', 'No textual findings.')}\n"
            )

        context_block = "\n\n".join(steps_summary)

        # Include retrieved document context directly in the final synthesis prompt
        doc_context = ""
        if shared_state.retrieved_chunks:
            doc_snippets = []
            for i, chunk in enumerate(shared_state.retrieved_chunks[:10], start=1):
                if isinstance(chunk, dict):
                    c_text = chunk.get("content", "").strip()
                    c_file = chunk.get("file_name", "Document")
                    if c_text:
                        doc_snippets.append(f"[{i}] ({c_file}): {c_text}")
                elif hasattr(chunk, "content") and chunk.content:
                    doc_snippets.append(f"[{i}] ({getattr(chunk, 'file_name', 'Document')}): {chunk.content.strip()}")
            if doc_snippets:
                doc_context = "\n\n--- Retrieved Document Context ---\n" + "\n\n".join(doc_snippets)

        system_prompt = (
            "You are a state-of-the-art AI Financial and Workflow Research Analyst. "
            "Synthesize the provided step findings, data facts, and document context into a "
            "comprehensive, executive-ready final report. Use clear markdown headings, bullet points, "
            "and cite all source data accurately. Adhere strictly to the agent instructions."
        )

        instructions_block = (
            f"Agent Instructions: {shared_state.agent_instructions}\n"
            if shared_state.agent_instructions
            else "Agent Instructions: None\n"
        )

        user_content = (
            f"Agent Name: {shared_state.workflow_name}\n"
            f"{instructions_block}"
            f"Goal / Query: {shared_state.query_or_goal}\n"
            f"Execution Mode: {shared_state.execution_mode}\n"
            f"Limit Docs Past 7 Days: {shared_state.limit_docs_past_7_days}\n"
            f"Skills Applied: {', '.join(shared_state.skills_applied) if shared_state.skills_applied else 'None'}\n"
            f"Tools Called: {', '.join(shared_state.tools_called) if shared_state.tools_called else 'None'}\n"
            f"Source Documents: {', '.join(shared_state.source_documents) if shared_state.source_documents else 'None'}\n\n"
            f"--- Step Execution Findings ---\n{context_block}\n"
            f"{doc_context}\n\n"
            "Provide the final executive synthesis now, ensuring all instructions and formatting guidelines are honored."
        )

        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.3,
                **kwargs,
            )
            final_text = response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Synthesis LLM call failed: {str(e)}", exc_info=True)
            final_text = (
                f"## Execution Summary for {shared_state.workflow_name}\n\n"
                f"Goal: {shared_state.query_or_goal}\n\n"
                f"{context_block}\n\n"
                f"*(Note: LLM synthesis fallback was used due to error: {str(e)})*"
            )

        # Generate / extract citations
        citations = []
        try:
            citation_service = get_citation_service()
            if shared_state.retrieved_chunks:
                rewritten_text, extracted_citations = citation_service.generate_citations(
                    final_text, sources=shared_state.retrieved_chunks
                )
                if extracted_citations:
                    final_text = rewritten_text
                    citations = extracted_citations
                else:
                    citations = citation_service.generate_citations_from_chunks(
                        sources=shared_state.retrieved_chunks
                    )
            elif hasattr(citation_service, "generate_citations"):
                _, citations = citation_service.generate_citations(final_text, sources=[])
        except Exception as e:
            logger.warning(f"Citation generation in workflow synthesis failed: {e}")

        return final_text, citations

    # =========================================================================
    # Main Orchestrator Entry Point (Trigger Agnostic)
    # =========================================================================

    async def execute_workflow(
        self,
        db: Session,
        *,
        workflow_id: Optional[str] = None,
        workflow: Optional[AgentWorkflow] = None,
        user_id: str,
        query: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        filter_past_7_days: Optional[bool] = None,
        trigger_source: str = "manual",
        parameters: Optional[Dict[str, Any]] = None,
        max_retries_per_step: int = 2,
        status_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
        cancellation_check: Optional[Callable[[], bool]] = None,
    ) -> AgentWorkflowExecutionResponse:
        """Run the complete Agent Workflow execution loop from start to finish with live status events."""
        start_time = time.time()

        async def emit_status(
            status: str,
            message: str,
            step_number: Optional[int] = None,
            step_name: Optional[str] = None,
            tool: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None,
        ):
            if not status_callback:
                return
            event_payload = {
                "type": "status",
                "status": status,
                "step": message,
                "message": message,
                "step_number": step_number,
                "step_name": step_name,
                "tool": tool,
                "details": details or {},
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            try:
                if asyncio.iscoroutinefunction(status_callback):
                    await status_callback(event_payload)
                else:
                    res = status_callback(event_payload)
                    if asyncio.iscoroutine(res):
                        await res
            except Exception as _cb_err:
                logger.debug(f"Error in status_callback: {_cb_err}")

        # 1. Resolve workflow entity if ID passed
        if workflow_id and not workflow:
            workflow = crud.agent_workflow.get_by_user_and_id(
                db, user_id=str(user_id), agent_id=workflow_id
            )

        # 2. Initialize Shared State & DB execution record
        shared_state, execution_db = self.initialize_shared_state(
            db,
            workflow=workflow,
            user_id=str(user_id),
            query=query,
            document_ids=document_ids,
            filter_past_7_days=filter_past_7_days,
            trigger_source=trigger_source,
            parameters=parameters,
        )

        try:
            # Emit agent_started event
            await emit_status(
                status="agent_started",
                message=f"Starting agent '{shared_state.workflow_name}'...",
                details={
                    "agent_id": workflow_id or (workflow.id if workflow else None),
                    "execution_mode": shared_state.execution_mode,
                },
            )

            # 3. Break task into steps (Planning)
            await emit_status(
                status="planning",
                message="Formulating execution plan...",
            )
            plan_items = self.plan_workflow_steps(shared_state, db)
            execution_db.plan_steps = [item.model_dump() for item in plan_items]
            db.add(execution_db)
            db.commit()

            # 4. Step Execution Loop
            for item in plan_items:
                # Check cancellation
                if cancellation_check and cancellation_check():
                    await emit_status(
                        status="cancelled",
                        message=f"Execution cancelled before step {item.step_number}.",
                    )
                    raise asyncio.CancelledError("Agent execution was cancelled by user.")

                step_log = AgentWorkflowStepLog(
                    execution_id=execution_db.id,
                    step_number=item.step_number,
                    step_name=item.step_name,
                    step_type="step",
                    status=StepExecutionStatus.RUNNING.value,
                    input_context={
                        "description": item.description,
                        "tool_hint": item.tool_hint,
                        "skill_hint": item.skill_hint,
                    },
                    created_at=datetime.utcnow(),
                )
                db.add(step_log)
                db.commit()
                db.refresh(step_log)

                # Emit step-specific live status before execution
                is_rag_step = item.tool_hint in ("rag", "workspace_documents") or (
                    bool(shared_state.document_ids) and "document" in item.step_name.lower()
                )
                if is_rag_step:
                    status_name = "retrieving_documents"
                    step_msg = f"Retrieving document context for Step {item.step_number}: {item.step_name}..."
                    tool_name_hint = "rag"
                elif item.tool_hint:
                    status_name = "executing_tool"
                    step_msg = f"Executing tool '{item.tool_hint}' for Step {item.step_number}: {item.step_name}..."
                    tool_name_hint = item.tool_hint
                else:
                    status_name = "processing_results"
                    step_msg = f"Processing Step {item.step_number}: {item.step_name}..."
                    tool_name_hint = None

                await emit_status(
                    status=status_name,
                    step_number=item.step_number,
                    step_name=item.step_name,
                    tool=tool_name_hint,
                    message=step_msg,
                )

                # Execute with skill, tool, context & retries
                step_result = await self.execute_step_with_retry(
                    db,
                    step=item,
                    shared_state=shared_state,
                    step_log=step_log,
                    max_retries=max_retries_per_step,
                )

                # Update Shared State
                self.update_shared_state_after_step(shared_state, step_result)

                # Emit status event after step completes
                await emit_status(
                    status="processing_results",
                    step_number=item.step_number,
                    step_name=item.step_name,
                    message=f"Completed Step {item.step_number}: {item.step_name}.",
                )

            # 5. Final Synthesis
            await emit_status(
                status="generating_response",
                message="Synthesizing final executive response and citations...",
            )
            final_response, citations = self.synthesize_final_response(shared_state)
            shared_state.citations = citations

            # 6. Complete Execution Record
            total_duration_ms = int((time.time() - start_time) * 1000)
            execution_db.status = ExecutionStatus.COMPLETED.value
            execution_db.final_response = final_response
            execution_db.citations = citations
            execution_db.source_documents = list(shared_state.source_documents)
            execution_db.skills_applied = list(dict.fromkeys(shared_state.skills_applied))
            execution_db.tools_called = list(dict.fromkeys(shared_state.tools_called))
            execution_db.shared_state = shared_state.to_dict()
            execution_db.execution_time_ms = total_duration_ms
            execution_db.completed_at = datetime.utcnow()

            if workflow:
                workflow.last_run_at = datetime.utcnow()
                db.add(workflow)

            db.add(execution_db)
            db.commit()
            db.refresh(execution_db)

            # Audit log
            AuditLogger.log(
                action="AGENT_EXECUTION_COMPLETED",
                user_id=str(user_id),
                workspace_id=shared_state.workspace_id,
                endpoint="/v1/agents/execute",
                http_method="POST",
                status="200",
                processing_time=total_duration_ms,
                metadata={
                    "execution_id": execution_db.id,
                    "workflow_id": workflow_id,
                    "trigger_source": trigger_source,
                    "steps_count": len(plan_items),
                },
            )

            # Emit complete status event
            await emit_status(
                status="complete",
                message=f"Agent '{shared_state.workflow_name}' completed.",
                details={
                    "execution_id": execution_db.id,
                    "execution_time_ms": total_duration_ms,
                    "source_documents": list(shared_state.source_documents),
                    "citations_count": len(citations),
                },
            )

            # Load step logs for response
            step_logs_db = crud.agent_workflow.get_step_logs_by_execution(
                db, execution_id=execution_db.id
            )
            step_log_responses = [
                AgentWorkflowStepLogResponse.model_validate(sl) for sl in step_logs_db
            ]

            response_obj = AgentWorkflowExecutionResponse.model_validate(execution_db)
            response_obj.step_logs = step_log_responses
            return response_obj

        except asyncio.CancelledError:
            total_duration_ms = int((time.time() - start_time) * 1000)
            logger.info("Agent execution was cancelled.")
            execution_db.status = ExecutionStatus.FAILED.value
            execution_db.error_message = "Execution cancelled."
            execution_db.execution_time_ms = total_duration_ms
            execution_db.completed_at = datetime.utcnow()
            db.add(execution_db)
            db.commit()
            db.refresh(execution_db)

            await emit_status(
                status="cancelled",
                message="Agent workflow execution was cancelled.",
            )

            step_logs_db = crud.agent_workflow.get_step_logs_by_execution(
                db, execution_id=execution_db.id
            )
            response_obj = AgentWorkflowExecutionResponse.model_validate(execution_db)
            response_obj.step_logs = [
                AgentWorkflowStepLogResponse.model_validate(sl) for sl in step_logs_db
            ]
            return response_obj

        except Exception as e:
            total_duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Agent execution failed: {str(e)}", exc_info=True)
            execution_db.status = ExecutionStatus.FAILED.value
            execution_db.error_message = str(e)
            execution_db.execution_time_ms = total_duration_ms
            execution_db.completed_at = datetime.utcnow()
            db.add(execution_db)
            db.commit()
            db.refresh(execution_db)

            await emit_status(
                status="error",
                message=f"Agent execution failed: {str(e)}",
                details={"error": str(e)},
            )

            AuditLogger.log(
                action="AGENT_EXECUTION_FAILED",
                user_id=str(user_id),
                workspace_id=shared_state.workspace_id,
                endpoint="/v1/agents/execute",
                http_method="POST",
                status="500",
                processing_time=total_duration_ms,
                metadata={
                    "execution_id": execution_db.id,
                    "workflow_id": workflow_id,
                    "error": str(e),
                },
            )

            step_logs_db = crud.agent_workflow.get_step_logs_by_execution(
                db, execution_id=execution_db.id
            )
            response_obj = AgentWorkflowExecutionResponse.model_validate(execution_db)
            response_obj.step_logs = [
                AgentWorkflowStepLogResponse.model_validate(sl) for sl in step_logs_db
            ]
            return response_obj


# Global Singleton Engine
agent_workflow_engine = AgentWorkflowEngine()


def get_agent_workflow_engine() -> AgentWorkflowEngine:
    return agent_workflow_engine
