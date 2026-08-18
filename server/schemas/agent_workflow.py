from datetime import datetime
from enum import Enum as PyEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TriggerType(str, PyEnum):
    AUTOMATIC = "automatic"
    SCHEDULED = "scheduled"
    BOTH = "both"


class ScheduleFrequency(str, PyEnum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    HOURLY = "hourly"
    CUSTOM = "custom"


class AIExecutionMode(str, PyEnum):
    AUTO = "auto"
    THINK_LONGER = "think_longer"
    DEEP_RESEARCH = "deep_research"


class AgentStatus(str, PyEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    INACTIVE = "inactive"
    ARCHIVED = "archived"


class AgentWorkflowBase(BaseModel):
    name: str = Field(
        ..., description="Name of the agent", min_length=1, max_length=255
    )
    description: Optional[str] = Field(
        None, description="Detailed description of the agent"
    )
    instructions: Optional[str] = Field(
        None, description="System instructions / prompt guiding the agent"
    )

    # Trigger configuration
    trigger_type: TriggerType = Field(
        default=TriggerType.AUTOMATIC,
        description="Trigger mechanism: automatic, scheduled, or both",
    )

    # Schedule configuration
    schedule_frequency: Optional[str] = Field(
        None, description="Schedule frequency: daily, weekly, monthly, hourly, custom"
    )
    schedule_day: Optional[str] = Field(
        None, description="Day of schedule (e.g. monday, tuesday or 1, 15)"
    )
    schedule_time: Optional[str] = Field(
        None, description="Execution time (e.g. 09:00 or 09:00-17:00)"
    )
    schedule_start_time: Optional[str] = Field(
        None, description="Schedule start time in 24-hour HH:MM format"
    )
    schedule_end_time: Optional[str] = Field(
        None, description="Schedule end time in 24-hour HH:MM format"
    )
    schedule_timezone: Optional[str] = Field(
        default="UTC", description="Timezone for scheduling (default UTC)"
    )
    schedule_cron: Optional[str] = Field(
        None, description="Computed or custom 5-part cron expression"
    )
    scheduled_count_limit: Optional[int] = Field(
        None, description="Maximum number of scheduled executions", ge=1
    )

    # Source & document filtering
    limit_docs_past_7_days: bool = Field(
        default=False,
        description="Whether to limit retrieved documents to the past 7 days",
    )
    limit_documents_to_past_7_days: Optional[bool] = Field(
        default=None, description="Alias for limit_docs_past_7_days"
    )
    selected_document_ids: Optional[List[str]] = Field(
        default_factory=list,
        description="IDs of specific workspace documents assigned to this agent",
    )
    selected_sources: Optional[List[str]] = Field(
        default_factory=list,
        description="Enabled tool/data sources (e.g. workspace_documents, sec_edgar, financial_news, roic_ai, companies_house, web_search)",
    )
    custom_sources: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list, description="Custom external sources or API integrations"
    )

    # AI Execution Mode
    execution_mode: AIExecutionMode = Field(
        default=AIExecutionMode.AUTO,
        description="AI reasoning mode: auto, think_longer, deep_research",
    )

    # Status & Workspace Scope
    status: AgentStatus = Field(
        default=AgentStatus.DRAFT,
        description="Agent lifecycle status: draft, active, paused, archived",
    )
    workspace_id: Optional[str] = Field(
        None, description="Optional Workspace ID to scope this agent"
    )
    version: str = Field(default="1.0.0", description="Agent version string")

    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_schedule_and_limits(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # 1. Sync duplicate limit fields
            if (
                "limit_documents_to_past_7_days" in data
                and "limit_docs_past_7_days" not in data
            ):
                data["limit_docs_past_7_days"] = bool(
                    data["limit_documents_to_past_7_days"]
                )
            elif (
                "limit_docs_past_7_days" in data
                and "limit_documents_to_past_7_days" not in data
            ):
                data["limit_documents_to_past_7_days"] = bool(
                    data["limit_docs_past_7_days"]
                )
            elif (
                "limit_docs_past_7_days" in data
                and "limit_documents_to_past_7_days" in data
            ):
                if data["limit_documents_to_past_7_days"] is not None and not data.get(
                    "limit_docs_past_7_days"
                ):
                    data["limit_docs_past_7_days"] = bool(
                        data["limit_documents_to_past_7_days"]
                    )
                data["limit_documents_to_past_7_days"] = bool(
                    data["limit_docs_past_7_days"]
                )

            # 2. Normalize times
            s_time = data.get("schedule_time")
            s_start = data.get("schedule_start_time")
            s_end = data.get("schedule_end_time")

            if s_start or s_end or s_time:
                from server.services.agent_validation_service import (
                    agent_validation_service,
                )

                norm_time, norm_start, norm_end, _ = (
                    agent_validation_service.normalize_time_string(
                        time_str=s_time,
                        start_time=s_start,
                        end_time=s_end,
                    )
                )
                if norm_time:
                    data["schedule_time"] = norm_time
                if norm_start:
                    data["schedule_start_time"] = norm_start
                if norm_end:
                    data["schedule_end_time"] = norm_end
        return data

    @field_validator("name")
    @classmethod
    def validate_name_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Agent name cannot be empty or whitespace only.")
        return v.strip()

    @field_validator("limit_docs_past_7_days", mode="before")
    @classmethod
    def default_limit_docs(cls, v: Any) -> bool:
        return False if v is None else bool(v)

    @field_validator("execution_mode", mode="before")
    @classmethod
    def default_execution_mode(cls, v: Any) -> AIExecutionMode:
        return AIExecutionMode.AUTO if v is None else v

    @field_validator("status", mode="before")
    @classmethod
    def default_status(cls, v: Any) -> AgentStatus:
        return AgentStatus.DRAFT if v is None else v

    @field_validator("trigger_type", mode="before")
    @classmethod
    def default_trigger_type(cls, v: Any) -> TriggerType:
        return TriggerType.AUTOMATIC if v is None else v

    @field_validator(
        "selected_document_ids", "selected_sources", "custom_sources", mode="before"
    )
    @classmethod
    def default_empty_lists(cls, v: Any) -> list:
        return [] if v is None else v


class AgentWorkflowCreate(AgentWorkflowBase):
    pass


class AgentWorkflowUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    instructions: Optional[str] = None
    trigger_type: Optional[TriggerType] = None
    schedule_frequency: Optional[str] = None
    schedule_day: Optional[str] = None
    schedule_time: Optional[str] = None
    schedule_start_time: Optional[str] = None
    schedule_end_time: Optional[str] = None
    schedule_timezone: Optional[str] = None
    schedule_cron: Optional[str] = None
    scheduled_count_limit: Optional[int] = Field(None, ge=1)
    limit_docs_past_7_days: Optional[bool] = None
    limit_documents_to_past_7_days: Optional[bool] = None
    selected_document_ids: Optional[List[str]] = None
    selected_sources: Optional[List[str]] = None
    custom_sources: Optional[List[Dict[str, Any]]] = None
    execution_mode: Optional[AIExecutionMode] = None
    status: Optional[AgentStatus] = None
    workspace_id: Optional[str] = None
    version: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_update_schedule_and_limits(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if (
                "limit_documents_to_past_7_days" in data
                and "limit_docs_past_7_days" not in data
            ):
                data["limit_docs_past_7_days"] = bool(
                    data["limit_documents_to_past_7_days"]
                )
            elif (
                "limit_docs_past_7_days" in data
                and "limit_documents_to_past_7_days" not in data
            ):
                data["limit_documents_to_past_7_days"] = bool(
                    data["limit_docs_past_7_days"]
                )

            s_time = data.get("schedule_time")
            s_start = data.get("schedule_start_time")
            s_end = data.get("schedule_end_time")

            if s_start or s_end or s_time:
                from server.services.agent_validation_service import (
                    agent_validation_service,
                )

                norm_time, norm_start, norm_end, _ = (
                    agent_validation_service.normalize_time_string(
                        time_str=s_time,
                        start_time=s_start,
                        end_time=s_end,
                    )
                )
                if norm_time:
                    data["schedule_time"] = norm_time
                if norm_start:
                    data["schedule_start_time"] = norm_start
                if norm_end:
                    data["schedule_end_time"] = norm_end
        return data

    @field_validator("name")
    @classmethod
    def validate_name_not_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            if not v or not v.strip():
                raise ValueError("Agent name cannot be empty or whitespace only.")
            return v.strip()
        return v


class AgentWorkflowResponse(AgentWorkflowBase):
    id: str
    user_id: str
    schedule_start_time: Optional[str] = None
    schedule_end_time: Optional[str] = None
    limit_documents_to_past_7_days: Optional[bool] = None
    description_html: Optional[str] = None
    instructions_html: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    activated_at: Optional[datetime] = None
    last_run_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class AgentWorkflowListItem(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    trigger_type: TriggerType
    schedule_frequency: Optional[str] = None
    schedule_day: Optional[str] = None
    schedule_time: Optional[str] = None
    schedule_start_time: Optional[str] = None
    schedule_end_time: Optional[str] = None
    schedule_timezone: Optional[str] = None
    execution_mode: AIExecutionMode
    status: AgentStatus
    version: str
    workspace_id: Optional[str] = None
    limit_docs_past_7_days: bool = False
    limit_documents_to_past_7_days: bool = False
    selected_document_count: int = 0
    selected_sources: Optional[List[str]] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    activated_at: Optional[datetime] = None
    last_run_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

    @field_validator("limit_docs_past_7_days", mode="before")
    @classmethod
    def default_list_limit_docs(cls, v: Any) -> bool:
        return False if v is None else bool(v)

    @field_validator("execution_mode", mode="before")
    @classmethod
    def default_list_execution_mode(cls, v: Any) -> AIExecutionMode:
        return AIExecutionMode.AUTO if v is None else v

    @field_validator("status", mode="before")
    @classmethod
    def default_list_status(cls, v: Any) -> AgentStatus:
        return AgentStatus.DRAFT if v is None else v

    @field_validator("trigger_type", mode="before")
    @classmethod
    def default_list_trigger_type(cls, v: Any) -> TriggerType:
        return TriggerType.AUTOMATIC if v is None else v

    @field_validator("selected_sources", mode="before")
    @classmethod
    def default_list_sources(cls, v: Any) -> list:
        return [] if v is None else v


# --- Validation Schemas ---


class ValidationIssue(BaseModel):
    field: str = Field(..., description="Field name with the issue")
    message: str = Field(..., description="Human-readable description of the issue")
    code: str = Field(..., description="Machine-readable issue code")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Optional extra details"
    )


class AgentValidationRequest(BaseModel):
    agent: AgentWorkflowBase = Field(
        ..., description="Agent workflow configuration to validate"
    )
    check_uniqueness: bool = Field(
        default=True,
        description="Whether to check name uniqueness against existing agents",
    )
    target_agent_id: Optional[str] = Field(
        default=None,
        description="Optional ID of existing agent to exclude from uniqueness check",
    )


class AgentValidationResponse(BaseModel):
    valid: bool = Field(..., description="Whether the configuration is valid")
    errors: List[ValidationIssue] = Field(
        default_factory=list,
        description="Critical errors that prevent activation or saving",
    )
    warnings: List[ValidationIssue] = Field(
        default_factory=list, description="Non-blocking warnings and recommendations"
    )


# --- Execution Schemas ---


class AgentExecutionRequest(BaseModel):
    """Payload for triggering an Agent Workflow execution."""

    query: Optional[str] = Field(
        None,
        description="Specific question or custom query for this run. If omitted, uses agent's configured instructions.",
    )
    document_ids: Optional[List[str]] = Field(
        None, description="Optional document IDs to scope retrieval for this execution"
    )
    filter_past_7_days: Optional[bool] = Field(
        None, description="Whether to limit retrieved documents to the past 7 days"
    )
    workspace_id: Optional[str] = Field(
        None, description="Optional workspace context ID"
    )
    chat_id: Optional[str] = Field(
        None, description="Optional chat session ID to associate this execution with"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Arbitrary custom parameters passed to the execution",
    )
    trigger_source: str = Field(
        default="manual",
        description="Source of the execution: manual, automatic, scheduled, api",
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Whether to stream execution status and tokens over SSE",
    )


class AgentStepPlanItem(BaseModel):
    step_number: int = Field(..., description="1-indexed sequence number")
    step_name: str = Field(..., description="Short title of the step")
    description: str = Field(
        ..., description="Detailed action description for the step"
    )
    tool_hint: Optional[str] = Field(
        None, description="Recommended tool to invoke, if any"
    )
    skill_hint: Optional[str] = Field(
        None, description="Recommended skill to apply, if any"
    )


class AgentWorkflowStepLogResponse(BaseModel):
    id: str
    execution_id: str
    step_number: int
    step_name: str
    step_type: str
    status: str
    skill_name: Optional[str] = None
    tool_name: Optional[str] = None
    input_context: Optional[Dict[str, Any]] = None
    output_result: Optional[Any] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    latency_ms: Optional[int] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AgentWorkflowExecutionResponse(BaseModel):
    id: str
    workflow_id: Optional[str] = None
    user_id: str
    workspace_id: Optional[str] = None
    trigger_source: str
    status: str
    query_or_goal: str
    plan_steps: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    final_response: Optional[str] = None
    citations: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    source_documents: Optional[List[str]] = Field(default_factory=list)
    skills_applied: Optional[List[str]] = Field(default_factory=list)
    tools_called: Optional[List[str]] = Field(default_factory=list)
    shared_state: Optional[Dict[str, Any]] = Field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    created_at: datetime
    step_logs: Optional[List[AgentWorkflowStepLogResponse]] = Field(
        default_factory=list
    )

    model_config = ConfigDict(from_attributes=True)


class AgentWorkflowExecutionListItem(BaseModel):
    id: str
    workflow_id: Optional[str] = None
    trigger_source: str
    status: str
    query_or_goal: str
    execution_time_ms: Optional[int] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    step_count: int = 0

    model_config = ConfigDict(from_attributes=True)
