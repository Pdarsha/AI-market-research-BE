from datetime import datetime
from enum import Enum as PyEnum
import re
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TriggerType(str, PyEnum):
    AUTOMATIC = "automatic"
    SCHEDULED = "scheduled"
    BOTH = "both"


class ScheduleFrequency(str, PyEnum):
    MINUTES = "minutes"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
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
    id: Optional[str] = Field(
        None, description="Optional client-specified or persistent ID of the agent"
    )
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
        None, description="Schedule frequency: minutes, hourly, daily, weekly, monthly, custom"
    )
    schedule_interval: Optional[int] = Field(
        None, description="Interval in minutes/hours for interval-based scheduling"
    )
    minute_interval: Optional[int] = Field(
        None, description="Alias for schedule_interval in minutes"
    )
    interval_minutes: Optional[int] = Field(
        None, description="Alias for schedule_interval in minutes"
    )
    schedule_day: Optional[str] = Field(
        None, description="Day of schedule (e.g. monday, tuesday or 1, 15)"
    )
    schedule_time: Optional[str] = Field(
        None, description="Execution time (e.g. 09:00, 09:00-17:00, 'Every 1 minute')"
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
    recipient_email: Optional[str] = Field(
        None, description="Email address to receive scheduled execution results"
    )
    user_email: Optional[str] = Field(
        None, description="Alias for recipient_email"
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
    prompt: Optional[str] = Field(
        None, description="Alias for instructions (system instructions / prompt guiding the agent)"
    )
    tools: Optional[List[str]] = Field(
        default_factory=list, description="Alias for selected_sources (enabled tool/data sources)"
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
            # 1. Sync duplicate prompt / instructions fields
            if data.get("prompt") and not data.get("instructions"):
                data["instructions"] = data["prompt"]
            elif data.get("instructions") and not data.get("prompt"):
                data["prompt"] = data["instructions"]

            # 2. Sync duplicate tools / selected_sources fields
            if data.get("tools") and not data.get("selected_sources"):
                data["selected_sources"] = data["tools"]
            elif data.get("selected_sources") and not data.get("tools"):
                data["tools"] = data["selected_sources"]

            # 3. Sync duplicate interval fields & extract from schedule_time if applicable
            interval = (
                data.get("schedule_interval")
                or data.get("minute_interval")
                or data.get("interval_minutes")
            )
            if not interval and data.get("schedule_time"):
                m = re.search(r"every\s+(\d+)\s+min", str(data.get("schedule_time")), re.IGNORECASE)
                if m:
                    interval = int(m.group(1))
            if interval is not None:
                try:
                    int_val = int(interval)
                    data["schedule_interval"] = int_val
                    data["minute_interval"] = int_val
                    data["interval_minutes"] = int_val
                except (ValueError, TypeError):
                    pass

            # 4. Sync duplicate limit fields
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

            # 5. Normalize times
            freq = str(data.get("schedule_frequency") or "").lower().strip()
            if freq not in ("minutes", "minute", "min", "mins"):
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
            elif data.get("schedule_time") is None and interval:
                data["schedule_time"] = f"Every {interval} minute{'s' if interval > 1 else ''}"

            # 6. Sync recipient_email / user_email
            if data.get("user_email") and not data.get("recipient_email"):
                data["recipient_email"] = data["user_email"]
            elif data.get("recipient_email") and not data.get("user_email"):
                data["user_email"] = data["recipient_email"]
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
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    instructions: Optional[str] = None
    prompt: Optional[str] = None
    tools: Optional[List[str]] = None
    trigger_type: Optional[TriggerType] = None
    schedule_frequency: Optional[str] = None
    schedule_interval: Optional[int] = None
    minute_interval: Optional[int] = None
    interval_minutes: Optional[int] = None
    schedule_day: Optional[str] = None
    schedule_time: Optional[str] = None
    schedule_start_time: Optional[str] = None
    schedule_end_time: Optional[str] = None
    schedule_timezone: Optional[str] = None
    schedule_cron: Optional[str] = None
    scheduled_count_limit: Optional[int] = Field(None, ge=1)
    recipient_email: Optional[str] = None
    user_email: Optional[str] = None
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
            # 0. Clean name if provided as empty string
            if "name" in data:
                if isinstance(data["name"], str):
                    clean_name = data["name"].strip()
                    data["name"] = clean_name if clean_name else None
                elif data["name"] is None:
                    data["name"] = None

            # 1. Sync prompt / instructions
            if data.get("prompt") and not data.get("instructions"):
                data["instructions"] = data["prompt"]
            elif data.get("instructions") and not data.get("prompt"):
                data["prompt"] = data["instructions"]

            # 2. Sync tools / selected_sources
            if data.get("tools") and not data.get("selected_sources"):
                data["selected_sources"] = data["tools"]
            elif data.get("selected_sources") and not data.get("tools"):
                data["tools"] = data["selected_sources"]

            # 3. Sync interval
            interval = (
                data.get("schedule_interval")
                or data.get("minute_interval")
                or data.get("interval_minutes")
            )
            if not interval and data.get("schedule_time"):
                m = re.search(r"every\s+(\d+)\s+min", str(data.get("schedule_time")), re.IGNORECASE)
                if m:
                    interval = int(m.group(1))
            if interval is not None:
                try:
                    int_val = int(interval)
                    data["schedule_interval"] = int_val
                    data["minute_interval"] = int_val
                    data["interval_minutes"] = int_val
                except (ValueError, TypeError):
                    pass

            # 4. Sync limits
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

            # 5. Times
            freq = str(data.get("schedule_frequency") or "").lower().strip()
            if freq not in ("minutes", "minute", "min", "mins"):
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

            # 6. Sync recipient_email / user_email
            if data.get("user_email") and not data.get("recipient_email"):
                data["recipient_email"] = data["user_email"]
            elif data.get("recipient_email") and not data.get("user_email"):
                data["user_email"] = data["recipient_email"]
        return data


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
    schedule_interval: Optional[int] = None
    schedule_day: Optional[str] = None
    schedule_time: Optional[str] = None
    schedule_start_time: Optional[str] = None
    schedule_end_time: Optional[str] = None
    schedule_timezone: Optional[str] = None
    recipient_email: Optional[str] = None
    user_email: Optional[str] = None

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
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    workflow_name: Optional[str] = None
    user_id: str
    workspace_id: Optional[str] = None
    trigger_source: str
    status: str
    query_or_goal: str
    plan_steps: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    final_response: Optional[str] = None
    response: Optional[str] = None
    content: Optional[str] = None
    citations: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    sources: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    source_documents: Optional[List[str]] = Field(default_factory=list)
    source_attribution: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = Field(
        default_factory=dict
    )
    follow_up_questions: Optional[List[str]] = Field(default_factory=list)
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

    @model_validator(mode="after")
    def sync_response_fields(self) -> "AgentWorkflowExecutionResponse":
        if not self.final_response and self.response:
            self.final_response = self.response
        if not self.final_response and self.content:
            self.final_response = self.content
        if not self.response and self.final_response:
            self.response = self.final_response
        if not self.content and self.final_response:
            self.content = self.final_response
        if not self.sources and self.citations:
            self.sources = self.citations
        if not self.citations and self.sources:
            self.citations = self.sources
        if isinstance(self.shared_state, dict):
            if not self.follow_up_questions and self.shared_state.get("follow_up_questions"):
                self.follow_up_questions = self.shared_state.get("follow_up_questions") or []
            if not self.source_attribution and self.shared_state.get("source_attribution"):
                self.source_attribution = self.shared_state.get("source_attribution") or {}
            if not self.citations and self.shared_state.get("citations"):
                self.citations = self.shared_state.get("citations") or []
                self.sources = self.citations
            if not self.source_documents and self.shared_state.get("source_documents"):
                self.source_documents = self.shared_state.get("source_documents") or []
        return self


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
