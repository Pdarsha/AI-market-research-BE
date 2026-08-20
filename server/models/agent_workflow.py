import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from server.db.base_class import Base
from server.utils.markdown import render_rich_text


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


class AgentWorkflow(Base):
    """Database model for Agent Workflow configuration and persistence."""

    __tablename__ = "agent_workflows"

    id = Column(String, default=lambda: str(uuid.uuid4()), primary_key=True)
    user_id = Column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    workspace_id = Column(
        String, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=True
    )

    # Core metadata
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    instructions = Column(Text, nullable=True)

    # Trigger configuration
    trigger_type = Column(
        Enum(TriggerType, native_enum=False, values_callable=lambda x: [e.value for e in x]),
        default=TriggerType.AUTOMATIC,
        nullable=False,
    )

    # Schedule configuration (applicable when trigger_type is 'scheduled' or 'both')
    schedule_frequency = Column(String(50), nullable=True)  # minutes, hourly, daily, weekly, monthly, custom
    schedule_interval = Column(Integer, default=1, nullable=True)  # interval for minutes/hours
    schedule_day = Column(String(100), nullable=True)  # e.g., "monday", "1,15", etc.
    schedule_time = Column(String(100), nullable=True)  # e.g., "09:00", "09:00-17:00", "Every 1 minute"
    schedule_timezone = Column(String(50), default="UTC", nullable=True)
    schedule_cron = Column(String(100), nullable=True)  # computed/custom cron expression
    scheduled_count_limit = Column(Integer, nullable=True)  # max scheduled execution count


    # Source & document filtering
    limit_docs_past_7_days = Column(Boolean, default=False, nullable=False)
    selected_document_ids = Column(JSON, default=list, nullable=True)
    selected_sources = Column(JSON, default=list, nullable=True)
    custom_sources = Column(JSON, default=list, nullable=True)

    # AI Execution Mode
    execution_mode = Column(
        Enum(AIExecutionMode, native_enum=False, values_callable=lambda x: [e.value for e in x]),
        default=AIExecutionMode.AUTO,
        nullable=False,
    )

    # Status & Versioning
    status = Column(
        Enum(AgentStatus, native_enum=False, values_callable=lambda x: [e.value for e in x]),
        default=AgentStatus.DRAFT,
        nullable=False,
    )
    version = Column(String(50), default="1.0.0", nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    activated_at = Column(DateTime, nullable=True)
    last_run_at = Column(DateTime, nullable=True)
    next_run_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User")
    workspace = relationship("Workspace")
    executions = relationship("AgentWorkflowExecution", back_populates="workflow", cascade="all, delete-orphan")

    @property
    def description_html(self) -> str:
        return render_rich_text(self.description) if self.description else ""

    @property
    def instructions_html(self) -> str:
        return render_rich_text(self.instructions) if self.instructions else ""

    @property
    def schedule_start_time(self) -> Optional[str]:
        if not self.schedule_time:
            return None
        return self.schedule_time.split("-")[0].strip() if "-" in self.schedule_time else self.schedule_time.strip()

    @schedule_start_time.setter
    def schedule_start_time(self, val: Optional[str]):
        if val:
            end = self.schedule_end_time
            self.schedule_time = f"{val}-{end}" if end else val
        elif not self.schedule_end_time:
            self.schedule_time = None

    @property
    def schedule_end_time(self) -> Optional[str]:
        if not self.schedule_time or "-" not in self.schedule_time:
            return None
        parts = self.schedule_time.split("-")
        return parts[1].strip() if len(parts) > 1 else None

    @schedule_end_time.setter
    def schedule_end_time(self, val: Optional[str]):
        start = self.schedule_start_time or "00:00"
        if val:
            self.schedule_time = f"{start}-{val}"
        else:
            self.schedule_time = self.schedule_start_time

    @property
    def limit_documents_to_past_7_days(self) -> bool:
        return bool(self.limit_docs_past_7_days)

    @limit_documents_to_past_7_days.setter
    def limit_documents_to_past_7_days(self, val: bool):
        self.limit_docs_past_7_days = bool(val)

    @property
    def minute_interval(self) -> Optional[int]:
        return self.schedule_interval

    @minute_interval.setter
    def minute_interval(self, val: Optional[int]):
        self.schedule_interval = val

    @property
    def interval_minutes(self) -> Optional[int]:
        return self.schedule_interval

    @interval_minutes.setter
    def interval_minutes(self, val: Optional[int]):
        self.schedule_interval = val

    @property
    def prompt(self) -> Optional[str]:
        return self.instructions

    @prompt.setter
    def prompt(self, val: Optional[str]):
        self.instructions = val

    @property
    def tools(self) -> List[str]:
        return self.selected_sources or []

    @tools.setter
    def tools(self, val: Optional[List[str]]):
        self.selected_sources = val or []


    __table_args__ = (
        Index("ix_agent_workflows_user_id", "user_id"),
        Index("ix_agent_workflows_workspace_id", "workspace_id"),
        Index("ix_agent_workflows_status", "status"),
        Index("ix_agent_workflows_trigger_type", "trigger_type"),
        Index("ix_agent_workflows_next_run_at", "next_run_at"),
        UniqueConstraint("user_id", "name", name="uix_agent_workflows_user_id_name"),
    )


class ExecutionStatus(str, PyEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepExecutionStatus(str, PyEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"


class AgentWorkflowExecution(Base):
    """Database model for tracking Agent Workflow executions and state."""

    __tablename__ = "agent_workflow_executions"

    id = Column(String, default=lambda: str(uuid.uuid4()), primary_key=True)
    workflow_id = Column(
        String, ForeignKey("agent_workflows.id", ondelete="CASCADE"), nullable=True
    )
    user_id = Column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    workspace_id = Column(
        String, ForeignKey("workspaces.id", ondelete="SET NULL"), nullable=True
    )

    # Execution context & metadata
    trigger_source = Column(String(50), default="manual", nullable=False)  # manual, automatic, scheduled, api
    status = Column(
        Enum(ExecutionStatus, native_enum=False, values_callable=lambda x: [e.value for e in x]),
        default=ExecutionStatus.PENDING,
        nullable=False,
    )
    query_or_goal = Column(Text, nullable=False)
    plan_steps = Column(JSON, default=list, nullable=True)
    final_response = Column(Text, nullable=True)
    citations = Column(JSON, default=list, nullable=True)
    source_documents = Column(JSON, default=list, nullable=True)
    skills_applied = Column(JSON, default=list, nullable=True)
    tools_called = Column(JSON, default=list, nullable=True)
    shared_state = Column(JSON, default=dict, nullable=True)
    error_message = Column(Text, nullable=True)
    execution_time_ms = Column(Integer, nullable=True)

    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    workflow = relationship("AgentWorkflow", back_populates="executions")
    user = relationship("User")
    workspace = relationship("Workspace")
    step_logs = relationship(
        "AgentWorkflowStepLog",
        back_populates="execution",
        cascade="all, delete-orphan",
        order_by="AgentWorkflowStepLog.step_number",
    )

    __table_args__ = (
        Index("ix_agent_workflow_executions_user_id", "user_id"),
        Index("ix_agent_workflow_executions_workflow_id", "workflow_id"),
        Index("ix_agent_workflow_executions_workspace_id", "workspace_id"),
        Index("ix_agent_workflow_executions_status", "status"),
        Index("ix_agent_workflow_executions_created_at", "created_at"),
    )


class AgentWorkflowStepLog(Base):
    """Database model for tracking individual step execution traces in an Agent Workflow."""

    __tablename__ = "agent_workflow_step_logs"

    id = Column(String, default=lambda: str(uuid.uuid4()), primary_key=True)
    execution_id = Column(
        String, ForeignKey("agent_workflow_executions.id", ondelete="CASCADE"), nullable=False
    )
    step_number = Column(Integer, nullable=False)
    step_name = Column(String(255), nullable=False)
    step_type = Column(String(100), default="step", nullable=False)  # planning, skill_evaluation, tool_execution, synthesis
    status = Column(
        Enum(StepExecutionStatus, native_enum=False, values_callable=lambda x: [e.value for e in x]),
        default=StepExecutionStatus.RUNNING,
        nullable=False,
    )
    skill_name = Column(String(255), nullable=True)
    tool_name = Column(String(255), nullable=True)
    input_context = Column(JSON, default=dict, nullable=True)
    output_result = Column(JSON, default=dict, nullable=True)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    latency_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    execution = relationship("AgentWorkflowExecution", back_populates="step_logs")

    __table_args__ = (
        Index("ix_agent_workflow_step_logs_execution_id", "execution_id"),
        Index("ix_agent_workflow_step_logs_step_number", "step_number"),
    )

