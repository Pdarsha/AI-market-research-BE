import re
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import desc
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from server.crud.base import CRUDBase
from server.models.agent_workflow import (
    AgentStatus,
    AgentWorkflow,
    AgentWorkflowExecution,
    AgentWorkflowStepLog,
    TriggerType,
)
from server.schemas.agent_workflow import AgentWorkflowCreate, AgentWorkflowUpdate
from server.services.agent_validation_service import agent_validation_service



class CRUDAgentWorkflow(
    CRUDBase[AgentWorkflow, AgentWorkflowCreate, AgentWorkflowUpdate]
):
    """Data Access Layer for Agent Workflows."""

    def get_by_user_and_id(
        self, db: Session, *, user_id: str, agent_id: str
    ) -> Optional[AgentWorkflow]:
        """Fetch an Agent by ID, ensuring user ownership."""
        return (
            db.query(AgentWorkflow)
            .filter(
                AgentWorkflow.id == str(agent_id), AgentWorkflow.user_id == str(user_id)
            )
            .first()
        )

    def get_by_user_and_name(
        self, db: Session, *, user_id: str, name: str
    ) -> Optional[AgentWorkflow]:
        """Fetch an Agent by exact name for a user."""
        return (
            db.query(AgentWorkflow)
            .filter(
                AgentWorkflow.user_id == str(user_id),
                AgentWorkflow.name == name.strip(),
            )
            .first()
        )

    def get_multi_by_user(
        self,
        db: Session,
        *,
        user_id: str,
        skip: int = 0,
        limit: int = 100,
        workspace_id: Optional[str] = None,
        trigger_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[AgentWorkflow]:
        """List all Agents owned by user with optional workspace, trigger_type, and status filters."""
        query = db.query(AgentWorkflow).filter(AgentWorkflow.user_id == str(user_id))

        if workspace_id:
            query = query.filter(AgentWorkflow.workspace_id == str(workspace_id))
        if trigger_type:
            query = query.filter(AgentWorkflow.trigger_type == trigger_type)
        if status:
            query = query.filter(AgentWorkflow.status == status)

        return (
            query.order_by(desc(AgentWorkflow.updated_at))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_unique_agent_name(
        self,
        db: Session,
        *,
        user_id: str,
        base_name: str,
        exclude_agent_id: Optional[str] = None,
        max_attempts: int = 100,
    ) -> str:
        """Generate a unique agent name for a given user.

        Appends incrementing integer if name collision occurs.
        """
        base_name = (base_name or "").strip()
        if not base_name:
            base_name = "Untitled Agent"

        query = db.query(AgentWorkflow).filter(
            AgentWorkflow.user_id == str(user_id), AgentWorkflow.name == base_name
        )
        if exclude_agent_id:
            query = query.filter(AgentWorkflow.id != str(exclude_agent_id))
        if not query.first():
            return base_name

        match = re.match(r"^(.*?)(?:\s+(\d+))?$", base_name)
        if match and match.group(2):
            root_name = match.group(1).strip()
            counter = int(match.group(2)) + 1
        else:
            root_name = base_name
            counter = 2

        attempts = 0
        while attempts < max_attempts:
            candidate = f"{root_name} {counter}"
            if len(candidate) > 255:
                suffix = f" {counter}"
                candidate = f"{root_name[:255 - len(suffix)]}{suffix}"

            q = db.query(AgentWorkflow).filter(
                AgentWorkflow.user_id == str(user_id), AgentWorkflow.name == candidate
            )
            if exclude_agent_id:
                q = q.filter(AgentWorkflow.id != str(exclude_agent_id))
            if not q.first():
                return candidate
            counter += 1
            attempts += 1

        raise ValueError(f"Agent with name '{base_name}' already exists.")

    def create_agent(
        self,
        db: Session,
        *,
        user_id: str,
        obj_in: AgentWorkflowCreate,
        status: Optional[str] = None,
    ) -> AgentWorkflow:
        """Create and persist a new Agent Workflow."""
        agent_data = obj_in.model_dump()
        agent_status = status or agent_data.get("status") or AgentStatus.DRAFT.value

        # Normalize limit docs
        if agent_data.get("limit_documents_to_past_7_days") is not None:
            agent_data["limit_docs_past_7_days"] = bool(agent_data.get("limit_docs_past_7_days") or agent_data["limit_documents_to_past_7_days"])

        # Normalize schedule time
        norm_time, _, _, _ = agent_validation_service.normalize_time_string(
            time_str=agent_data.get("schedule_time"),
            start_time=agent_data.get("schedule_start_time"),
            end_time=agent_data.get("schedule_end_time"),
        )
        if norm_time:
            agent_data["schedule_time"] = norm_time

        # Auto-compute cron if frequency and time are provided and cron is empty
        trigger_type = agent_data.get("trigger_type")
        if trigger_type in (TriggerType.SCHEDULED.value, TriggerType.BOTH.value):
            if not agent_data.get("schedule_cron") and agent_data.get(
                "schedule_frequency"
            ):
                computed_cron = agent_validation_service.compute_cron_expression(
                    frequency=agent_data.get("schedule_frequency"),
                    day=agent_data.get("schedule_day"),
                    time_str=agent_data.get("schedule_time"),
                )
                if computed_cron:
                    agent_data["schedule_cron"] = computed_cron

        agent_data["id"] = agent_data.get("id") or str(uuid.uuid4())
        agent_data["user_id"] = str(user_id)
        agent_data["status"] = agent_status
        agent_data["created_at"] = agent_data.get("created_at") or datetime.utcnow()
        agent_data["updated_at"] = agent_data.get("updated_at") or datetime.utcnow()
        if agent_status == AgentStatus.ACTIVE.value:
            agent_data["activated_at"] = datetime.utcnow()

        db_data = {
            k: v for k, v in agent_data.items() if k in AgentWorkflow.__table__.columns
        }
        agent = AgentWorkflow(**db_data)

        try:
            db.add(agent)
            db.commit()
            db.refresh(agent)
            return agent
        except IntegrityError:
            db.rollback()
            raise ValueError(f"Agent with name '{obj_in.name}' already exists.")

    def update_agent(
        self,
        db: Session,
        *,
        db_obj: AgentWorkflow,
        obj_in: AgentWorkflowUpdate,
    ) -> AgentWorkflow:
        """Update an existing Agent Workflow, recalculating computed cron when schedule changes."""
        update_data = obj_in.model_dump(exclude_unset=True)

        if "limit_documents_to_past_7_days" in update_data and "limit_docs_past_7_days" not in update_data:
            update_data["limit_docs_past_7_days"] = bool(update_data["limit_documents_to_past_7_days"])

        # Normalize schedule time if changed
        if any(k in update_data for k in ("schedule_time", "schedule_start_time", "schedule_end_time")):
            s_time = update_data.get("schedule_time", db_obj.schedule_time)
            s_start = update_data.get("schedule_start_time")
            s_end = update_data.get("schedule_end_time")
            norm_time, _, _, _ = agent_validation_service.normalize_time_string(
                time_str=s_time,
                start_time=s_start,
                end_time=s_end,
            )
            if norm_time:
                update_data["schedule_time"] = norm_time

        # Recalculate schedule_cron if schedule fields change
        new_trigger = update_data.get("trigger_type", db_obj.trigger_type)
        is_scheduled = new_trigger in (
            TriggerType.SCHEDULED.value,
            TriggerType.BOTH.value,
        )

        if is_scheduled:
            if "schedule_cron" not in update_data and any(
                k in update_data
                for k in ("schedule_frequency", "schedule_day", "schedule_time", "schedule_start_time", "schedule_end_time")
            ):
                freq = update_data.get("schedule_frequency", db_obj.schedule_frequency)
                day = update_data.get("schedule_day", db_obj.schedule_day)
                time_str = update_data.get("schedule_time", db_obj.schedule_time)
                if freq and freq != "custom":
                    computed_cron = agent_validation_service.compute_cron_expression(
                        frequency=freq,
                        day=day,
                        time_str=time_str,
                    )
                    if computed_cron:
                        update_data["schedule_cron"] = computed_cron

        update_data["updated_at"] = datetime.utcnow()
        table_columns = AgentWorkflow.__table__.columns.keys()
        for field, value in update_data.items():
            if field in table_columns:
                setattr(db_obj, field, value)

        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def activate_agent(
        self,
        db: Session,
        *,
        db_obj: AgentWorkflow,
    ) -> AgentWorkflow:
        """Transition agent to active state with timestamp."""
        db_obj.status = AgentStatus.ACTIVE.value
        db_obj.activated_at = datetime.utcnow()
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def pause_agent(
        self,
        db: Session,
        *,
        db_obj: AgentWorkflow,
    ) -> AgentWorkflow:
        """Transition agent to paused state."""
        db_obj.status = AgentStatus.PAUSED.value
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def deactivate_agent(
        self,
        db: Session,
        *,
        db_obj: AgentWorkflow,
    ) -> AgentWorkflow:
        """Transition agent to inactive/paused state."""
        db_obj.status = AgentStatus.PAUSED.value
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def remove(self, db: Session, *, id: str) -> Optional[AgentWorkflow]:
        return super().remove(db, id=id)

    def get_execution_by_id(
        self, db: Session, *, user_id: str, execution_id: str
    ) -> Optional[AgentWorkflowExecution]:
        """Fetch an execution record by ID and user ownership."""
        return (
            db.query(AgentWorkflowExecution)
            .filter(
                AgentWorkflowExecution.id == str(execution_id),
                AgentWorkflowExecution.user_id == str(user_id),
            )
            .first()
        )

    def list_executions_by_workflow(
        self,
        db: Session,
        *,
        user_id: str,
        workflow_id: str,
        skip: int = 0,
        limit: int = 50,
    ) -> List[AgentWorkflowExecution]:
        """List execution runs for a specific agent workflow."""
        return (
            db.query(AgentWorkflowExecution)
            .filter(
                AgentWorkflowExecution.workflow_id == str(workflow_id),
                AgentWorkflowExecution.user_id == str(user_id),
            )
            .order_by(desc(AgentWorkflowExecution.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def list_executions_by_user(
        self,
        db: Session,
        *,
        user_id: str,
        skip: int = 0,
        limit: int = 50,
    ) -> List[AgentWorkflowExecution]:
        """List all execution runs belonging to a user across workflows."""
        return (
            db.query(AgentWorkflowExecution)
            .filter(AgentWorkflowExecution.user_id == str(user_id))
            .order_by(desc(AgentWorkflowExecution.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_step_logs_by_execution(
        self, db: Session, *, execution_id: str
    ) -> List[AgentWorkflowStepLog]:
        """Fetch step execution logs for a given run in chronological order."""
        return (
            db.query(AgentWorkflowStepLog)
            .filter(AgentWorkflowStepLog.execution_id == str(execution_id))
            .order_by(AgentWorkflowStepLog.step_number)
            .all()
        )


agent_workflow = CRUDAgentWorkflow(AgentWorkflow)

