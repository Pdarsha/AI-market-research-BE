"""Agent Scheduler Service.

Implements the Scheduled Trigger mechanism for Agent Workflows:
  1. Identify active Agents configured for Scheduled Trigger (or Both).
  2. Parse schedule configuration (frequency, day, time, timezone, custom cron).
  3. Prevent duplicate executions via in-memory concurrency locks and last_run_at throttling.
  4. Respect scheduled execution limits (scheduled_count_limit) and 7-day document limits.
  5. When schedule fires, invoke the SAME Agent Execution Engine (with trigger_source="scheduled").
  6. Provide modular background polling loop, manual tick execution, and status inspection.
"""

from __future__ import annotations

import asyncio
import calendar
import logging
import re
import zoneinfo
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from server import crud, schemas
from server.db.session import SessionLocal

from server.models.agent_workflow import (
    AgentStatus,
    AgentWorkflow,
    AgentWorkflowExecution,
    TriggerType,
)
from server.schemas.agent_workflow import AgentWorkflowExecutionResponse
from server.services.agent_workflow_engine import (
    AgentWorkflowEngine,
    get_agent_workflow_engine,
)
from server.services.audit_logger import AuditLogger
from server.models.agent_workflow import ExecutionStatus
from server.services.agent_title_service import (
    generate_agent_chat_title,
)
from server.models.genai import ChatMessage

logger = logging.getLogger(__name__)

# Day name mapping for weekly schedules
DAY_NAME_TO_INT = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}


class AgentSchedulerService:
    """Modular scheduler engine to evaluate and execute scheduled Agent Workflows."""

    def __init__(
        self,
        workflow_engine: Optional[AgentWorkflowEngine] = None,
    ):
        self.workflow_engine = workflow_engine or get_agent_workflow_engine()
        self._running_agent_ids: Set[str] = set()
        self._lock = asyncio.Lock()
        self._background_task: Optional[asyncio.Task] = None
        self._is_running = False

    # --- 1. Candidate Discovery ---

    def get_active_scheduled_agents(
        self, db: Session, *, user_id: Optional[str] = None
    ) -> List[AgentWorkflow]:
        """Query all active agents configured with scheduled or both trigger types."""
        query = crud.agent_workflow.get_active_scheduled_agents(db=db)

        if user_id:
            query = query.filter(AgentWorkflow.user_id == str(user_id))
        return query.all()

    # --- 2. Schedule & Cron Evaluation ---

    @staticmethod
    def get_timezone(tz_str: Optional[str]) -> timezone | zoneinfo.ZoneInfo:
        """Resolve timezone string to ZoneInfo or fallback to UTC."""
        if not tz_str:
            return timezone.utc
        try:
            return zoneinfo.ZoneInfo(tz_str)
        except Exception:
            return timezone.utc

    @staticmethod
    def parse_time_field(time_str: Optional[str]) -> Tuple[int, int]:
        """Extract (hour, minute) from time string e.g. '09:30', '14:00-15:00', '6:00 AM-6:30 AM'."""
        if not time_str:
            return (0, 0)
        from server.services.agent_validation_service import agent_validation_service

        # Take first part if range e.g. '09:00-10:00'
        first_part = time_str.split("-")[0].strip()
        parsed = agent_validation_service.parse_single_time(first_part)
        if parsed:
            return (max(0, min(23, parsed[0])), max(0, min(59, parsed[1])))
        return (0, 0)

    @classmethod
    def matches_cron_field(cls, field_expr: str, current_val: int) -> bool:
        """Evaluate a single cron field expression (*, */N, A-B, A,B,C)."""
        field_expr = field_expr.strip()
        if field_expr == "*":
            return True

        if field_expr.startswith("*/"):
            try:
                step = int(field_expr[2:])
                return step > 0 and (current_val % step == 0)
            except ValueError:
                return False

        if "," in field_expr:
            sub_fields = field_expr.split(",")
            return any(cls.matches_cron_field(sub, current_val) for sub in sub_fields)

        if "-" in field_expr:
            try:
                start_str, end_str = field_expr.split("-")
                start, end = int(start_str), int(end_str)
                return start <= current_val <= end
            except ValueError:
                return False

        try:
            return int(field_expr) == current_val
        except ValueError:
            return False

    @classmethod
    def matches_cron(cls, cron_expr: str, dt: datetime) -> bool:
        """Evaluate a 5-part cron expression (min, hour, dom, month, dow)."""
        parts = cron_expr.strip().split()
        if len(parts) != 5:
            return False

        min_expr, hour_expr, dom_expr, mon_expr, dow_expr = parts

        # Match minute (0-59)
        if not cls.matches_cron_field(min_expr, dt.minute):
            return False
        # Match hour (0-23)
        if not cls.matches_cron_field(hour_expr, dt.hour):
            return False
        # Match day of month (1-31)
        if not cls.matches_cron_field(dom_expr, dt.day):
            return False
        # Match month (1-12)
        if not cls.matches_cron_field(mon_expr, dt.month):
            return False
        # Match day of week (0-6, Monday=0 or Sunday=0)
        # Python weekday: Mon=0, Sun=6. Standard cron: Sun=0 or 7.
        python_weekday = dt.weekday()
        # Convert python_weekday (0-6 where Mon=0) to cron weekday (0=Sun, 1=Mon, ..., 6=Sat, 7=Sun)
        cron_weekday = (python_weekday + 1) % 7
        dow_matches = (
            cls.matches_cron_field(dow_expr, python_weekday)
            or cls.matches_cron_field(dow_expr, cron_weekday)
            or (python_weekday == 6 and cls.matches_cron_field(dow_expr, 7))
        )
        if not dow_matches:
            return False

        return True

    def should_run_now(
        self,
        agent: AgentWorkflow,
        current_utc: Optional[datetime] = None,
        db: Optional[Session] = None,
    ) -> bool:
        """Check if an active agent is scheduled to run at the given UTC time."""
        agent_status = getattr(agent.status, "value", agent.status)
        if agent_status != AgentStatus.ACTIVE.value:
            return False

        trigger_type = getattr(agent.trigger_type, "value", agent.trigger_type)
        if trigger_type not in (TriggerType.SCHEDULED.value, TriggerType.BOTH.value):
            return False

        now_utc = current_utc or datetime.now(timezone.utc)
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)

        # 1. Throttling: Check if already ran within the last 55 seconds
        if agent.last_run_at:
            last_run = agent.last_run_at
            if last_run.tzinfo is None:
                last_run = last_run.replace(tzinfo=timezone.utc)
            if (now_utc - last_run).total_seconds() < 55:
                return False

        # 2. Check scheduled count limit
        if agent.scheduled_count_limit is not None and db is not None:
            exec_count = (
                db.query(func.count(AgentWorkflowExecution.id))
                .filter(
                    AgentWorkflowExecution.workflow_id == agent.id,
                    AgentWorkflowExecution.trigger_source == "scheduled",
                )
                .scalar()
                or 0
            )
            if exec_count >= agent.scheduled_count_limit:
                logger.info(
                    "[AGENT_SCHEDULER] Agent '%s' (id=%s) reached scheduled count limit (%d/%d)",
                    agent.name,
                    agent.id,
                    exec_count,
                    agent.scheduled_count_limit,
                )
                return False

        # Convert now_utc to agent's local timezone
        agent_tz = self.get_timezone(agent.schedule_timezone)
        local_dt = now_utc.astimezone(agent_tz)

        # 3. Evaluate Custom Cron
        if agent.schedule_cron and agent.schedule_cron.strip():
            return self.matches_cron(agent.schedule_cron, local_dt)

        # 4. Evaluate Frequency
        freq = (agent.schedule_frequency or "").strip().lower()
        sched_hour, sched_min = self.parse_time_field(agent.schedule_time)

        if freq == "daily":
            # Match hour and minute
            return local_dt.hour == sched_hour and local_dt.minute == sched_min

        elif freq == "weekly":
            # Match weekday(s) + hour + minute
            day_str = (agent.schedule_day or "monday").strip().lower()
            raw_days = [d.strip().lower() for d in day_str.split(",") if d.strip()]
            target_weekdays = [
                DAY_NAME_TO_INT[d] for d in raw_days if d in DAY_NAME_TO_INT
            ]
            if not target_weekdays:
                target_weekdays = [0]
            return (
                local_dt.weekday() in target_weekdays
                and local_dt.hour == sched_hour
                and local_dt.minute == sched_min
            )

        elif freq == "monthly":
            # Match day(s) of month + hour + minute
            day_str = (agent.schedule_day or "1").strip().lower()
            raw_days = [d.strip().lower() for d in day_str.split(",") if d.strip()]
            _, last_day = calendar.monthrange(local_dt.year, local_dt.month)
            target_days = []
            for d in raw_days:
                if d == "last":
                    target_days.append(last_day)
                elif d.isdigit() and 1 <= int(d) <= 31:
                    target_days.append(int(d))
            if not target_days:
                target_days = [1]

            return (
                local_dt.day in target_days
                and local_dt.hour == sched_hour
                and local_dt.minute == sched_min
            )

        elif freq == "hourly":
            # Match minute of every hour
            return local_dt.minute == sched_min

        elif freq in ("minutes", "minute", "min", "mins"):
            interval = getattr(agent, "schedule_interval", None) or 1
            if not interval or interval < 1:
                interval = 1
            return (local_dt.minute % interval) == 0

        return False

    def calculate_next_run_time(
        self, agent: AgentWorkflow, from_utc: Optional[datetime] = None
    ) -> Optional[datetime]:
        """Estimate the next UTC run time for an agent for user visibility."""
        agent_status = getattr(agent.status, "value", agent.status)
        if agent_status != AgentStatus.ACTIVE.value:
            return None

        trigger_type = getattr(agent.trigger_type, "value", agent.trigger_type)
        if trigger_type not in (TriggerType.SCHEDULED.value, TriggerType.BOTH.value):
            return None

        start_utc = from_utc or datetime.now(timezone.utc)
        if start_utc.tzinfo is None:
            start_utc = start_utc.replace(tzinfo=timezone.utc)

        agent_tz = self.get_timezone(agent.schedule_timezone)
        freq = (agent.schedule_frequency or "").strip().lower()
        sched_hour, sched_min = self.parse_time_field(agent.schedule_time)

        local_now = start_utc.astimezone(agent_tz)

        # 1. Custom Cron or explicit cron expression
        if agent.schedule_cron and agent.schedule_cron.strip():
            candidate = (local_now + timedelta(minutes=1)).replace(
                second=0, microsecond=0
            )
            max_minutes = 60 * 24 * 31  # up to 31 days
            for _ in range(max_minutes):
                if self.matches_cron(agent.schedule_cron, candidate):
                    return candidate.astimezone(timezone.utc)
                candidate += timedelta(minutes=1)
            return None

        if freq in ("minutes", "minute", "min", "mins"):
            interval = getattr(agent, "schedule_interval", None) or 1
            if not interval or interval < 1:
                interval = 1
            candidate = (local_now + timedelta(minutes=1)).replace(
                second=0, microsecond=0
            )
            while candidate.minute % interval != 0:
                candidate += timedelta(minutes=1)
            return candidate.astimezone(timezone.utc)

        elif freq == "daily":
            candidate = local_now.replace(
                hour=sched_hour, minute=sched_min, second=0, microsecond=0
            )
            if candidate <= local_now:
                candidate += timedelta(days=1)
            return candidate.astimezone(timezone.utc)

        elif freq == "weekly":
            day_str = (agent.schedule_day or "monday").strip().lower()
            raw_days = [d.strip().lower() for d in day_str.split(",") if d.strip()]
            target_dows = [DAY_NAME_TO_INT[d] for d in raw_days if d in DAY_NAME_TO_INT]
            if not target_dows:
                target_dows = [0]

            candidates = []
            for target_dow in target_dows:
                days_ahead = (target_dow - local_now.weekday()) % 7
                candidate = local_now.replace(
                    hour=sched_hour, minute=sched_min, second=0, microsecond=0
                ) + timedelta(days=days_ahead)
                if candidate <= local_now:
                    candidate += timedelta(days=7)
                candidates.append(candidate)
            return min(candidates).astimezone(timezone.utc)

        elif freq == "monthly":
            day_str = (agent.schedule_day or "1").strip().lower()
            raw_days = [d.strip().lower() for d in day_str.split(",") if d.strip()]
            if not raw_days:
                raw_days = ["1"]

            candidates = []
            for year_offset, month_val in [
                (0, local_now.month),
                (1 if local_now.month == 12 else 0, (local_now.month % 12) + 1),
            ]:
                target_year = local_now.year + year_offset
                _, last_day = calendar.monthrange(target_year, month_val)
                for d in raw_days:
                    t_day = (
                        last_day
                        if d == "last"
                        else (int(d) if d.isdigit() and 1 <= int(d) <= 31 else 1)
                    )
                    if t_day > last_day:
                        t_day = last_day
                    try:
                        cand = local_now.replace(
                            year=target_year,
                            month=month_val,
                            day=t_day,
                            hour=sched_hour,
                            minute=sched_min,
                            second=0,
                            microsecond=0,
                        )
                        if cand > local_now:
                            candidates.append(cand)
                    except (ValueError, OverflowError):
                        continue

            if candidates:
                return min(candidates).astimezone(timezone.utc)

        elif freq == "hourly":
            candidate = local_now.replace(minute=sched_min, second=0, microsecond=0)
            if candidate <= local_now:
                candidate += timedelta(hours=1)
            return candidate.astimezone(timezone.utc)

        return None

    # --- 3. Execution Trigger ---

    async def execute_scheduled_agent(
        self,
        agent_id: str,
        db: Optional[Session] = None,
        *,
        triggered_at: Optional[datetime] = None,
        status_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> Optional[AgentWorkflowExecutionResponse]:
        """Execute a scheduled Agent Workflow through the common execution engine.

        Guarantees:
          - Concurrency-safe lock prevents duplicate simultaneous execution for same agent.
          - Manages independent DB session lifecycle when called asynchronously.
          - Increments run counter & auto-pauses when count limit is reached.
          - Passes trigger_source="scheduled".
          - Creates daily agent chat session & records message history.
          - Captures and records errors gracefully without breaking the scheduler.
        """
        # Concurrency check under lock (fast critical section)
        async with self._lock:
            if agent_id in self._running_agent_ids:
                logger.warning(
                    "[AGENT_SCHEDULER] Agent '%s' is already currently executing. Skipping duplicate trigger.",
                    agent_id,
                )
                return None
            self._running_agent_ids.add(agent_id)

        try:
            from server.services.agent_execution_service import agent_execution_service

            logger.info(
                "[AGENT_SCHEDULER] [TRIGGER] Fired scheduled trigger for agent id=%s",
                agent_id,
            )

            execution_response = await agent_execution_service.execute(
                agent_id=agent_id,
                trigger_source="scheduled",
                triggered_at=triggered_at,
                status_callback=status_callback,
                db=db,
            )
            return execution_response

        except Exception as e:
            logger.error(
                f"[AGENT_SCHEDULER] [ERROR] Failed scheduled execution for agent {agent_id}: {str(e)}",
                exc_info=True,
            )
            return None

        finally:
            async with self._lock:
                self._running_agent_ids.discard(agent_id)


    # --- 4. Scheduler Tick & Background Loop ---

    async def tick(
        self,
        db_session_factory=SessionLocal,
        current_utc: Optional[datetime] = None,
        db: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """Evaluate all active scheduled agents and fire jobs that are due."""
        now = current_utc or datetime.now(timezone.utc)
        owns_db = False
        if db is None:
            db = db_session_factory()
            owns_db = True
        triggered_agent_ids = []

        try:
            candidates = self.get_active_scheduled_agents(db)
            for agent in candidates:
                if self.should_run_now(agent, current_utc=now, db=db):
                    logger.info(
                        "[AGENT_SCHEDULER] Agent '%s' is due for execution at %s",
                        agent.name,
                        now.isoformat(),
                    )
                    triggered_agent_ids.append(agent.id)
                    # Trigger asynchronously with isolated DB session
                    asyncio.create_task(
                        self.execute_scheduled_agent(
                            agent.id, db=None, triggered_at=now
                        )
                    )

            return {
                "timestamp": now.isoformat(),
                "checked_count": len(candidates),
                "triggered_count": len(triggered_agent_ids),
                "triggered_agent_ids": triggered_agent_ids,
            }
        finally:
            if owns_db:
                db.close()

    async def _scheduler_loop(self, poll_interval_seconds: int = 30):
        """Continuous background loop evaluating schedules every poll_interval_seconds."""
        logger.info(
            f"[AGENT_SCHEDULER] Background scheduler started (poll_interval={poll_interval_seconds}s)"
        )
        self._is_running = True
        while self._is_running:
            try:
                await self.tick()
            except Exception as e:
                logger.error(
                    f"[AGENT_SCHEDULER] Error in scheduler loop tick: {e}",
                    exc_info=True,
                )
            await asyncio.sleep(poll_interval_seconds)

    def start(self, poll_interval_seconds: int = 30):
        """Start the background scheduler task."""
        if self._background_task and not self._background_task.done():
            logger.info("[AGENT_SCHEDULER] Scheduler is already running.")
            return

        self._is_running = True
        try:
            loop = asyncio.get_running_loop()
            self._background_task = loop.create_task(
                self._scheduler_loop(poll_interval_seconds=poll_interval_seconds)
            )
        except RuntimeError:
            # No running event loop yet (e.g. running in thread before asyncio starts)
            pass

    def stop(self):
        """Stop the background scheduler cleanly."""
        self._is_running = False
        if self._background_task:
            self._background_task.cancel()
            self._background_task = None
            logger.info("[AGENT_SCHEDULER] Background scheduler stopped.")

    def recover_stale_executions(self, db: Session) -> int:
        """Mark orphaned/stuck running or pending executions as failed upon application restart."""
        try:

            stale_executions = crud.agent_workflow.stale_executions(db)
            count = 0
            for ex in stale_executions:
                ex.status = ExecutionStatus.FAILED.value
                ex.error_message = (
                    "Execution aborted: Server restarted while task was in progress."
                )
                ex.completed_at = datetime.utcnow()
                db.add(ex)
                count += 1
            if count > 0:
                db.commit()
                logger.info(
                    f"[AGENT_SCHEDULER] Recovered and marked {count} stale execution(s) as failed on startup."
                )
            return count
        except Exception as e:
            logger.error(
                f"[AGENT_SCHEDULER] Error during stale executions recovery: {e}",
                exc_info=True,
            )
            db.rollback()
            return 0

    @property
    def is_running(self) -> bool:
        return self._is_running


# Singleton
agent_scheduler_service = AgentSchedulerService()


def get_agent_scheduler_service() -> AgentSchedulerService:
    return agent_scheduler_service
