"""Agent Validation Service.

Provides deterministic structural and semantic validation for Agent Workflow configurations,
including trigger types, schedule parameters, source boundaries, and status transitions.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from server import crud
from server.schemas.agent_workflow import (
    AgentStatus,
    AgentValidationResponse,
    AgentWorkflowBase,
    TriggerType,
    ValidationIssue,
)

logger = logging.getLogger(__name__)

VALID_WEEKDAYS = {
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "mon",
    "tue",
    "wed",
    "thu",
    "fri",
    "sat",
    "sun",
}

VALID_SOURCES = {
    "workspace_documents",
    "sec_edgar",
    "financial_news",
    "roic_ai",
    "companies_house",
    "web_search",
}


class AgentValidationError(Exception):
    def __init__(self, message: str, code: str = "VALIDATION_FAILED", details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class AgentValidationService:
    """Service to validate Agent Workflow configurations."""

    @staticmethod
    def parse_single_time(time_str: Optional[str]) -> Optional[Tuple[int, int]]:
        """Parse a single time string in 24h ('09:00', '9:00') or 12h ('9:00 AM', '09:00 AM', '12:30 PM') format.
        Returns (hour_24, minute) or None if invalid.
        """
        if not time_str or not isinstance(time_str, str):
            return None
        s = time_str.strip()

        # 12-hour format: e.g. "6:00 AM", "12:30 pm", "06:00AM"
        m12 = re.match(r"^(\d{1,2}):([0-5]\d)\s*([AaPp][Mm])$", s)
        if m12:
            hour = int(m12.group(1))
            minute = int(m12.group(2))
            meridian = m12.group(3).upper()
            if not (1 <= hour <= 12):
                return None
            if meridian == "AM":
                hour_24 = 0 if hour == 12 else hour
            else:  # PM
                hour_24 = 12 if hour == 12 else hour + 12
            return (hour_24, minute)

        # 24-hour format: e.g. "06:00", "6:00", "23:59"
        m24 = re.match(r"^([01]?\d|2[0-3]):([0-5]\d)$", s)
        if m24:
            hour = int(m24.group(1))
            minute = int(m24.group(2))
            return (hour, minute)

        return None

    @classmethod
    def normalize_time_string(
        cls,
        time_str: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Normalize time inputs into (normalized_schedule_time, start_time_24, end_time_24, error_message).
        Returns 24-hour format strings e.g. ("06:00-06:30", "06:00", "06:30", None).
        """
        if start_time and end_time:
            t_start = cls.parse_single_time(start_time)
            t_end = cls.parse_single_time(end_time)
            if not t_start:
                return (None, None, None, "Invalid schedule start time format. Use 'HH:MM' (24-hour).")
            if not t_end:
                return (None, None, None, "Invalid schedule end time format. Use 'HH:MM' (24-hour).")
            start_mins = t_start[0] * 60 + t_start[1]
            end_mins = t_end[0] * 60 + t_end[1]
            if end_mins <= start_mins:
                return (None, None, None, "End time must be later than start time.")
            s_start = f"{t_start[0]:02d}:{t_start[1]:02d}"
            s_end = f"{t_end[0]:02d}:{t_end[1]:02d}"
            return (f"{s_start}-{s_end}", s_start, s_end, None)

        if start_time and not end_time and (not time_str or time_str == start_time):
            t_start = cls.parse_single_time(start_time)
            if not t_start:
                return (None, None, None, "Invalid schedule start time format. Use 'HH:MM' (24-hour).")
            s_start = f"{t_start[0]:02d}:{t_start[1]:02d}"
            return (s_start, s_start, None, None)

        if time_str and isinstance(time_str, str):
            s = time_str.strip()
            if "-" in s:
                parts = s.split("-")
                if len(parts) != 2:
                    return (None, None, None, "Invalid schedule time format. Use 'HH:MM' (24-hour) or range 'HH:MM-HH:MM'.")
                t_start = cls.parse_single_time(parts[0].strip())
                t_end = cls.parse_single_time(parts[1].strip())
                if not t_start or not t_end:
                    return (None, None, None, "Invalid schedule time format. Use 'HH:MM' (24-hour) or range 'HH:MM-HH:MM'.")
                start_mins = t_start[0] * 60 + t_start[1]
                end_mins = t_end[0] * 60 + t_end[1]
                if end_mins <= start_mins:
                    return (None, None, None, "End time must be later than start time.")
                s_start = f"{t_start[0]:02d}:{t_start[1]:02d}"
                s_end = f"{t_end[0]:02d}:{t_end[1]:02d}"
                return (f"{s_start}-{s_end}", s_start, s_end, None)
            else:
                t_single = cls.parse_single_time(s)
                if not t_single:
                    return (None, None, None, "Invalid schedule time format. Use 'HH:MM' (24-hour) or range 'HH:MM-HH:MM'.")
                s_single = f"{t_single[0]:02d}:{t_single[1]:02d}"
                return (s_single, s_single, None, None)

        return (None, None, None, None)

    @classmethod
    def validate_time_format(cls, time_str: str) -> bool:
        """Validate time string (e.g., '09:00', '14:30', '09:00-17:00', '6:00 AM-6:30 AM')."""
        if not time_str or not isinstance(time_str, str):
            return False
        norm_time, _, _, err = cls.normalize_time_string(time_str=time_str)
        return err is None and norm_time is not None

    @staticmethod
    def validate_cron_expression(cron_str: str) -> bool:
        """Validate standard 5-part cron expression (minute hour day-of-month month day-of-week)."""
        if not cron_str or not isinstance(cron_str, str):
            return False
        parts = cron_str.strip().split()
        if len(parts) != 5:
            return False
        # Basic character validation for standard cron fields
        cron_part_regex = r"^(\*|[0-9,\/\-]+)$"
        return all(re.match(cron_part_regex, p) for p in parts)

    @classmethod
    def compute_cron_expression(
        cls,
        frequency: Optional[str],
        day: Optional[str],
        time_str: Optional[str],
    ) -> Optional[str]:
        """Compute standard 5-part cron expression from structured frequency, day, and time."""
        if not frequency:
            return None

        freq = frequency.lower().strip()
        minute = "0"
        hour = "9"

        if time_str:
            first_part = time_str.split("-")[0].strip()
            parsed = cls.parse_single_time(first_part)
            if parsed:
                hour = str(parsed[0])
                minute = str(parsed[1])

        if freq == "hourly":
            return f"{minute} * * * *"
        elif freq == "daily":
            return f"{minute} {hour} * * *"
        elif freq == "weekly":
            day_map = {
                "sunday": "0", "sun": "0",
                "monday": "1", "mon": "1",
                "tuesday": "2", "tue": "2",
                "wednesday": "3", "wed": "3",
                "thursday": "4", "thu": "4",
                "friday": "5", "fri": "5",
                "saturday": "6", "sat": "6",
            }
            target_days = []
            if day:
                raw_days = [d.strip().lower() for d in day.split(",")]
                for d in raw_days:
                    if d in day_map:
                        target_days.append(day_map[d])
            cron_days = ",".join(target_days) if target_days else "1"  # Default Monday
            return f"{minute} {hour} * * {cron_days}"
        elif freq == "monthly":
            dom = "1"
            if day:
                # Extract valid numbers 1-31
                raw_days = [d.strip() for d in day.split(",")]
                valid_doms = [d for d in raw_days if d.isdigit() and 1 <= int(d) <= 31]
                if valid_doms:
                    dom = ",".join(valid_doms)
            return f"{minute} {hour} {dom} * *"
        return None

    def validate_agent_config(
        self,
        agent: AgentWorkflowBase | dict,
        user_id: Optional[str] = None,
        db: Optional[Session] = None,
        check_uniqueness: bool = True,
        exclude_agent_id: Optional[str] = None,
    ) -> AgentValidationResponse:
        """Perform comprehensive validation of an Agent Workflow configuration."""
        data = agent if isinstance(agent, dict) else agent.model_dump()
        errors: List[ValidationIssue] = []
        warnings: List[ValidationIssue] = []

        # 1. Name validation
        name = (data.get("name") or "").strip()
        if not name:
            errors.append(
                ValidationIssue(
                    field="name",
                    message="Agent name is required and cannot be empty.",
                    code="EMPTY_NAME",
                )
            )
        elif len(name) > 255:
            errors.append(
                ValidationIssue(
                    field="name",
                    message="Agent name cannot exceed 255 characters.",
                    code="NAME_TOO_LONG",
                    details={"length": len(name), "max": 255},
                )
            )
        elif check_uniqueness and db and user_id:
            existing = crud.agent_workflow.get_by_user_and_name(db, user_id=user_id, name=name)
            if existing and (not exclude_agent_id or str(existing.id) != str(exclude_agent_id)):
                errors.append(
                    ValidationIssue(
                        field="name",
                        message=f"An Agent with the name '{name}' already exists. Please choose a different name.",
                        code="DUPLICATE_NAME",
                    )
                )

        # 2. Status & Instructions validation
        status = data.get("status") or AgentStatus.DRAFT.value
        instructions = (data.get("instructions") or "").strip()

        if status == AgentStatus.ACTIVE.value:
            if not instructions:
                errors.append(
                    ValidationIssue(
                        field="instructions",
                        message="Instructions are required when creating or activating an active Agent.",
                        code="EMPTY_INSTRUCTIONS",
                    )
                )
            elif len(instructions) < 10:
                warnings.append(
                    ValidationIssue(
                        field="instructions",
                        message="Instructions are very short. Provide detailed prompt steps for better reasoning.",
                        code="SHORT_INSTRUCTIONS",
                    )
                )
        elif not instructions:
            warnings.append(
                ValidationIssue(
                    field="instructions",
                    message="No instructions provided yet. Instructions will be required before activating the Agent.",
                    code="MISSING_DRAFT_INSTRUCTIONS",
                )
            )

        # 3. Trigger & Scheduling validation
        trigger_type = data.get("trigger_type") or TriggerType.AUTOMATIC.value
        is_scheduled = trigger_type in (TriggerType.SCHEDULED.value, TriggerType.BOTH.value)
        schedule_freq = (data.get("schedule_frequency") or "").lower().strip()
        schedule_time = data.get("schedule_time")
        schedule_start_time = data.get("schedule_start_time")
        schedule_end_time = data.get("schedule_end_time")
        schedule_day = data.get("schedule_day")
        schedule_cron = data.get("schedule_cron")
        scheduled_count_limit = data.get("scheduled_count_limit")

        if trigger_type not in (TriggerType.AUTOMATIC.value, TriggerType.SCHEDULED.value, TriggerType.BOTH.value):
            errors.append(
                ValidationIssue(
                    field="trigger_type",
                    message="Invalid trigger_type. Must be 'automatic', 'scheduled', or 'both'.",
                    code="INVALID_TRIGGER_TYPE",
                )
            )

        if is_scheduled:
            # If status is active, scheduling parameters are strictly required
            if status == AgentStatus.ACTIVE.value:
                if not schedule_freq:
                    errors.append(
                        ValidationIssue(
                            field="schedule_frequency",
                            message="Schedule frequency ('daily', 'weekly', 'monthly', 'hourly', 'custom') is required for scheduled agents.",
                            code="MISSING_SCHEDULE_FREQUENCY",
                        )
                    )
                elif schedule_freq not in ("daily", "weekly", "monthly", "hourly", "custom"):
                    errors.append(
                        ValidationIssue(
                            field="schedule_frequency",
                            message="Schedule frequency must be one of: 'daily', 'weekly', 'monthly', 'hourly', 'custom'.",
                            code="INVALID_SCHEDULE_FREQUENCY",
                        )
                    )
                else:
                    if schedule_freq == "custom":
                        if not schedule_cron or not self.validate_cron_expression(schedule_cron):
                            errors.append(
                                ValidationIssue(
                                    field="schedule_cron",
                                    message="A valid 5-part cron expression is required when frequency is 'custom'.",
                                    code="INVALID_CRON_EXPRESSION",
                                )
                            )
                    else:
                        has_time_input = bool(schedule_time or schedule_start_time)
                        if not has_time_input:
                            errors.append(
                                ValidationIssue(
                                    field="schedule_time",
                                    message=f"Schedule time (e.g. '09:00' or '09:00-17:00') is required for '{schedule_freq}' frequency.",
                                    code="MISSING_SCHEDULE_TIME",
                                )
                            )
                        else:
                            norm_time, s_start, s_end, err_msg = self.normalize_time_string(
                                time_str=schedule_time,
                                start_time=schedule_start_time,
                                end_time=schedule_end_time,
                            )
                            if err_msg:
                                is_range_err = "later than start time" in err_msg
                                code = "INVALID_TIME_RANGE" if is_range_err else "INVALID_TIME_FORMAT"
                                field = "schedule_end_time" if (is_range_err and schedule_end_time) else (
                                    "schedule_start_time" if (schedule_start_time and not s_start) else "schedule_time"
                                )
                                errors.append(
                                    ValidationIssue(
                                        field=field,
                                        message=err_msg,
                                        code=code,
                                    )
                                )

                        if schedule_freq == "weekly":
                            if not schedule_day:
                                errors.append(
                                    ValidationIssue(
                                        field="schedule_day",
                                        message="Schedule day (e.g., 'Monday' or 'Monday,Wednesday') is required for weekly frequency.",
                                        code="MISSING_WEEKLY_DAY",
                                    )
                                )
                            else:
                                days = [d.strip().lower() for d in schedule_day.split(",")]
                                invalid_days = [d for d in days if d not in VALID_WEEKDAYS]
                                if invalid_days:
                                    errors.append(
                                        ValidationIssue(
                                            field="schedule_day",
                                            message=f"Invalid weekdays: {', '.join(invalid_days)}.",
                                            code="INVALID_WEEKDAY",
                                        )
                                    )

                        elif schedule_freq == "monthly":
                            if not schedule_day:
                                errors.append(
                                    ValidationIssue(
                                        field="schedule_day",
                                        message="Day of month (1-31) is required for monthly frequency.",
                                        code="MISSING_MONTHLY_DAY",
                                    )
                                )
                            else:
                                days = [d.strip() for d in schedule_day.split(",")]
                                invalid_dom = [d for d in days if not d.isdigit() or not (1 <= int(d) <= 31)]
                                if invalid_dom:
                                    errors.append(
                                        ValidationIssue(
                                            field="schedule_day",
                                            message=f"Invalid day of month (must be 1-31): {', '.join(invalid_dom)}.",
                                            code="INVALID_DAY_OF_MONTH",
                                        )
                                    )
            else:
                # Draft mode warnings
                if not schedule_freq:
                    warnings.append(
                        ValidationIssue(
                            field="schedule_frequency",
                            message="Schedule frequency is not configured yet. It will be required before activating.",
                            code="INCOMPLETE_SCHEDULE_CONFIG",
                        )
                    )
                elif schedule_time or schedule_start_time or schedule_end_time:
                    norm_time, s_start, s_end, err_msg = self.normalize_time_string(
                        time_str=schedule_time,
                        start_time=schedule_start_time,
                        end_time=schedule_end_time,
                    )
                    if err_msg:
                        is_range_err = "later than start time" in err_msg
                        code = "INVALID_TIME_RANGE" if is_range_err else "INVALID_TIME_FORMAT"
                        field = "schedule_end_time" if (is_range_err and schedule_end_time) else (
                            "schedule_start_time" if (schedule_start_time and not s_start) else "schedule_time"
                        )
                        errors.append(
                            ValidationIssue(
                                field=field,
                                message=err_msg,
                                code=code,
                            )
                        )

            if scheduled_count_limit is not None and scheduled_count_limit < 1:
                errors.append(
                    ValidationIssue(
                        field="scheduled_count_limit",
                        message="Scheduled count limit must be greater than or equal to 1.",
                        code="INVALID_COUNT_LIMIT",
                    )
                )
        else:
            # Automatic only
            if schedule_freq or schedule_time or schedule_start_time or schedule_cron:
                warnings.append(
                    ValidationIssue(
                        field="trigger_type",
                        message="Trigger type is set to 'automatic' only. Provided schedule settings will be ignored until set to 'scheduled' or 'both'.",
                        code="SCHEDULE_SETTINGS_IGNORED",
                    )
                )

        # 4. Workspace & Documents verification
        workspace_id = data.get("workspace_id")
        selected_doc_ids = data.get("selected_document_ids") or []

        if db and user_id:
            if workspace_id:
                ws = crud.workspace._get_owned_workspace(db, workspace_id, user_id)
                if not ws:
                    errors.append(
                        ValidationIssue(
                            field="workspace_id",
                            message=f"Workspace '{workspace_id}' not found or access denied.",
                            code="WORKSPACE_NOT_FOUND",
                        )
                    )

            if selected_doc_ids and workspace_id:
                valid_docs = crud.document.get_multi_by_workspace(db, workspace_id=workspace_id)
                valid_ids = {str(d.id) for d in valid_docs}
                missing_doc_ids = [d_id for d_id in selected_doc_ids if str(d_id) not in valid_ids]
                if missing_doc_ids:
                    errors.append(
                        ValidationIssue(
                            field="selected_document_ids",
                            message=f"The following document IDs do not belong to the selected workspace: {missing_doc_ids}",
                            code="DOCUMENTS_NOT_FOUND",
                        )
                    )

        # 5. Selected Sources validation
        selected_sources = data.get("selected_sources") or []
        if selected_sources:
            unrecognized_sources = [s for s in selected_sources if s not in VALID_SOURCES]
            if unrecognized_sources:
                warnings.append(
                    ValidationIssue(
                        field="selected_sources",
                        message=f"Unrecognized sources detected: {unrecognized_sources}. Supported standard sources: {list(VALID_SOURCES)}",
                        code="UNRECOGNIZED_SOURCE",
                    )
                )

        return AgentValidationResponse(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )


agent_validation_service = AgentValidationService()
