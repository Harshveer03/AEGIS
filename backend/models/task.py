from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from datetime import datetime
import uuid


class TaskOutcome(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PENDING = "pending"


class FailureType(str, Enum):
    BRAIN = "brain"
    AGENT = "agent"
    SKILL = "skill"
    EXTERNAL = "external"


class TaskStatus(str, Enum):
    RECEIVED = "received"
    PLANNING = "planning"
    EXECUTING = "executing"
    SYNTHESISING = "synthesising"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_CONFIRMATION = "awaiting_confirmation"


class Task(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    raw_input: str
    status: TaskStatus = TaskStatus.RECEIVED
    execution_plan: Optional[dict] = None
    agents_used: list[str] = Field(default_factory=list)
    skills_used: list[str] = Field(default_factory=list)
    outcome: TaskOutcome = TaskOutcome.PENDING
    failure_type: Optional[FailureType] = None
    error_log: Optional[str] = None
    result: Optional[dict] = None
    duration_ms: float = 0.0
    correction_issued: bool = False
    correction_window_ms: float = 60000.0
    quality_signals: dict = Field(default_factory=dict)
    project_id: Optional[str] = None
    checkpoint_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def mark_updated(self):
        self.updated_at = datetime.utcnow()