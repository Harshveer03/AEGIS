"""
AEGIS Phase 2 — Memory schemas
Task data model as defined in SOW v2 section 8.3
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TaskOutcome(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    USER_CANCELLED = "user_cancelled"


class QualitySignals(BaseModel):
    """Implicit and explicit output quality signals (SOW section 5.2)."""

    # Implicit signals
    task_rerun: bool = False          # user re-ran the same task
    file_edited_after: bool = False   # user edited a file AEGIS wrote
    follow_up_suggests_incomplete: bool = False

    # Explicit signals
    user_said_wrong: bool = False     # "that was wrong" / "redo this"

    # Voice (Phase 4 — placeholder kept for schema stability)
    voice_positive: bool = False
    voice_negative: bool = False

    # Free-form notes
    notes: str = ""

    model_config = ConfigDict(extra="allow")


class TaskRecord(BaseModel):
    """
    Complete task log entry — SOW v2 section 8.3.
    Written once per task by the Brain orchestrator.
    """

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    raw_input: str                           # verbatim user input
    agents_used: list[str] = Field(default_factory=list)  # ordered
    outcome: TaskOutcome = TaskOutcome.FAILED
    duration_ms: int = 0                     # wall-clock ms, input → response
    correction_issued: bool = False          # user corrected within 60 s
    quality_signals: QualitySignals = Field(default_factory=QualitySignals)
    error_log: str | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Phase 2 extras (not in SOW table but needed for scoring)
    skills_used: list[str] = Field(default_factory=list)
    task_category: str = ""   # inferred by Brain: file | search | code | browser | mixed

    model_config = ConfigDict(use_enum_values=True)

    def to_json_dict(self) -> dict[str, Any]:
        """Serialisable dict safe for JSON storage."""
        d = self.model_dump()
        d["created_at"] = self.created_at.isoformat()
        return d

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> "TaskRecord":
        if isinstance(d.get("created_at"), str):
            d["created_at"] = datetime.fromisoformat(d["created_at"])
        return cls(**d)