from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class FailureType(str, Enum):
    BRAIN = "brain"
    AGENT = "agent"
    SKILL = "skill"
    EXTERNAL = "external"


class TrustLevel(str, Enum):
    AUTONOMOUS = "autonomous"
    NOTIFY = "notify"
    APPROVE = "approve"
    COLLABORATE = "collaborate"


class CapabilitySchema(BaseModel):
    domain: str
    task_types: list[str] = Field(default_factory=list)
    input_formats: list[str] = Field(default_factory=list)
    output_formats: list[str] = Field(default_factory=list)


class AgentSpec(BaseModel):
    name: str
    description: str
    capability_schema: CapabilitySchema
    system_prompt: str
    skills: list[str] = Field(default_factory=list)
    performance_score: float = 50.0
    task_count: int = 0
    failure_count: int = 0
    failure_attribution: dict = Field(default_factory=lambda: {
        "brain": 0, "agent": 0, "skill": 0, "external": 0
    })
    version: int = 1
    is_active: bool = True
    is_generated: bool = False
    predecessor_briefing: Optional[str] = None
    trust_levels: dict = Field(default_factory=lambda: {
        "read": "autonomous",
        "write": "approve",
        "delete": "approve",
        "send": "approve"
    })