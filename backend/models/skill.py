from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class OperationType(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SEND = "send"


class SkillCategory(str, Enum):
    FILE = "file"
    SEARCH = "search"
    BROWSER = "browser"
    CODE = "code"
    KNOWLEDGE = "knowledge"
    PROJECT = "project"
    CUSTOM = "custom"


class RetryPolicy(BaseModel):
    max_retries: int = 3
    backoff_seconds: float = 1.0
    retryable_errors: list[str] = Field(default_factory=lambda: [
        "TimeoutError", "ConnectionError", "httpx.TimeoutException"
    ])


class PermissionManifest(BaseModel):
    allowed_domains: list[str] = Field(default_factory=list)
    allowed_file_paths: list[str] = Field(default_factory=list)
    allowed_system_calls: list[str] = Field(default_factory=list)


class SkillSpec(BaseModel):
    name: str
    version: int = 1
    description: str
    module_path: str
    function_name: str
    parameters: dict = Field(default_factory=dict)
    output_schema: dict = Field(default_factory=dict)
    permission_manifest: PermissionManifest = Field(
        default_factory=PermissionManifest
    )
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    requires_confirmation: bool = False
    operation_type: OperationType = OperationType.READ
    category: SkillCategory = SkillCategory.CUSTOM
    is_generated: bool = False


class SkillResult(BaseModel):
    success: bool
    error: Optional[str] = None
    data: dict = Field(default_factory=dict)
    skill_name: str = ""
    duration_ms: float = 0.0