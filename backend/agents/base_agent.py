import time
from abc import ABC, abstractmethod
from typing import Optional
from config import settings
from backend.models.agent import AgentSpec
from backend.models.task import Task, TaskOutcome, TaskStatus, FailureType
from backend.models.skill import SkillResult
from backend.skills.executor import skill_executor
from backend.registry.skill_registry import skill_registry
import structlog

log = structlog.get_logger()


class BaseAgent(ABC):

    def __init__(self, spec: AgentSpec):
        self.spec = spec
        self.name = spec.name

    # ── Main execute entry point ─────────────────────────────────
    def execute(self, task: Task, goal: str, context: dict = {}) -> dict:
        log.info("Agent executing", agent=self.name, task_id=task.task_id, goal=goal[:80])
        start = time.time()

        task.status = TaskStatus.EXECUTING
        if self.name not in task.agents_used:
            task.agents_used.append(self.name)

        try:
            result = self.run(task, goal, context)

            duration = (time.time() - start) * 1000
            log.info(
                "Agent completed",
                agent=self.name,
                task_id=task.task_id,
                duration_ms=round(duration, 1),
                success=result.get("success", False),
            )
            return result

        except Exception as e:
            duration = (time.time() - start) * 1000
            log.error("Agent raised exception", agent=self.name, error=str(e))
            task.failure_type = FailureType.AGENT
            return {
                "success": False,
                "error": str(e),
                "agent": self.name,
                "failure_type": FailureType.AGENT,
            }

    # ── Abstract — each agent implements this ────────────────────
    @abstractmethod
    def run(self, task: Task, goal: str, context: dict) -> dict:
        pass

    # ── Skill execution helpers ──────────────────────────────────
    def use_skill(
        self,
        skill_name: str,
        params: dict,
        task: Task,
        auto_confirm: bool = False,
    ) -> SkillResult:

        # Check skill is in this agent's list
        if skill_name not in self.spec.skills:
            log.error(
                "Agent attempted skill not in its list",
                agent=self.name,
                skill=skill_name,
            )
            return SkillResult(
                success=False,
                error=f"Skill '{skill_name}' is not in {self.name}'s skill list",
                skill_name=skill_name,
            )

        # Track skill usage on task
        if skill_name not in task.skills_used:
            task.skills_used.append(skill_name)

        result = skill_executor.execute(
            skill_name, params, task_id=task.task_id, auto_confirm=auto_confirm
        )

        # Tag skill failures on the task
        if not result.success and result.error != "AWAITING_CONFIRMATION":
            task.failure_type = FailureType.SKILL

        return result

    def confirm_skill(
        self,
        skill_name: str,
        params: dict,
        task: Task,
    ) -> SkillResult:
        if skill_name not in self.spec.skills:
            return SkillResult(
                success=False,
                error=f"Skill '{skill_name}' is not in {self.name}'s skill list",
                skill_name=skill_name,
            )
        if skill_name not in task.skills_used:
            task.skills_used.append(skill_name)
        return skill_executor.confirm_and_execute(skill_name, params, task.task_id)

    # ── Skill list management ────────────────────────────────────
    def has_skill(self, skill_name: str) -> bool:
        return skill_name in self.spec.skills

    def get_skills(self) -> list[str]:
        return list(self.spec.skills)

    def get_skill_descriptions(self) -> list[dict]:
        descriptions = []
        for name in self.spec.skills:
            spec = skill_registry.get(name)
            if spec:
                descriptions.append({
                    "name": name,
                    "description": spec.description,
                    "requires_confirmation": spec.requires_confirmation,
                    "operation_type": spec.operation_type,
                })
        return descriptions

    # ── Result builders ──────────────────────────────────────────
    def success(self, data: dict = {}, message: str = "") -> dict:
        return {
            "success": True,
            "agent": self.name,
            "data": data,
            "message": message,
            "failure_type": None,
        }

    def failure(self, error: str, failure_type: FailureType = FailureType.AGENT) -> dict:
        return {
            "success": False,
            "agent": self.name,
            "error": error,
            "failure_type": failure_type,
        }

    def awaiting_confirmation(self, skill_name: str, params: dict, message: str) -> dict:
        return {
            "success": False,
            "agent": self.name,
            "awaiting_confirmation": True,
            "skill_name": skill_name,
            "params": params,
            "message": message,
        }

    # ── Repr ─────────────────────────────────────────────────────
    def __repr__(self):
        return f"<Agent: {self.name} score={self.spec.performance_score} skills={len(self.spec.skills)}>"