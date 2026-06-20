from backend.models.agent import AgentSpec, CapabilitySchema
from backend.models.task import Task, FailureType
from backend.agents.base_agent import BaseAgent


class CodeAgent(BaseAgent):

    def run(self, task: Task, goal: str, context: dict) -> dict:
        goal_lower = goal.lower()
        code = context.get("code", "")

        # ── Syntax validation ────────────────────────────────────
        if any(w in goal_lower for w in ["validate", "check syntax", "syntax check", "lint"]):
            if not code:
                return self.failure("No code provided for syntax validation")
            result = self.use_skill("validate_syntax", {"code": code}, task)
            if result.success:
                return self.success(result.data, "Syntax validation complete")
            return self.failure(result.error or "Validation failed", FailureType.SKILL)

        # ── Run code ─────────────────────────────────────────────
        if any(w in goal_lower for w in ["run", "execute", "eval"]):
            if not code:
                return self.failure("No code provided for execution")
            # Always validate before running
            validate = self.use_skill("validate_syntax", {"code": code}, task)
            if not validate.success:
                return self.failure(f"Syntax error before execution: {validate.error}")

            result = self.use_skill("run_python", {"code": code, "timeout": 30}, task)
            if result.error == "AWAITING_CONFIRMATION":
                return self.awaiting_confirmation(
                    "run_python",
                    {"code": code, "timeout": 30},
                    "Execute this Python code?"
                )
            if result.success:
                return self.success(result.data, "Code executed successfully")
            return self.failure(result.error or "Execution failed", FailureType.SKILL)

        # ── Default: validate then run ───────────────────────────
        if code:
            validate = self.use_skill("validate_syntax", {"code": code}, task)
            if not validate.success:
                return self.failure(f"Syntax error: {validate.error}")
            result = self.use_skill("run_python", {"code": code, "timeout": 30}, task)
            if result.error == "AWAITING_CONFIRMATION":
                return self.awaiting_confirmation(
                    "run_python",
                    {"code": code, "timeout": 30},
                    "Execute this Python code?"
                )
            if result.success:
                return self.success(result.data, "Code executed")
            return self.failure(result.error or "Execution failed", FailureType.SKILL)

        return self.failure("No code or clear instruction provided to CodeAgent")


def create_code_agent() -> CodeAgent:
    spec = AgentSpec(
        name="code_agent",
        description="Writes, validates, and executes Python code. Validates syntax before running. Handles code generation and script execution tasks.",
        capability_schema=CapabilitySchema(
            domain="code",
            task_types=["run_python", "validate_syntax", "execute_code", "code_generation"],
            input_formats=["code", "text"],
            output_formats=["text", "stdout", "structured"],
        ),
        system_prompt=(
            "You are the Code Agent, a specialist in Python code execution and validation. "
            "You always validate syntax before executing. "
            "You never run code that could cause harm to the system. "
            "You always require confirmation before executing code."
        ),
        skills=["run_python", "validate_syntax", "run_python_with_input"],
    )
    return CodeAgent(spec)