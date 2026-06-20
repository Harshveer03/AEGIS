import time
import asyncio
from typing import Any
from config import settings
from backend.models.skill import SkillResult, OperationType
from backend.registry.skill_registry import skill_registry
import structlog

log = structlog.get_logger()


class SkillExecutor:

    def __init__(self):
        self._pending_confirmations: dict[str, dict] = {}

    # ── Main entry point ─────────────────────────────────────────
    def execute(
        self,
        skill_name: str,
        params: dict,
        task_id: str = "",
        auto_confirm: bool = False,
    ) -> SkillResult:

        spec = skill_registry.get(skill_name)
        if not spec:
            return SkillResult(
                success=False,
                error=f"Skill '{skill_name}' not found in registry",
                skill_name=skill_name,
            )

        # ── Confirmation gate ────────────────────────────────────
        if spec.requires_confirmation and not auto_confirm:
            trust = self._get_trust_level(spec.operation_type)
            if trust == "approve":
                return SkillResult(
                    success=False,
                    error="AWAITING_CONFIRMATION",
                    skill_name=skill_name,
                    data={
                        "requires_confirmation": True,
                        "skill_name": skill_name,
                        "operation_type": spec.operation_type,
                        "params": params,
                        "message": f"AEGIS wants to {spec.operation_type.value} using '{skill_name}'. Confirm? (yes/no)",
                    },
                )

        # ── Resolve callable ────────────────────────────────────
        try:
            fn = skill_registry.resolve_callable(skill_name)
        except ImportError as e:
            return SkillResult(
                success=False,
                error=str(e),
                skill_name=skill_name,
            )

        # ── Execute with retry policy ────────────────────────────
        policy = spec.retry_policy
        last_result = None
        start = time.time()

        for attempt in range(policy.max_retries + 1):
            try:
                result = fn(**params)
                result.skill_name = skill_name
                result.duration_ms = (time.time() - start) * 1000

                if result.success:
                    log.info(
                        "Skill executed",
                        skill=skill_name,
                        attempt=attempt + 1,
                        duration_ms=round(result.duration_ms, 1),
                    )
                    return result

                # Check if error is retryable
                if attempt < policy.max_retries and self._is_retryable(
                    result.error or "", policy.retryable_errors
                ):
                    log.warning(
                        "Skill failed, retrying",
                        skill=skill_name,
                        attempt=attempt + 1,
                        error=result.error,
                    )
                    time.sleep(policy.backoff_seconds * (attempt + 1))
                    last_result = result
                    continue

                last_result = result
                break

            except Exception as e:
                last_result = SkillResult(
                    success=False,
                    error=str(e),
                    skill_name=skill_name,
                    duration_ms=(time.time() - start) * 1000,
                )
                if attempt < policy.max_retries:
                    log.warning(
                        "Skill raised exception, retrying",
                        skill=skill_name,
                        attempt=attempt + 1,
                        error=str(e),
                    )
                    time.sleep(policy.backoff_seconds * (attempt + 1))
                    continue
                break

        if last_result:
            last_result.skill_name = skill_name
            last_result.duration_ms = (time.time() - start) * 1000
            log.error(
                "Skill failed after retries",
                skill=skill_name,
                attempts=attempt + 1,
                error=last_result.error,
            )
            return last_result

        return SkillResult(
            success=False,
            error="Unknown execution error",
            skill_name=skill_name,
        )

    # ── Trust level resolution ───────────────────────────────────
    def _get_trust_level(self, operation_type: OperationType) -> str:
        mapping = {
            OperationType.READ:   settings.default_trust_read,
            OperationType.WRITE:  settings.default_trust_write,
            OperationType.DELETE: settings.default_trust_delete,
            OperationType.SEND:   settings.default_trust_send,
        }
        return mapping.get(operation_type, "approve")

    # ── Retryable error check ────────────────────────────────────
    def _is_retryable(self, error: str, retryable_errors: list[str]) -> bool:
        if not retryable_errors:
            return False
        return any(e.lower() in error.lower() for e in retryable_errors)

    # ── Confirmation handling ────────────────────────────────────
    def confirm_and_execute(
        self,
        skill_name: str,
        params: dict,
        task_id: str = "",
    ) -> SkillResult:
        log.info("Confirmation received", skill=skill_name, task_id=task_id)
        return self.execute(skill_name, params, task_id, auto_confirm=True)


# ── Singleton ────────────────────────────────────────────────────
skill_executor = SkillExecutor()