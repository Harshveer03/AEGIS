import json
import importlib
import sys
from pathlib import Path
from config import settings
from backend.models.skill import SkillSpec, SkillCategory, OperationType, RetryPolicy, PermissionManifest
from backend.registry.skill_registry import skill_registry
from backend.factory.validator import validate_skill_code, validate_skill_spec
import structlog

log = structlog.get_logger()

SKILL_CREATION_PROMPT = """You are an expert Python developer building skills for an AI system.

Write a single Python function for this capability: {capability}

Requirements:
- Function name must be: {function_name}
- Must accept only simple parameters (str, int, float, bool, list, dict)
- Must return a SkillResult object
- Import SkillResult like this: from backend.models.skill import SkillResult
- No os.system(), no subprocess, no eval(), no exec()
- Handle all exceptions and return SkillResult(success=False, error=str(e)) on failure
- Return SkillResult(success=True, data={{...}}) on success

Write ONLY the Python function — no explanation, no markdown, no imports except SkillResult."""


class SkillFactory:

    def __init__(self, llm_call):
        self.llm_call = llm_call
        self._output_path = Path(settings.generated_skills_path)
        self._output_path.mkdir(parents=True, exist_ok=True)

    def create(self, capability: str, function_name: str = "") -> tuple[bool, str, SkillSpec | None]:
        log.info("Skill Factory: creating skill", capability=capability)

        # ── Step 1: Check similarity ─────────────────────────────
        similar = skill_registry.find_similar(capability)
        if similar:
            log.info(
                "Similar skill exists",
                existing=similar[0].name,
                capability=capability,
            )
            return False, f"Similar skill already exists: '{similar[0].name}' — {similar[0].description}", similar[0]

        # ── Step 2: Derive function name ─────────────────────────
        if not function_name:
            function_name = self._derive_function_name(capability)

        # ── Step 3: Generate code ────────────────────────────────
        prompt = SKILL_CREATION_PROMPT.format(
            capability=capability,
            function_name=function_name,
        )

        try:
            code = self.llm_call(prompt).strip()
            # Strip markdown fences if present
            if code.startswith("```"):
                lines = code.split("\n")
                code = "\n".join(lines[1:-1])
        except Exception as e:
            return False, f"LLM call failed: {e}", None

        # ── Step 4: Validate ─────────────────────────────────────
        ok, msg = validate_skill_code(code)
        if not ok:
            log.error("Skill validation failed", reason=msg)
            # Retry once with error context
            retry_prompt = f"{prompt}\n\nPrevious attempt failed: {msg}\nFix this issue."
            try:
                code = self.llm_call(retry_prompt).strip()
                if code.startswith("```"):
                    lines = code.split("\n")
                    code = "\n".join(lines[1:-1])
                ok, msg = validate_skill_code(code)
                if not ok:
                    return False, f"Skill validation failed after retry: {msg}", None
            except Exception as e:
                return False, f"Retry failed: {e}", None

        # ── Step 5: Write to file ────────────────────────────────
        skill_file = self._output_path / f"{function_name}.py"
        full_code = f"from backend.models.skill import SkillResult\n\n{code}\n"
        skill_file.write_text(full_code, encoding="utf-8")
        log.info("Skill file written", path=str(skill_file))

        # ── Step 6: Smoke test ───────────────────────────────────
        module_path = f"generated.skills.{function_name}"
        try:
            if module_path in sys.modules:
                del sys.modules[module_path]
            module = importlib.import_module(module_path)
            fn = getattr(module, function_name)
            log.info("Skill imported successfully", function=function_name)
        except Exception as e:
            skill_file.unlink(missing_ok=True)
            return False, f"Import failed: {e}", None

        # ── Step 7: Register ─────────────────────────────────────
        spec = SkillSpec(
            name=function_name,
            description=capability,
            module_path=module_path,
            function_name=function_name,
            is_generated=True,
            category=SkillCategory.CUSTOM,
            operation_type=OperationType.READ,
            retry_policy=RetryPolicy(max_retries=2, backoff_seconds=1.0),
            permission_manifest=PermissionManifest(),
        )

        registered = skill_registry.register(spec)
        log.info("Skill registered", name=registered.name)
        return True, f"Skill '{function_name}' created and registered", registered

    def _derive_function_name(self, capability: str) -> str:
        import re
        name = capability.lower()
        name = re.sub(r"[^a-z0-9\s]", "", name)
        words = name.split()[:4]
        return "_".join(words) or "custom_skill"


# ── Factory instance (initialised with Brain's LLM call) ─────────
_skill_factory: SkillFactory | None = None


def get_skill_factory(llm_call) -> SkillFactory:
    global _skill_factory
    if _skill_factory is None:
        _skill_factory = SkillFactory(llm_call)
    return _skill_factory