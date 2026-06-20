import ast
import re
from backend.models.skill import SkillSpec
import structlog

log = structlog.get_logger()

# Operations that are never allowed in generated skills
BANNED_PATTERNS = [
    r"os\.system\(",
    r"subprocess\.call\(",
    r"subprocess\.Popen\(",
    r"eval\(",
    r"exec\(",
    r"__import__\(",
    r"open\([^)]*['\"]w['\"]",   # open() in write mode
    r"shutil\.rmtree\(",
    r"os\.remove\(",
    r"os\.unlink\(",
]


def safety_scan(code: str) -> tuple[bool, str]:
    for pattern in BANNED_PATTERNS:
        if re.search(pattern, code):
            return False, f"Banned pattern detected: {pattern}"
    return True, "OK"


def syntax_check(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
        return True, "OK"
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def validate_skill_code(code: str) -> tuple[bool, str]:
    ok, msg = syntax_check(code)
    if not ok:
        return False, msg
    ok, msg = safety_scan(code)
    if not ok:
        return False, msg
    return True, "OK"


def validate_skill_spec(spec: dict) -> tuple[bool, str]:
    required = ["name", "description", "module_path", "function_name"]
    for field in required:
        if not spec.get(field):
            return False, f"Missing required field: {field}"

    name = spec["name"]
    if not re.match(r"^[a-z][a-z0-9_]*$", name):
        return False, f"Invalid skill name '{name}' — must be snake_case"

    return True, "OK"


def validate_agent_spec(spec: dict) -> tuple[bool, str]:
    required = ["name", "description", "system_prompt", "skills"]
    for field in required:
        if not spec.get(field):
            return False, f"Missing required field: {field}"

    name = spec["name"]
    if not re.match(r"^[a-z][a-z0-9_]*$", name):
        return False, f"Invalid agent name '{name}' — must be snake_case"

    if not isinstance(spec.get("skills"), list) or len(spec["skills"]) == 0:
        return False, "Agent must have at least one skill"

    return True, "OK"