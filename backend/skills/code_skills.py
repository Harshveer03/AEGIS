import subprocess
import tempfile
import os
from pathlib import Path
from backend.models.skill import SkillResult


def run_python(code: str, timeout: int = 30) -> SkillResult:
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        os.unlink(tmp_path)

        return SkillResult(
            success=result.returncode == 0,
            error=result.stderr if result.returncode != 0 else None,
            data={"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
        )
    except subprocess.TimeoutExpired:
        return SkillResult(success=False, error=f"Code execution timed out after {timeout}s")
    except Exception as e:
        return SkillResult(success=False, error=str(e))


def validate_syntax(code: str) -> SkillResult:
    try:
        compile(code, "<string>", "exec")
        return SkillResult(success=True, data={"valid": True, "message": "Syntax is valid"})
    except SyntaxError as e:
        return SkillResult(
            success=False,
            error=str(e),
            data={"valid": False, "line": e.lineno, "message": str(e.msg)}
        )


def run_python_with_input(code: str, stdin_input: str = "", timeout: int = 30) -> SkillResult:
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            ["python", tmp_path],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        os.unlink(tmp_path)

        return SkillResult(
            success=result.returncode == 0,
            error=result.stderr if result.returncode != 0 else None,
            data={"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
        )
    except subprocess.TimeoutExpired:
        return SkillResult(success=False, error=f"Code execution timed out after {timeout}s")
    except Exception as e:
        return SkillResult(success=False, error=str(e))