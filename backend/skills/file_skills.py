import os
import shutil
from pathlib import Path
from backend.models.skill import SkillResult


def read_file(path: str) -> SkillResult:
    try:
        p = Path(path)
        if not p.exists():
            return SkillResult(success=False, error=f"File not found: {path}")
        if not p.is_file():
            return SkillResult(success=False, error=f"Path is not a file: {path}")
        content = p.read_text(encoding="utf-8")
        return SkillResult(success=True, data={"content": content, "path": str(p)})
    except Exception as e:
        return SkillResult(success=False, error=str(e))


def write_file(path: str, content: str) -> SkillResult:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return SkillResult(success=True, data={"path": str(p), "bytes_written": len(content)})
    except Exception as e:
        return SkillResult(success=False, error=str(e))


def append_file(path: str, content: str) -> SkillResult:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(content)
        return SkillResult(success=True, data={"path": str(p)})
    except Exception as e:
        return SkillResult(success=False, error=str(e))


def delete_file(path: str) -> SkillResult:
    try:
        p = Path(path)
        if not p.exists():
            return SkillResult(success=False, error=f"File not found: {path}")
        p.unlink()
        return SkillResult(success=True, data={"deleted": str(p)})
    except Exception as e:
        return SkillResult(success=False, error=str(e))


def list_directory(path: str) -> SkillResult:
    try:
        p = Path(path)
        if not p.exists():
            return SkillResult(success=False, error=f"Directory not found: {path}")
        entries = [
            {"name": e.name, "is_file": e.is_file(), "size": e.stat().st_size if e.is_file() else 0}
            for e in p.iterdir()
        ]
        return SkillResult(success=True, data={"entries": entries, "path": str(p)})
    except Exception as e:
        return SkillResult(success=False, error=str(e))


def search_files(directory: str, pattern: str) -> SkillResult:
    try:
        p = Path(directory)
        if not p.exists():
            return SkillResult(success=False, error=f"Directory not found: {directory}")
        matches = [str(f) for f in p.rglob(pattern)]
        return SkillResult(success=True, data={"matches": matches, "count": len(matches)})
    except Exception as e:
        return SkillResult(success=False, error=str(e))


def move_file(source: str, destination: str) -> SkillResult:
    try:
        src = Path(source)
        dst = Path(destination)
        if not src.exists():
            return SkillResult(success=False, error=f"Source not found: {source}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return SkillResult(success=True, data={"from": str(src), "to": str(dst)})
    except Exception as e:
        return SkillResult(success=False, error=str(e))


def copy_file(source: str, destination: str) -> SkillResult:
    try:
        src = Path(source)
        dst = Path(destination)
        if not src.exists():
            return SkillResult(success=False, error=f"Source not found: {source}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        return SkillResult(success=True, data={"from": str(src), "to": str(dst)})
    except Exception as e:
        return SkillResult(success=False, error=str(e))