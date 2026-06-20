import json
import importlib
from pathlib import Path
from typing import Optional
from config import settings
from backend.models.skill import SkillSpec
import structlog

log = structlog.get_logger()


class SkillRegistry:
    def __init__(self):
        self._path = settings.skills_registry_path
        self._skills: dict[str, SkillSpec] = {}
        self.load()

    # ── Load ────────────────────────────────────────────────────
    def load(self):
        self._skills = {}
        if not self._path.exists():
            log.warning("Skills registry not found, starting empty", path=str(self._path))
            return

        with open(self._path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        for entry in data.get("skills", []):
            try:
                spec = SkillSpec(**entry)
                self._skills[spec.name] = spec
            except Exception as e:
                log.error("Failed to load skill", name=entry.get("name"), error=str(e))

        log.info("Skill registry loaded", count=len(self._skills))

    # ── Save ────────────────────────────────────────────────────
    def save(self):
        data = {"skills": [s.model_dump() for s in self._skills.values()]}
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        log.info("Skill registry saved", count=len(self._skills))

    # ── CRUD ────────────────────────────────────────────────────
    def register(self, spec: SkillSpec) -> SkillSpec:
        if spec.name in self._skills:
            existing = self._skills[spec.name]
            spec.version = existing.version + 1
            log.info("Skill updated", name=spec.name, version=spec.version)
        else:
            log.info("Skill registered", name=spec.name, version=spec.version)

        self._skills[spec.name] = spec
        self.save()
        return spec

    def get(self, name: str) -> Optional[SkillSpec]:
        return self._skills.get(name)

    def get_all(self) -> list[SkillSpec]:
        return list(self._skills.values())

    def get_by_category(self, category: str) -> list[SkillSpec]:
        return [s for s in self._skills.values() if s.category == category]

    def exists(self, name: str) -> bool:
        return name in self._skills

    def remove(self, name: str) -> bool:
        if name in self._skills:
            del self._skills[name]
            self.save()
            log.info("Skill removed", name=name)
            return True
        return False

    def count(self) -> int:
        return len(self._skills)

    # ── Callable resolution ──────────────────────────────────────
    def resolve_callable(self, name: str):
        spec = self.get(name)
        if not spec:
            raise ValueError(f"Skill '{name}' not found in registry")
        try:
            module = importlib.import_module(spec.module_path)
            fn = getattr(module, spec.function_name)
            return fn
        except Exception as e:
            raise ImportError(
                f"Could not resolve skill '{name}' "
                f"from {spec.module_path}.{spec.function_name}: {e}"
            )

    # ── Similarity search (pre-KAG, simple keyword match) ───────
    def find_similar(self, description: str, threshold: int = 3) -> list[SkillSpec]:
        # Strip common stop words before matching
        STOP_WORDS = {
            "a", "an", "the", "and", "or", "of", "to", "in", "for",
            "is", "it", "its", "from", "with", "that", "this", "on",
            "at", "by", "as", "are", "be", "was", "were", "been",
            "return", "returns", "result", "results", "using", "use",
            "get", "set", "give", "given", "based", "into",
        }

        query_words = set(description.lower().split()) - STOP_WORDS
        matches = []
        for spec in self._skills.values():
            skill_words = set(spec.description.lower().split()) - STOP_WORDS
            overlap = len(query_words & skill_words)
            if overlap >= threshold:
                matches.append((overlap, spec))
        matches.sort(key=lambda x: x[0], reverse=True)
        return [spec for _, spec in matches]


# ── Singleton ────────────────────────────────────────────────────
skill_registry = SkillRegistry()