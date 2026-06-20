import json
from pathlib import Path
from typing import Optional
from config import settings
from backend.models.agent import AgentSpec
import structlog

log = structlog.get_logger()


class AgentRegistry:
    def __init__(self):
        self._path = settings.agents_registry_path
        self._agents: dict[str, AgentSpec] = {}
        self.load()

    # ── Load ────────────────────────────────────────────────────
    def load(self):
        self._agents = {}
        if not self._path.exists():
            log.warning("Agent registry not found, starting empty", path=str(self._path))
            return

        with open(self._path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)

        for entry in data.get("agents", []):
            try:
                spec = AgentSpec(**entry)
                self._agents[spec.name] = spec
            except Exception as e:
                log.error("Failed to load agent", name=entry.get("name"), error=str(e))

        log.info("Agent registry loaded", count=len(self._agents))

    # ── Save ────────────────────────────────────────────────────
    def save(self):
        data = {"agents": [a.model_dump() for a in self._agents.values()]}
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        log.info("Agent registry saved", count=len(self._agents))

    # ── CRUD ────────────────────────────────────────────────────
    def register(self, spec: AgentSpec) -> AgentSpec:
        if spec.name in self._agents:
            log.info("Agent updated", name=spec.name, version=spec.version)
        else:
            log.info("Agent registered", name=spec.name)
        self._agents[spec.name] = spec
        self.save()
        return spec

    def get(self, name: str) -> Optional[AgentSpec]:
        return self._agents.get(name)

    def get_all(self) -> list[AgentSpec]:
        return list(self._agents.values())

    def get_active(self) -> list[AgentSpec]:
        return [a for a in self._agents.values() if a.is_active]

    def exists(self, name: str) -> bool:
        return name in self._agents

    def retire(self, name: str) -> bool:
        if name in self._agents:
            self._agents[name].is_active = False
            self.save()
            log.info("Agent retired", name=name)
            return True
        return False

    def count(self) -> int:
        return len([a for a in self._agents.values() if a.is_active])

    # ── Scoring ──────────────────────────────────────────────────
    def update_score(self, name: str, new_score: float) -> bool:
        agent = self.get(name)
        if not agent:
            return False
        agent.performance_score = round(new_score, 2)
        agent.task_count += 1
        self.save()
        log.info("Agent score updated", name=name, score=new_score)
        return True

    def increment_failure(self, name: str, failure_type: str) -> bool:
        agent = self.get(name)
        if not agent:
            return False
        agent.failure_count += 1
        if failure_type in agent.failure_attribution:
            agent.failure_attribution[failure_type] += 1
        self.save()
        return True

    # ── Skill list patching ──────────────────────────────────────
    def patch_skills(self, name: str, add_skills: list[str]) -> bool:
        agent = self.get(name)
        if not agent:
            log.error("Cannot patch skills — agent not found", name=name)
            return False

        added = []
        for skill in add_skills:
            if skill not in agent.skills:
                agent.skills.append(skill)
                added.append(skill)

        if added:
            agent.version += 1
            self.save()
            log.info("Agent skills patched", name=name, added=added, version=agent.version)

        return True

    # ── Routing helpers ──────────────────────────────────────────
    def find_by_domain(self, domain: str) -> list[AgentSpec]:
        return [
            a for a in self.get_active()
            if a.capability_schema.domain.lower() == domain.lower()
        ]

    def find_by_task_type(self, task_type: str) -> list[AgentSpec]:
        return [
            a for a in self.get_active()
            if any(
                task_type.lower() in t.lower()
                for t in a.capability_schema.task_types
            )
        ]

    def get_best_for_domain(self, domain: str) -> Optional[AgentSpec]:
        candidates = self.find_by_domain(domain)
        if not candidates:
            return None
        return max(candidates, key=lambda a: a.performance_score)


# ── Singleton ────────────────────────────────────────────────────
agent_registry = AgentRegistry()