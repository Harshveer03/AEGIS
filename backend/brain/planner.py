import json
import re
from typing import Optional
from backend.registry.skill_registry import skill_registry
import structlog

log = structlog.get_logger()


class TaskPlanner:

    AMBIGUITY_SIGNALS = [
        ("missing_path",    ["file", "document", "folder"], ["path", "location", "where", "which", "/"]),
        ("missing_query",   ["search", "find", "look up"],  ["for", "about", "on", "query"]),
        ("missing_url",     ["fetch", "website", "page"],   ["http", "www", "url", ".com"]),
        ("vague_scope",     ["do", "handle", "process"],    ["what", "how", "which"]),
    ]

    def assess_ambiguity(self, task_input: str) -> Optional[str]:
        task_lower = task_input.lower()
        words = set(task_lower.split())

        for ambiguity_type, triggers, required_signals in self.AMBIGUITY_SIGNALS:
            has_trigger = any(t in task_lower for t in triggers)
            has_signal = any(s in task_lower for s in required_signals)
            if has_trigger and not has_signal and len(task_input.split()) < 4:
                return ambiguity_type

        return None

    def build_plan(self, routing: dict) -> dict:
        return {
            "agent": routing.get("agent"),
            "goal": routing.get("goal", ""),
            "context": routing.get("context", {}),
            "confidence": routing.get("confidence", 0.5),
            "fallback_agent": self._get_fallback(routing.get("agent")),
            "needs_factory": routing.get("needs_factory", False),
        }

    def _get_fallback(self, agent_name: Optional[str]) -> Optional[str]:
        fallbacks = {
            "browser_agent": "search_agent",
            "search_agent":  "browser_agent",
            "file_agent":    None,
            "code_agent":    None,
        }
        return fallbacks.get(agent_name)

    def get_skills_summary(self) -> str:
        skills = skill_registry.get_all()
        return "\n".join([f"- {s.name}: {s.description}" for s in skills])