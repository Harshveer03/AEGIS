"""
AEGIS Phase 2 Week 2 — Evolution Engine
=========================================
Implements SOW v2 section 5.3: Evolution and Retirement Thresholds.

Condition                       Action
──────────────────────────────  ─────────────────────────────────────────────
Score < 35 after 5+ tasks       Evolution: Brain rewrites system_prompt.
                                 version increments. Agent retains name,
                                 skills, and history.

Score < 20 after 10+ tasks      Retirement: is_active set to False.
                                 Brain notifies user. Factory optionally
                                 creates replacement.

The EvolutionEngine is the only component that writes to agents.json.
It reads scores from AgentScorer and reads/writes AgentSpec via the
agent registry JSON file directly (registry_store/agents.json).

LLM rewriting is injected at construction time so the engine stays
testable without a live LLM.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .agent_scorer import AgentScorer, get_agent_scorer
from .task_store import TaskStore, get_task_store

# ── thresholds (SOW section 5.3) ──────────────────────────────────────────────
_EVOLVE_SCORE_THRESHOLD   = 35.0
_EVOLVE_MIN_TASKS         = 5

_RETIRE_SCORE_THRESHOLD   = 20.0
_RETIRE_MIN_TASKS         = 10

# ── registry path ─────────────────────────────────────────────────────────────
_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "registry_store" / "agents.json"
)


class EvolutionOutcome(str, Enum):
    EVOLVED  = "evolved"
    RETIRED  = "retired"
    STABLE   = "stable"       # score healthy, no action needed
    TOO_NEW  = "too_new"      # not enough tasks to evaluate
    UNKNOWN  = "unknown"      # agent not found in registry


@dataclass
class EvolutionResult:
    agent_name:   str
    outcome:      EvolutionOutcome
    score_before: float
    score_after:  float | None = None   # only set after evolution
    version:      int | None = None     # new version after evolution
    message:      str = ""

    def as_dict(self) -> dict:
        return {
            "agent_name":   self.agent_name,
            "outcome":      self.outcome,
            "score_before": round(self.score_before, 2),
            "score_after":  round(self.score_after, 2) if self.score_after is not None else None,
            "version":      self.version,
            "message":      self.message,
        }


class EvolutionEngine:
    """
    Evaluates agents after each task and applies evolution or retirement
    when SOW thresholds are crossed.

    Parameters
    ----------
    llm_call:   Callable that takes a prompt string and returns LLM response.
                Pass None to run in no-LLM mode (evolution skipped, useful
                for testing retirement and score logic independently).
    store:      TaskStore instance (injectable for tests).
    scorer:     AgentScorer instance (injectable for tests).
    registry_path: Path to agents.json (injectable for tests).
    """

    def __init__(
        self,
        llm_call: Callable[[str], str] | None = None,
        store: TaskStore | None = None,
        scorer: AgentScorer | None = None,
        registry_path: Path | None = None,
    ) -> None:
        self._llm_call     = llm_call
        self._store        = store  or get_task_store()
        self._scorer       = scorer or get_agent_scorer()
        self._registry_path = Path(registry_path) if registry_path else _REGISTRY_PATH
        self._lock         = threading.Lock()

    # ── public API ────────────────────────────────────────────────────────────

    def evaluate(self, agent_name: str) -> EvolutionResult:
        """
        Check an agent's current rolling score and apply evolution or
        retirement if thresholds are crossed.

        Called by the Brain after every task completion.
        """
        score = self._scorer.rolling_score(agent_name)
        tasks = self._store.by_agent(agent_name)
        task_count = len(tasks)

        # ── retirement check (stricter threshold, checked first) ─────────────
        if score < _RETIRE_SCORE_THRESHOLD and task_count >= _RETIRE_MIN_TASKS:
            return self._retire(agent_name, score, task_count)

        # ── evolution check ──────────────────────────────────────────────────
        if score < _EVOLVE_SCORE_THRESHOLD and task_count >= _EVOLVE_MIN_TASKS:
            return self._evolve(agent_name, score, task_count)

        # ── not enough tasks yet ─────────────────────────────────────────────
        if task_count < _EVOLVE_MIN_TASKS:
            return EvolutionResult(
                agent_name=agent_name,
                outcome=EvolutionOutcome.TOO_NEW,
                score_before=score,
                message=f"Only {task_count} tasks — need {_EVOLVE_MIN_TASKS} before evaluation.",
            )

        # ── healthy ──────────────────────────────────────────────────────────
        return EvolutionResult(
            agent_name=agent_name,
            outcome=EvolutionOutcome.STABLE,
            score_before=score,
            message=f"Score {score:.1f} — within healthy range.",
        )

    def get_agent_spec(self, agent_name: str) -> dict | None:
        """Read a single agent spec from agents.json."""
        agents = self._load_registry()
        return next((a for a in agents if a.get("name") == agent_name), None)

    def list_active_agents(self) -> list[dict]:
        """All agents where is_active is True (or field absent)."""
        return [
            a for a in self._load_registry()
            if a.get("is_active", True)
        ]

    # ── evolution ─────────────────────────────────────────────────────────────

    def _evolve(self, agent_name: str, score: float, task_count: int) -> EvolutionResult:
        spec = self.get_agent_spec(agent_name)
        if not spec:
            return EvolutionResult(
                agent_name=agent_name,
                outcome=EvolutionOutcome.UNKNOWN,
                score_before=score,
                message="Agent not found in registry.",
            )

        # Build failure summary for the LLM
        failure_summary = self._build_failure_summary(agent_name)

        new_prompt: str
        if self._llm_call:
            new_prompt = self._rewrite_prompt(spec, failure_summary)
        else:
            # No LLM available (test mode) — append a generic improvement note
            new_prompt = (
                spec.get("system_prompt", "") +
                "\n\n[EVOLVED] Focus on accuracy and completeness. "
                "Avoid empty responses. Double-check all outputs before returning."
            )

        old_version = spec.get("version", 1)
        new_version = old_version + 1

        self._patch_agent(agent_name, {
            "system_prompt": new_prompt,
            "version":       new_version,
        })

        # Re-compute score after evolution (same history, just updated prompt)
        score_after = self._scorer.rolling_score(agent_name)

        return EvolutionResult(
            agent_name=agent_name,
            outcome=EvolutionOutcome.EVOLVED,
            score_before=score,
            score_after=score_after,
            version=new_version,
            message=(
                f"Score {score:.1f} below {_EVOLVE_SCORE_THRESHOLD} after {task_count} tasks. "
                f"System prompt rewritten. Version {old_version} → {new_version}."
            ),
        )

    def _rewrite_prompt(self, spec: dict, failure_summary: str) -> str:
        """Ask the LLM to produce an improved system prompt."""
        prompt = _EVOLUTION_PROMPT.format(
            agent_name=spec.get("name", "unknown"),
            description=spec.get("description", ""),
            current_prompt=spec.get("system_prompt", ""),
            skills=", ".join(spec.get("skills", [])),
            failure_summary=failure_summary,
        )
        return self._llm_call(prompt).strip()

    def _build_failure_summary(self, agent_name: str) -> str:
        """Summarise recent failures for the LLM rewrite context."""
        tasks = self._store.by_agent(agent_name)
        recent = tasks[-20:]
        failures = [t for t in recent if t.outcome != "success"]
        corrections = [t for t in recent if t.correction_issued]

        lines = [f"Recent tasks: {len(recent)}, Failures: {len(failures)}, Corrections: {len(corrections)}"]
        for t in failures[-5:]:
            if t.error_log:
                lines.append(f"  Error: {t.error_log[:200]}")
        return "\n".join(lines)

    # ── retirement ────────────────────────────────────────────────────────────

    def _retire(self, agent_name: str, score: float, task_count: int) -> EvolutionResult:
        spec = self.get_agent_spec(agent_name)
        if not spec:
            return EvolutionResult(
                agent_name=agent_name,
                outcome=EvolutionOutcome.UNKNOWN,
                score_before=score,
                message="Agent not found in registry.",
            )

        self._patch_agent(agent_name, {
            "is_active":   False,
            "retired_at":  datetime.now(timezone.utc).isoformat(),
            "retire_reason": (
                f"Score {score:.1f} below {_RETIRE_SCORE_THRESHOLD} "
                f"after {task_count} tasks."
            ),
        })

        return EvolutionResult(
            agent_name=agent_name,
            outcome=EvolutionOutcome.RETIRED,
            score_before=score,
            message=(
                f"Score {score:.1f} below {_RETIRE_SCORE_THRESHOLD} after {task_count} tasks. "
                f"Agent archived. is_active set to False."
            ),
        )

    # ── registry I/O ──────────────────────────────────────────────────────────

    def _load_registry(self) -> list[dict]:
        if not self._registry_path.exists():
            return []
        try:
            text = self._registry_path.read_text(encoding="utf-8-sig")
            data = json.loads(text) if text.strip() else []
            # registry can be a list or {"agents": [...]}
            if isinstance(data, dict):
                return data.get("agents", [])
            return data
        except (json.JSONDecodeError, OSError):
            return []

    def _save_registry(self, agents: list[dict]) -> None:
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        # preserve original wrapper format if it was a dict
        if self._registry_path.exists():
            try:
                text = self._registry_path.read_text(encoding="utf-8-sig")
                raw = json.loads(text)
                if isinstance(raw, dict):
                    raw["agents"] = agents
                    self._registry_path.write_text(
                        json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                    return
            except Exception:
                pass
        self._registry_path.write_text(
            json.dumps(agents, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _patch_agent(self, agent_name: str, fields: dict[str, Any]) -> bool:
        """Atomically update specific fields on an agent spec."""
        with self._lock:
            agents = self._load_registry()
            for agent in agents:
                if agent.get("name") == agent_name:
                    agent.update(fields)
                    self._save_registry(agents)
                    return True
        return False

    def _add_agent(self, spec: dict) -> None:
        """Insert a new agent spec (used by tests and factory integration)."""
        with self._lock:
            agents = self._load_registry()
            agents.append(spec)
            self._save_registry(agents)


# ── LLM prompt for evolution ──────────────────────────────────────────────────
_EVOLUTION_PROMPT = """\
You are the AEGIS Brain evolution engine. An agent is underperforming and needs a better system prompt.

Agent name: {agent_name}
Description: {description}
Available skills: {skills}

Current system prompt:
\"\"\"
{current_prompt}
\"\"\"

Recent failure summary:
{failure_summary}

Write an improved system prompt for this agent. The new prompt must:
1. Correct the patterns that caused failures (see failure summary above).
2. Be specific and directive — tell the agent exactly how to handle edge cases.
3. Keep the same domain focus and skill usage.
4. Be concise — under 400 words.

Return ONLY the new system prompt text. No preamble, no explanation, no quotes.
"""


# ── module-level singleton ────────────────────────────────────────────────────
_engine: EvolutionEngine | None = None


def get_evolution_engine(llm_call: Callable[[str], str] | None = None) -> EvolutionEngine:
    global _engine
    if _engine is None:
        _engine = EvolutionEngine(llm_call=llm_call)
    return _engine
