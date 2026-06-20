"""
AEGIS Phase 2 — Preference Engine
Extracts user habits and style defaults from task history.

This module analyses TaskRecord history and produces a UserProfile that the
Brain can inject into its planning context to personalise responses.

Runs on-demand (called by Brain before planning) and after every 10 new tasks
to keep the profile fresh without overhead on every task.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import TaskOutcome, TaskRecord
from .task_store import TaskStore, get_task_store

_PROFILE_PATH = (
    Path(__file__).resolve().parents[2] / "registry_store" / "user_profile.json"
)

# Minimum tasks before extraction is meaningful
_MIN_TASKS_FOR_PROFILE = 5


@dataclass
class UserProfile:
    """
    Distilled user preferences and habits.
    Serialised to registry_store/user_profile.json after each update.
    """

    # ── usage patterns ────────────────────────────────────────────────────────
    preferred_agents: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    top_task_categories: list[str] = field(default_factory=list)

    # ── style defaults (inferred from correction patterns) ────────────────────
    prefers_verbose: bool = False       # tends to ask follow-ups for more detail
    prefers_concise: bool = False       # rarely asks for elaboration
    correction_rate: float = 0.0       # fraction of tasks that got corrected

    # ── temporal habits ───────────────────────────────────────────────────────
    peak_hours: list[int] = field(default_factory=list)   # UTC hours 0-23
    avg_tasks_per_day: float = 0.0

    # ── agent performance context ─────────────────────────────────────────────
    agent_success_rates: dict[str, float] = field(default_factory=dict)
    most_corrected_agents: list[str] = field(default_factory=list)

    # ── frequently used patterns ──────────────────────────────────────────────
    frequent_keywords: list[str] = field(default_factory=list)
    task_count: int = 0
    last_updated: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "preferred_agents": self.preferred_agents,
            "preferred_skills": self.preferred_skills,
            "top_task_categories": self.top_task_categories,
            "prefers_verbose": self.prefers_verbose,
            "prefers_concise": self.prefers_concise,
            "correction_rate": self.correction_rate,
            "peak_hours": self.peak_hours,
            "avg_tasks_per_day": self.avg_tasks_per_day,
            "agent_success_rates": self.agent_success_rates,
            "most_corrected_agents": self.most_corrected_agents,
            "frequent_keywords": self.frequent_keywords,
            "task_count": self.task_count,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UserProfile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def as_brain_context(self) -> str:
        """
        Compact string injected into the Brain's planning prompt.
        Keeps token cost low while giving the LLM enough signal.
        """
        if self.task_count < _MIN_TASKS_FOR_PROFILE:
            return ""

        lines = [f"[User profile — {self.task_count} tasks logged]"]

        if self.preferred_agents:
            lines.append(f"Most-used agents: {', '.join(self.preferred_agents[:3])}")
        if self.top_task_categories:
            lines.append(f"Top task types: {', '.join(self.top_task_categories[:3])}")
        if self.prefers_verbose:
            lines.append("Style: user prefers detailed responses.")
        elif self.prefers_concise:
            lines.append("Style: user prefers concise responses.")
        if self.correction_rate > 0.15:
            lines.append(
                f"Note: {self.correction_rate:.0%} of tasks get corrected — "
                "double-check output quality before responding."
            )
        if self.most_corrected_agents:
            lines.append(
                f"Agents that get corrected most: {', '.join(self.most_corrected_agents[:2])} "
                "— consider extra validation."
            )

        return "\n".join(lines)


class PreferenceEngine:
    """
    Analyses task history and updates the UserProfile.

    Trigger modes
    -------------
    - On-demand: ``engine.refresh()``
    - Auto: the Brain calls ``engine.maybe_refresh()`` every task;
      extraction only runs every _REFRESH_EVERY tasks.
    """

    _REFRESH_EVERY = 10   # re-extract after this many new tasks

    def __init__(self, store: TaskStore | None = None) -> None:
        self._store = store or get_task_store()
        self._profile_path = _PROFILE_PATH
        self._profile_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_extracted_count = 0
        self._profile: UserProfile | None = None

    # ── public API ────────────────────────────────────────────────────────────

    def get_profile(self) -> UserProfile:
        """Return cached profile, loading from disk if needed."""
        if self._profile is None:
            self._profile = self._load_profile()
        return self._profile

    def maybe_refresh(self) -> bool:
        """
        Refresh profile if enough new tasks have accumulated.
        Returns True if a refresh was performed.
        """
        current_count = self._store.count()
        if current_count - self._last_extracted_count >= self._REFRESH_EVERY:
            self.refresh()
            return True
        return False

    def refresh(self) -> UserProfile:
        """Force a full profile rebuild from current task history."""
        tasks = self._store.all()
        profile = self._extract(tasks)
        self._profile = profile
        self._last_extracted_count = len(tasks)
        self._save_profile(profile)
        return profile

    # ── extraction logic ──────────────────────────────────────────────────────

    def _extract(self, tasks: list[TaskRecord]) -> UserProfile:
        if not tasks:
            return UserProfile()

        # ── agent usage ───────────────────────────────────────────────────────
        agent_counter: Counter[str] = Counter()
        agent_corrections: Counter[str] = Counter()
        agent_totals: Counter[str] = Counter()

        for t in tasks:
            for a in t.agents_used:
                agent_counter[a] += 1
                agent_totals[a] += 1
                if t.correction_issued:
                    agent_corrections[a] += 1

        preferred_agents = [a for a, _ in agent_counter.most_common(5)]

        # per-agent success rate
        agent_success: dict[str, float] = {}
        for agent in agent_totals:
            agent_tasks = [t for t in tasks if agent in t.agents_used]
            successes = sum(1 for t in agent_tasks if t.outcome == TaskOutcome.SUCCESS)
            agent_success[agent] = round(successes / len(agent_tasks), 3)

        most_corrected = [
            a for a, _ in agent_corrections.most_common(3)
            if agent_corrections[a] / max(agent_totals[a], 1) > 0.1
        ]

        # ── skill usage ───────────────────────────────────────────────────────
        skill_counter: Counter[str] = Counter()
        for t in tasks:
            for s in t.skills_used:
                skill_counter[s] += 1
        preferred_skills = [s for s, _ in skill_counter.most_common(5)]

        # ── task categories ───────────────────────────────────────────────────
        cat_counter: Counter[str] = Counter(
            t.task_category for t in tasks if t.task_category
        )
        top_categories = [c for c, _ in cat_counter.most_common(3)]

        # ── correction rate + verbosity inference ─────────────────────────────
        correction_rate = sum(1 for t in tasks if t.correction_issued) / len(tasks)

        # Infer verbosity from keywords in raw_input
        verbose_signals = sum(
            1 for t in tasks
            if re.search(
                r"\b(explain|detail|elaborate|why|how does|full|complete)\b",
                t.raw_input, re.I
            )
        )
        concise_signals = sum(
            1 for t in tasks
            if re.search(
                r"\b(brief|short|quick|just|tldr|summary)\b",
                t.raw_input, re.I
            )
        )
        prefers_verbose = verbose_signals > concise_signals and verbose_signals > 3
        prefers_concise = concise_signals > verbose_signals and concise_signals > 3

        # ── temporal patterns ─────────────────────────────────────────────────
        hour_counter: Counter[int] = Counter()
        day_counter: Counter[str] = Counter()
        for t in tasks:
            dt = t.created_at
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            hour_counter[dt.hour] += 1
            day_counter[dt.strftime("%Y-%m-%d")] += 1

        peak_hours = [h for h, _ in hour_counter.most_common(3)]
        avg_per_day = round(len(tasks) / max(len(day_counter), 1), 2)

        # ── keyword extraction (simple) ───────────────────────────────────────
        stop_words = {
            "the","a","an","is","it","to","of","in","and","or","for",
            "me","my","this","that","can","you","i","please","with",
            "on","at","be","do","what","how","get","make","run",
        }
        word_counter: Counter[str] = Counter()
        for t in tasks:
            words = re.findall(r"\b[a-zA-Z]{3,}\b", t.raw_input.lower())
            for w in words:
                if w not in stop_words:
                    word_counter[w] += 1
        frequent_keywords = [w for w, _ in word_counter.most_common(15)]

        return UserProfile(
            preferred_agents=preferred_agents,
            preferred_skills=preferred_skills,
            top_task_categories=top_categories,
            prefers_verbose=prefers_verbose,
            prefers_concise=prefers_concise,
            correction_rate=round(correction_rate, 3),
            peak_hours=peak_hours,
            avg_tasks_per_day=avg_per_day,
            agent_success_rates=agent_success,
            most_corrected_agents=most_corrected,
            frequent_keywords=frequent_keywords,
            task_count=len(tasks),
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

    # ── persistence ───────────────────────────────────────────────────────────

    def _save_profile(self, profile: UserProfile) -> None:
        self._profile_path.write_text(
            json.dumps(profile.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load_profile(self) -> UserProfile:
        if not self._profile_path.exists():
            return UserProfile()
        try:
            text = self._profile_path.read_text(encoding="utf-8-sig")
            return UserProfile.from_dict(json.loads(text))
        except Exception:
            return UserProfile()


# ── module-level singleton ────────────────────────────────────────────────────
_engine: PreferenceEngine | None = None


def get_preference_engine() -> PreferenceEngine:
    global _engine
    if _engine is None:
        _engine = PreferenceEngine()
    return _engine