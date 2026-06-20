"""
AEGIS Phase 2 — Task Store
JSON-backed persistent task log.

Design choices:
  - Keeps JSON store as-is from Phase 1 (no MongoDB yet).
  - Single file: registry_store/tasks.json
  - Thread-safe via a file-level lock (threading.Lock).
  - Loads entire store into memory on first access; flushes on every write.
    Acceptable for personal-scale usage (thousands of tasks, not millions).
  - Public API mirrors what MongoDB would expose so the swap is a one-file change.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .schemas import TaskOutcome, TaskRecord

# ── storage path ──────────────────────────────────────────────────────────────
_DEFAULT_STORE = Path(__file__).resolve().parents[2] / "registry_store" / "tasks.json"


class TaskStore:
    """
    Persistent JSON task log.

    Usage
    -----
        store = TaskStore()
        store.insert(task_record)
        recent = store.recent(n=10)
        by_agent = store.by_agent("file_agent", limit=20)
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path else _DEFAULT_STORE
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: list[dict[str, Any]] | None = None

    # ── private helpers ───────────────────────────────────────────────────────

    def _load(self) -> list[dict[str, Any]]:
        if self._cache is not None:
            return self._cache
        if not self._path.exists():
            self._cache = []
            return self._cache
        try:
            text = self._path.read_text(encoding="utf-8-sig")
            self._cache = json.loads(text) if text.strip() else []
        except (json.JSONDecodeError, OSError):
            self._cache = []
        return self._cache

    def _flush(self) -> None:
        self._path.write_text(
            json.dumps(self._cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── write API ─────────────────────────────────────────────────────────────

    def insert(self, task: TaskRecord) -> None:
        """Append a completed task to the store."""
        with self._lock:
            records = self._load()
            records.append(task.to_json_dict())
            self._flush()

    def update(self, task_id: str, **fields: Any) -> bool:
        """
        Patch specific fields on an existing record.
        Returns True if the record was found and updated.
        """
        with self._lock:
            records = self._load()
            for rec in records:
                if rec.get("task_id") == task_id:
                    rec.update(fields)
                    self._flush()
                    return True
        return False

    def mark_correction(self, task_id: str) -> bool:
        """Called when user corrects AEGIS within the 60-second window."""
        return self.update(task_id, correction_issued=True)

    # ── read API ──────────────────────────────────────────────────────────────

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            records = self._load()
        for rec in records:
            if rec.get("task_id") == task_id:
                return TaskRecord.from_json_dict(dict(rec))
        return None

    def all(self) -> list[TaskRecord]:
        with self._lock:
            records = list(self._load())
        return [TaskRecord.from_json_dict(dict(r)) for r in records]

    def recent(self, n: int = 20) -> list[TaskRecord]:
        """Return the n most recent tasks, newest first."""
        return self.all()[-n:][::-1]

    def by_agent(self, agent_name: str, limit: int = 50) -> list[TaskRecord]:
        """All tasks that used a specific agent."""
        return [
            t for t in self.all()
            if agent_name in t.agents_used
        ][-limit:]

    def by_outcome(self, outcome: TaskOutcome, limit: int = 50) -> list[TaskRecord]:
        return [
            t for t in self.all()
            if t.outcome == outcome
        ][-limit:]

    def since(self, dt: datetime) -> list[TaskRecord]:
        """Tasks created after a given UTC datetime."""
        return [
            t for t in self.all()
            if t.created_at >= dt
        ]

    def last_n_days(self, days: int = 7) -> list[TaskRecord]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return self.since(cutoff)

    # ── analytics helpers ─────────────────────────────────────────────────────

    def success_rate(self, agent_name: str | None = None) -> float:
        """Overall or per-agent success rate (0.0–1.0)."""
        tasks = self.by_agent(agent_name) if agent_name else self.all()
        if not tasks:
            return 0.0
        successes = sum(1 for t in tasks if t.outcome == TaskOutcome.SUCCESS)
        return successes / len(tasks)

    def agent_median_duration(self, agent_name: str) -> float | None:
        """Median wall-clock duration (ms) for a given agent."""
        tasks = [t for t in self.by_agent(agent_name) if t.duration_ms > 0]
        if not tasks:
            return None
        durations = sorted(t.duration_ms for t in tasks)
        mid = len(durations) // 2
        if len(durations) % 2 == 0:
            return (durations[mid - 1] + durations[mid]) / 2
        return float(durations[mid])

    def count(self) -> int:
        with self._lock:
            return len(self._load())

    def summary(self) -> dict[str, Any]:
        """Quick stats dict — useful for Brain context and debugging."""
        tasks = self.all()
        if not tasks:
            return {"total": 0}

        from collections import Counter
        outcomes = Counter(t.outcome for t in tasks)
        agents = Counter(a for t in tasks for a in t.agents_used)
        categories = Counter(t.task_category for t in tasks if t.task_category)

        return {
            "total": len(tasks),
            "outcomes": dict(outcomes),
            "top_agents": dict(agents.most_common(5)),
            "top_categories": dict(categories.most_common(5)),
            "correction_rate": round(
                sum(1 for t in tasks if t.correction_issued) / len(tasks), 3
            ),
            "avg_duration_ms": round(
                sum(t.duration_ms for t in tasks) / len(tasks)
            ),
        }


# ── module-level singleton ────────────────────────────────────────────────────
_store: TaskStore | None = None


def get_task_store() -> TaskStore:
    """Return the shared TaskStore instance (lazy-initialised)."""
    global _store
    if _store is None:
        _store = TaskStore()
    return _store