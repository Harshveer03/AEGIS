"""
AEGIS Phase 2 — Week 1 Acceptance Tests
Task memory + preference learning

Run from project root:
    python -m pytest backend/memory/tests/test_week1.py -v

All tests use a temporary directory — nothing touches your real task store.
"""

import json
import sys
import time
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.memory.schemas import QualitySignals, TaskOutcome, TaskRecord
from backend.memory.task_store import TaskStore
from backend.memory.preference_engine import PreferenceEngine
from backend.memory.brain_integration import TaskLogger


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_store(tmp_path):
    """TaskStore backed by a temp file — isolated per test."""
    return TaskStore(path=tmp_path / "tasks.json")


@pytest.fixture
def tmp_engine(tmp_path):
    """PreferenceEngine with isolated store and profile path."""
    store = TaskStore(path=tmp_path / "tasks.json")
    engine = PreferenceEngine(store=store)
    engine._profile_path = tmp_path / "user_profile.json"
    return store, engine


@pytest.fixture
def tmp_logger(tmp_path):
    """TaskLogger backed by isolated store."""
    store = TaskStore(path=tmp_path / "tasks.json")
    logger = TaskLogger()
    logger._store = store
    logger._engine = PreferenceEngine(store=store)
    logger._engine._profile_path = tmp_path / "user_profile.json"
    return logger, store


# ══════════════════════════════════════════════════════════════════════════════
# 1. TaskRecord schema
# ══════════════════════════════════════════════════════════════════════════════

class TestTaskRecord:

    def test_default_fields(self):
        t = TaskRecord(raw_input="search for python tutorials")
        assert t.task_id
        assert t.outcome == TaskOutcome.FAILED   # default until set
        assert t.agents_used == []
        assert t.correction_issued is False
        assert t.created_at.tzinfo is not None   # timezone-aware

    def test_roundtrip_json(self):
        t = TaskRecord(
            raw_input="write a report",
            agents_used=["file_agent", "code_agent"],
            outcome=TaskOutcome.SUCCESS,
            duration_ms=1234,
            correction_issued=True,
            task_category="file",
        )
        d = t.to_json_dict()
        assert isinstance(d["created_at"], str)  # serialisable
        t2 = TaskRecord.from_json_dict(d)
        assert t2.task_id == t.task_id
        assert t2.outcome == TaskOutcome.SUCCESS
        assert t2.agents_used == ["file_agent", "code_agent"]
        assert t2.duration_ms == 1234

    def test_quality_signals_extra_fields(self):
        qs = QualitySignals(task_rerun=True, notes="looked wrong")
        t = TaskRecord(raw_input="x", quality_signals=qs)
        d = t.to_json_dict()
        t2 = TaskRecord.from_json_dict(d)
        assert t2.quality_signals.task_rerun is True


# ══════════════════════════════════════════════════════════════════════════════
# 2. TaskStore
# ══════════════════════════════════════════════════════════════════════════════

class TestTaskStore:

    def _make(self, **kwargs) -> TaskRecord:
        defaults = dict(
            raw_input="test task",
            outcome=TaskOutcome.SUCCESS,
            agents_used=["search_agent"],
            duration_ms=500,
            task_category="search",
        )
        defaults.update(kwargs)
        return TaskRecord(**defaults)

    def test_insert_and_count(self, tmp_store):
        assert tmp_store.count() == 0
        tmp_store.insert(self._make())
        assert tmp_store.count() == 1

    def test_persist_across_instances(self, tmp_path):
        path = tmp_path / "tasks.json"
        s1 = TaskStore(path=path)
        t = self._make(raw_input="persistent task")
        s1.insert(t)

        s2 = TaskStore(path=path)
        records = s2.all()
        assert len(records) == 1
        assert records[0].raw_input == "persistent task"

    def test_get_by_id(self, tmp_store):
        t = self._make()
        tmp_store.insert(t)
        fetched = tmp_store.get(t.task_id)
        assert fetched is not None
        assert fetched.task_id == t.task_id

    def test_get_missing_returns_none(self, tmp_store):
        assert tmp_store.get("nonexistent-id") is None

    def test_recent_newest_first(self, tmp_store):
        for i in range(5):
            tmp_store.insert(self._make(raw_input=f"task {i}"))
        recent = tmp_store.recent(n=3)
        assert len(recent) == 3
        assert recent[0].raw_input == "task 4"

    def test_by_agent(self, tmp_store):
        tmp_store.insert(self._make(agents_used=["file_agent"]))
        tmp_store.insert(self._make(agents_used=["search_agent"]))
        tmp_store.insert(self._make(agents_used=["file_agent", "code_agent"]))

        file_tasks = tmp_store.by_agent("file_agent")
        assert len(file_tasks) == 2

    def test_mark_correction(self, tmp_store):
        t = self._make()
        tmp_store.insert(t)
        result = tmp_store.mark_correction(t.task_id)
        assert result is True
        updated = tmp_store.get(t.task_id)
        assert updated.correction_issued is True

    def test_mark_correction_missing(self, tmp_store):
        assert tmp_store.mark_correction("bad-id") is False

    def test_success_rate(self, tmp_store):
        tmp_store.insert(self._make(outcome=TaskOutcome.SUCCESS))
        tmp_store.insert(self._make(outcome=TaskOutcome.SUCCESS))
        tmp_store.insert(self._make(outcome=TaskOutcome.FAILED))
        assert tmp_store.success_rate() == pytest.approx(2/3)

    def test_agent_median_duration(self, tmp_store):
        for ms in [100, 200, 300, 400, 500]:
            tmp_store.insert(self._make(agents_used=["file_agent"], duration_ms=ms))
        assert tmp_store.agent_median_duration("file_agent") == 300.0

    def test_summary_structure(self, tmp_store):
        tmp_store.insert(self._make())
        s = tmp_store.summary()
        assert "total" in s
        assert "outcomes" in s
        assert "correction_rate" in s

    def test_utf8_bom_resilience(self, tmp_path):
        """Store must not crash on files saved with BOM (common on Windows)."""
        path = tmp_path / "tasks.json"
        t = TaskRecord(raw_input="bom test", outcome=TaskOutcome.SUCCESS)
        data = json.dumps([t.to_json_dict()], ensure_ascii=False)
        path.write_bytes(b"\xef\xbb\xbf" + data.encode("utf-8"))
        store = TaskStore(path=path)
        assert store.count() == 1


# ══════════════════════════════════════════════════════════════════════════════
# 3. PreferenceEngine
# ══════════════════════════════════════════════════════════════════════════════

class TestPreferenceEngine:

    def _populate(self, store, n=25):
        """Insert n varied tasks for extraction tests."""
        agents = ["file_agent", "search_agent", "code_agent"]
        cats = ["file", "search", "code"]
        for i in range(n):
            store.insert(TaskRecord(
                raw_input=f"{'explain' if i % 3 == 0 else 'find'} something {i}",
                agents_used=[agents[i % 3]],
                skills_used=[f"{cats[i % 3]}_skill"],
                outcome=TaskOutcome.SUCCESS if i % 5 != 0 else TaskOutcome.FAILED,
                duration_ms=200 + i * 10,
                correction_issued=(i % 8 == 0),
                task_category=cats[i % 3],
                created_at=datetime.now(timezone.utc),
            ))

    def test_empty_store_returns_blank_profile(self, tmp_engine):
        _, engine = tmp_engine
        profile = engine.refresh()
        assert profile.task_count == 0

    def test_preferred_agents_populated(self, tmp_engine):
        store, engine = tmp_engine
        self._populate(store, n=25)
        profile = engine.refresh()
        assert len(profile.preferred_agents) > 0
        assert profile.task_count == 25

    def test_top_categories_populated(self, tmp_engine):
        store, engine = tmp_engine
        self._populate(store, n=25)
        profile = engine.refresh()
        assert len(profile.top_task_categories) > 0

    def test_correction_rate_calculated(self, tmp_engine):
        store, engine = tmp_engine
        self._populate(store, n=24)  # every 8th gets corrected → 3/24 = 0.125
        profile = engine.refresh()
        assert 0.0 < profile.correction_rate < 1.0

    def test_verbosity_detection(self, tmp_engine):
        store, engine = tmp_engine
        for _ in range(10):
            store.insert(TaskRecord(
                raw_input="explain this in detail and elaborate fully",
                agents_used=["search_agent"],
                outcome=TaskOutcome.SUCCESS,
            ))
        profile = engine.refresh()
        assert profile.prefers_verbose is True
        assert profile.prefers_concise is False

    def test_profile_persists_to_disk(self, tmp_engine, tmp_path):
        store, engine = tmp_engine
        self._populate(store, n=10)
        engine.refresh()
        assert engine._profile_path.exists()
        loaded = engine._load_profile()
        assert loaded.task_count == 10

    def test_maybe_refresh_triggers_after_threshold(self, tmp_engine):
        store, engine = tmp_engine
        self._populate(store, n=10)
        refreshed = engine.maybe_refresh()
        assert refreshed is True

    def test_maybe_refresh_skips_before_threshold(self, tmp_engine):
        store, engine = tmp_engine
        self._populate(store, n=5)
        engine._last_extracted_count = 0   # simulate already at 5
        engine._last_extracted_count = 5   # simulate 5 tasks already extracted
        # only 0 new tasks → no refresh
        refreshed = engine.maybe_refresh()
        assert refreshed is False

    def test_brain_context_empty_below_minimum(self, tmp_engine):
        _, engine = tmp_engine
        profile = engine.refresh()
        assert profile.as_brain_context() == ""

    def test_brain_context_non_empty_above_minimum(self, tmp_engine):
        store, engine = tmp_engine
        self._populate(store, n=10)
        profile = engine.refresh()
        ctx = profile.as_brain_context()
        assert "[User profile" in ctx


# ══════════════════════════════════════════════════════════════════════════════
# 4. TaskLogger (Brain integration)
# ══════════════════════════════════════════════════════════════════════════════

class TestTaskLogger:

    def test_basic_logging(self, tmp_logger):
        logger, store = tmp_logger
        with logger.task("find the latest AI news") as ctx:
            ctx.add_agent("search_agent")
            ctx.add_skill("web_search")
            ctx.set_outcome("success")
            ctx.set_category("search")

        assert store.count() == 1
        records = store.all()
        t = records[0]
        assert t.raw_input == "find the latest AI news"
        assert t.agents_used == ["search_agent"]
        assert t.skills_used == ["web_search"]
        assert t.outcome == TaskOutcome.SUCCESS
        assert t.task_category == "search"

    def test_duration_recorded(self, tmp_logger):
        logger, store = tmp_logger
        with logger.task("slow task") as ctx:
            time.sleep(0.05)   # 50 ms
            ctx.set_outcome("success")
        t = store.all()[0]
        assert t.duration_ms >= 40   # allow some clock slack

    def test_exception_sets_failed_outcome(self, tmp_logger):
        logger, store = tmp_logger
        with pytest.raises(ValueError):
            with logger.task("broken task") as ctx:
                raise ValueError("something went wrong")
        t = store.all()[0]
        assert t.outcome == TaskOutcome.FAILED
        assert "ValueError" in (t.error_log or "")

    def test_task_id_accessible_after_context(self, tmp_logger):
        logger, store = tmp_logger
        with logger.task("get task id") as ctx:
            ctx.set_outcome("success")
        assert ctx.task_id != ""
        assert store.get(ctx.task_id) is not None

    def test_flag_correction(self, tmp_logger):
        logger, store = tmp_logger
        with logger.task("write a file") as ctx:
            ctx.add_agent("file_agent")
            ctx.set_outcome("partial")
        flagged = logger.flag_correction(ctx.task_id)
        assert flagged is True
        t = store.get(ctx.task_id)
        assert t.correction_issued is True

    def test_multiple_tasks_accumulate(self, tmp_logger):
        logger, store = tmp_logger
        for i in range(5):
            with logger.task(f"task {i}") as ctx:
                ctx.set_outcome("success")
        assert store.count() == 5

    def test_brain_context_string_returned(self, tmp_logger):
        logger, _ = tmp_logger
        ctx_str = logger.get_brain_context()
        assert isinstance(ctx_str, str)  # empty or populated — just not an error


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import subprocess, sys
    subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        check=False,
    )