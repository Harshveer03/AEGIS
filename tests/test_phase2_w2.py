"""
AEGIS Phase 2 — Week 2 Acceptance Tests
Agent scoring + evolution + retirement

Run from project root:
    python -m pytest tests/test_phase2_w2.py -v

All tests use temp directories — nothing touches real registry or task store.
"""

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.memory.schemas import TaskOutcome, TaskRecord
from backend.memory.task_store import TaskStore
from backend.memory.agent_scorer import AgentScorer, _W_SUCCESS, _W_NO_CORRECT, _W_SPEED, _W_NON_EMPTY
from backend.memory.evolution_engine import (
    EvolutionEngine, EvolutionOutcome,
    _EVOLVE_SCORE_THRESHOLD, _EVOLVE_MIN_TASKS,
    _RETIRE_SCORE_THRESHOLD, _RETIRE_MIN_TASKS,
)
from backend.memory.brain_integration import TaskLogger
from backend.memory.preference_engine import PreferenceEngine


# ── helpers ───────────────────────────────────────────────────────────────────

def make_task(
    agents=None,
    outcome=TaskOutcome.SUCCESS,
    correction=False,
    duration_ms=500,
    error_log=None,
) -> TaskRecord:
    return TaskRecord(
        raw_input="test task",
        agents_used=agents or ["search_agent"],
        outcome=outcome,
        duration_ms=duration_ms,
        correction_issued=correction,
        error_log=error_log,
    )


def agent_spec(name="search_agent", score=50, version=1, active=True) -> dict:
    return {
        "name": name,
        "description": f"{name} description",
        "system_prompt": f"You are the {name}.",
        "skills": ["web_search"],
        "performance_score": score,
        "task_count": 0,
        "failure_count": 0,
        "version": version,
        "is_active": active,
    }


@pytest.fixture
def tmp_store(tmp_path):
    return TaskStore(path=tmp_path / "tasks.json")


@pytest.fixture
def tmp_scorer(tmp_path):
    store = TaskStore(path=tmp_path / "tasks.json")
    return AgentScorer(store=store), store


@pytest.fixture
def tmp_evo(tmp_path):
    store = TaskStore(path=tmp_path / "tasks.json")
    scorer = AgentScorer(store=store)
    registry = tmp_path / "agents.json"
    engine = EvolutionEngine(
        llm_call=None,   # no-LLM mode
        store=store,
        scorer=scorer,
        registry_path=registry,
    )
    return engine, store, registry


@pytest.fixture
def tmp_logger(tmp_path):
    store = TaskStore(path=tmp_path / "tasks.json")
    pref_engine = PreferenceEngine(store=store)
    pref_engine._profile_path = tmp_path / "user_profile.json"
    scorer = AgentScorer(store=store)
    registry = tmp_path / "agents.json"
    evo = EvolutionEngine(llm_call=None, store=store, scorer=scorer, registry_path=registry)

    logger = TaskLogger()
    logger._store  = store
    logger._engine = pref_engine
    logger._scorer = scorer
    logger._evo    = evo
    return logger, store, evo


# ══════════════════════════════════════════════════════════════════════════════
# 1. AgentScorer — signal calculation
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentScorer:

    def test_perfect_task_scores_100(self, tmp_scorer):
        scorer, store = tmp_scorer
        task = make_task(outcome=TaskOutcome.SUCCESS, correction=False, duration_ms=300)
        store.insert(task)
        bd = scorer.score_task(task, "search_agent", output_non_empty=True)
        # speed skipped (< 5 prior tasks) → all four signals = 1.0
        assert bd.task_score == pytest.approx(100.0)

    def test_failed_task_loses_success_weight(self, tmp_scorer):
        scorer, store = tmp_scorer
        task = make_task(outcome=TaskOutcome.FAILED, correction=False)
        store.insert(task)
        bd = scorer.score_task(task, "search_agent", output_non_empty=False)
        # success=0, no_correct=1, speed=1(skipped), non_empty=0
        expected = (_W_NO_CORRECT + _W_SPEED) * 100
        assert bd.task_score == pytest.approx(expected)

    def test_correction_reduces_score(self, tmp_scorer):
        scorer, store = tmp_scorer
        task = make_task(outcome=TaskOutcome.SUCCESS, correction=True)
        store.insert(task)
        bd = scorer.score_task(task, "search_agent", output_non_empty=True)
        # no_correct=0
        expected = (_W_SUCCESS + _W_SPEED + _W_NON_EMPTY) * 100
        assert bd.task_score == pytest.approx(expected)

    def test_empty_output_reduces_score(self, tmp_scorer):
        scorer, store = tmp_scorer
        task = make_task(outcome=TaskOutcome.SUCCESS, correction=False)
        store.insert(task)
        bd = scorer.score_task(task, "search_agent", output_non_empty=False)
        expected = (_W_SUCCESS + _W_NO_CORRECT + _W_SPEED) * 100
        assert bd.task_score == pytest.approx(expected)

    def test_slow_task_reduces_score_after_5_tasks(self, tmp_scorer):
        scorer, store = tmp_scorer
        # insert 5 fast tasks to establish median of 200ms
        for _ in range(5):
            t = make_task(duration_ms=200)
            store.insert(t)
        # now a slow task: 2× median is 400ms, this is 900ms → speed=0
        slow = make_task(duration_ms=900)
        store.insert(slow)
        bd = scorer.score_task(slow, "search_agent", output_non_empty=True)
        assert bd.speed_signal == 0.0
        assert bd.speed_skipped is False

    def test_speed_skipped_below_5_tasks(self, tmp_scorer):
        scorer, store = tmp_scorer
        task = make_task(duration_ms=9999)
        store.insert(task)
        bd = scorer.score_task(task, "search_agent", output_non_empty=True)
        assert bd.speed_skipped is True
        assert bd.speed_signal == 1.0

    def test_rolling_average_over_20_tasks(self, tmp_scorer):
        scorer, store = tmp_scorer
        # 15 perfect + 5 totally failed
        for _ in range(15):
            store.insert(make_task(outcome=TaskOutcome.SUCCESS, correction=False))
        for _ in range(5):
            store.insert(make_task(outcome=TaskOutcome.FAILED, correction=True))

        score = scorer.rolling_score("search_agent")
        # perfect tasks score ≈ 100, failed ≈ 15 (speed skipped → 0.15×100)
        # weighted rolling → between 15 and 100
        assert 15 < score < 100

    def test_rolling_score_default_50_no_history(self, tmp_scorer):
        scorer, _ = tmp_scorer
        assert scorer.rolling_score("nonexistent_agent") == 50.0

    def test_rolling_window_capped_at_20(self, tmp_scorer):
        scorer, store = tmp_scorer
        # 25 tasks: first 5 failed, last 20 perfect
        for _ in range(5):
            store.insert(make_task(outcome=TaskOutcome.FAILED))
        for _ in range(20):
            store.insert(make_task(outcome=TaskOutcome.SUCCESS))
        score = scorer.rolling_score("search_agent")
        # window only covers last 20 (all perfect) → high score
        assert score > 80

    def test_score_breakdown_has_all_fields(self, tmp_scorer):
        scorer, store = tmp_scorer
        task = make_task()
        store.insert(task)
        bd = scorer.score_task(task, "search_agent")
        assert hasattr(bd, "success_signal")
        assert hasattr(bd, "no_correct_signal")
        assert hasattr(bd, "speed_signal")
        assert hasattr(bd, "non_empty_signal")
        assert hasattr(bd, "task_score")
        assert hasattr(bd, "rolling_score")

    def test_score_stays_in_0_100_range(self, tmp_scorer):
        scorer, store = tmp_scorer
        for _ in range(30):
            t = make_task(
                outcome=TaskOutcome.FAILED,
                correction=True,
                duration_ms=9999,
            )
            store.insert(t)
        score = scorer.rolling_score("search_agent")
        assert 0.0 <= score <= 100.0


# ══════════════════════════════════════════════════════════════════════════════
# 2. EvolutionEngine — thresholds and actions
# ══════════════════════════════════════════════════════════════════════════════

class TestEvolutionEngine:

    def _seed_registry(self, registry_path, *specs):
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps(list(specs), indent=2), encoding="utf-8")

    def _fill_failing_tasks(self, store, agent_name, n):
        """Insert n failing tasks for an agent."""
        for _ in range(n):
            store.insert(make_task(
                agents=[agent_name],
                outcome=TaskOutcome.FAILED,
                correction=True,
            ))

    # ── too new ───────────────────────────────────────────────────────────────

    def test_too_new_below_min_tasks(self, tmp_evo):
        engine, store, registry = tmp_evo
        self._seed_registry(registry, agent_spec())
        store.insert(make_task())   # only 1 task
        result = engine.evaluate("search_agent")
        assert result.outcome == EvolutionOutcome.TOO_NEW

    # ── stable ────────────────────────────────────────────────────────────────

    def test_stable_healthy_score(self, tmp_evo):
        engine, store, registry = tmp_evo
        self._seed_registry(registry, agent_spec())
        for _ in range(6):
            store.insert(make_task(outcome=TaskOutcome.SUCCESS))
        result = engine.evaluate("search_agent")
        assert result.outcome == EvolutionOutcome.STABLE

    # ── evolution ─────────────────────────────────────────────────────────────

    def test_evolution_triggered_at_threshold(self, tmp_evo):
        engine, store, registry = tmp_evo
        self._seed_registry(registry, agent_spec())
        self._fill_failing_tasks(store, "search_agent", _EVOLVE_MIN_TASKS)
        result = engine.evaluate("search_agent")
        assert result.outcome == EvolutionOutcome.EVOLVED

    def test_evolution_increments_version(self, tmp_evo):
        engine, store, registry = tmp_evo
        self._seed_registry(registry, agent_spec(version=1))
        self._fill_failing_tasks(store, "search_agent", _EVOLVE_MIN_TASKS)
        result = engine.evaluate("search_agent")
        assert result.version == 2

    def test_evolution_updates_system_prompt(self, tmp_evo):
        engine, store, registry = tmp_evo
        self._seed_registry(registry, agent_spec())
        self._fill_failing_tasks(store, "search_agent", _EVOLVE_MIN_TASKS)
        engine.evaluate("search_agent")
        spec = engine.get_agent_spec("search_agent")
        assert "[EVOLVED]" in spec["system_prompt"]

    def test_evolution_preserves_name_and_skills(self, tmp_evo):
        engine, store, registry = tmp_evo
        self._seed_registry(registry, agent_spec())
        self._fill_failing_tasks(store, "search_agent", _EVOLVE_MIN_TASKS)
        engine.evaluate("search_agent")
        spec = engine.get_agent_spec("search_agent")
        assert spec["name"] == "search_agent"
        assert spec["skills"] == ["web_search"]

    def test_evolution_with_llm_call(self, tmp_evo):
        engine, store, registry = tmp_evo
        engine._llm_call = lambda prompt: "Improved system prompt from LLM."
        self._seed_registry(registry, agent_spec())
        self._fill_failing_tasks(store, "search_agent", _EVOLVE_MIN_TASKS)
        result = engine.evaluate("search_agent")
        assert result.outcome == EvolutionOutcome.EVOLVED
        spec = engine.get_agent_spec("search_agent")
        assert "Improved system prompt" in spec["system_prompt"]

    # ── retirement ────────────────────────────────────────────────────────────

    def test_retirement_triggered_at_threshold(self, tmp_evo):
        engine, store, registry = tmp_evo
        self._seed_registry(registry, agent_spec())
        self._fill_failing_tasks(store, "search_agent", _RETIRE_MIN_TASKS)
        result = engine.evaluate("search_agent")
        assert result.outcome == EvolutionOutcome.RETIRED

    def test_retirement_sets_is_active_false(self, tmp_evo):
        engine, store, registry = tmp_evo
        self._seed_registry(registry, agent_spec())
        self._fill_failing_tasks(store, "search_agent", _RETIRE_MIN_TASKS)
        engine.evaluate("search_agent")
        spec = engine.get_agent_spec("search_agent")
        assert spec["is_active"] is False

    def test_retirement_sets_retired_at(self, tmp_evo):
        engine, store, registry = tmp_evo
        self._seed_registry(registry, agent_spec())
        self._fill_failing_tasks(store, "search_agent", _RETIRE_MIN_TASKS)
        engine.evaluate("search_agent")
        spec = engine.get_agent_spec("search_agent")
        assert "retired_at" in spec

    def test_retirement_checked_before_evolution(self, tmp_evo):
        """An agent below retire threshold should be retired, not evolved."""
        engine, store, registry = tmp_evo
        self._seed_registry(registry, agent_spec())
        self._fill_failing_tasks(store, "search_agent", _RETIRE_MIN_TASKS)
        result = engine.evaluate("search_agent")
        assert result.outcome == EvolutionOutcome.RETIRED

    # ── registry I/O ──────────────────────────────────────────────────────────

    def test_unknown_agent_returns_unknown_outcome(self, tmp_evo):
        engine, store, registry = tmp_evo
        self._seed_registry(registry)   # empty registry
        self._fill_failing_tasks(store, "ghost_agent", _EVOLVE_MIN_TASKS)
        result = engine.evaluate("ghost_agent")
        assert result.outcome == EvolutionOutcome.UNKNOWN

    def test_list_active_agents(self, tmp_evo):
        engine, _, registry = tmp_evo
        self._seed_registry(
            registry,
            agent_spec("file_agent", active=True),
            agent_spec("dead_agent",  active=False),
        )
        active = engine.list_active_agents()
        assert len(active) == 1
        assert active[0]["name"] == "file_agent"

    def test_registry_preserves_other_agents(self, tmp_evo):
        """Patching one agent must not corrupt others in the registry."""
        engine, store, registry = tmp_evo
        self._seed_registry(
            registry,
            agent_spec("search_agent"),
            agent_spec("file_agent"),
        )
        self._fill_failing_tasks(store, "search_agent", _EVOLVE_MIN_TASKS)
        engine.evaluate("search_agent")
        file_spec = engine.get_agent_spec("file_agent")
        assert file_spec is not None
        assert file_spec["name"] == "file_agent"

    def test_dict_format_registry_preserved(self, tmp_evo):
        """Registry stored as {"agents": [...]} keeps that wrapper."""
        engine, store, registry = tmp_evo
        data = {"agents": [agent_spec("search_agent")], "version": "1.0"}
        registry.parent.mkdir(parents=True, exist_ok=True)
        registry.write_text(json.dumps(data), encoding="utf-8")
        self._fill_failing_tasks(store, "search_agent", _EVOLVE_MIN_TASKS)
        engine.evaluate("search_agent")
        saved = json.loads(registry.read_text())
        assert "version" in saved   # wrapper preserved


# ══════════════════════════════════════════════════════════════════════════════
# 3. TaskLogger — scoring + evolution wired in
# ══════════════════════════════════════════════════════════════════════════════

class TestTaskLoggerWeek2:

    def _seed_registry(self, evo, registry_path, *specs):
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps(list(specs), indent=2), encoding="utf-8")

    def test_score_breakdown_accessible_after_task(self, tmp_logger):
        logger, store, evo = tmp_logger
        self._seed_registry(evo, evo._registry_path, agent_spec())

        with logger.task("search something") as ctx:
            ctx.add_agent("search_agent")
            ctx.set_outcome("success")

        assert "search_agent" in ctx.score_breakdowns
        bd = ctx.score_breakdowns["search_agent"]
        assert bd.task_score >= 0

    def test_evolution_result_accessible_after_task(self, tmp_logger):
        logger, store, evo = tmp_logger
        self._seed_registry(evo, evo._registry_path, agent_spec())

        with logger.task("search something") as ctx:
            ctx.add_agent("search_agent")
            ctx.set_outcome("success")

        assert "search_agent" in ctx.evolution_results

    def test_scoring_runs_for_each_agent(self, tmp_logger):
        logger, store, evo = tmp_logger
        self._seed_registry(
            evo, evo._registry_path,
            agent_spec("search_agent"),
            agent_spec("file_agent"),
        )

        with logger.task("search and save") as ctx:
            ctx.add_agent("search_agent")
            ctx.add_agent("file_agent")
            ctx.set_outcome("success")

        assert "search_agent" in ctx.score_breakdowns
        assert "file_agent"   in ctx.score_breakdowns

    def test_scoring_error_does_not_crash_brain(self, tmp_logger):
        """Scorer crash must be silently swallowed — Brain must complete."""
        logger, store, evo = tmp_logger
        # no registry seeded → evo will get UNKNOWN but must not raise

        completed = False
        with logger.task("test task") as ctx:
            ctx.add_agent("unknown_agent")
            ctx.set_outcome("success")
            completed = True

        assert completed
        assert store.count() == 1

    def test_failed_task_scores_lower_than_success(self, tmp_logger):
        logger, store, evo = tmp_logger
        self._seed_registry(evo, evo._registry_path, agent_spec())

        with logger.task("good task") as ctx:
            ctx.add_agent("search_agent")
            ctx.set_outcome("success")
        good_score = ctx.score_breakdowns["search_agent"].task_score

        with logger.task("bad task") as ctx:
            ctx.add_agent("search_agent")
            ctx.set_outcome("failed")
        bad_score = ctx.score_breakdowns["search_agent"].task_score

        assert good_score > bad_score


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import subprocess
    subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        check=False,
    )
