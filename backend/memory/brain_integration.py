"""
AEGIS Phase 2 — Brain memory integration
=========================================
Wraps the existing Brain orchestrator so every task is automatically:
  1. Timed (wall-clock ms)
  2. Logged to the task store on completion
  3. Used to trigger preference profile refresh (every 10 tasks)

How to integrate with your existing orchestrator
-------------------------------------------------
In backend/brain/orchestrator.py, wrap the main execute call:

    from backend.memory.brain_integration import TaskLogger

    class Brain:
        def __init__(self):
            ...
            self._task_logger = TaskLogger()

        def run(self, user_input: str) -> dict:
            with self._task_logger.task(user_input) as ctx:
                result = self._execute(user_input, ctx)
            return result

        def _execute(self, user_input: str, ctx) -> dict:
            # your existing logic — just call ctx.add_agent(), ctx.add_skill(), etc.
            ...
            ctx.set_outcome("success")
            ctx.set_category("search")
            return result

The context object (TaskContext) is the only thing you touch inside execute().
TaskLogger handles all storage and timing automatically.

Alternatively, use the function-based decorator:

    @task_logger.log_task
    def run(self, user_input: str) -> dict:
        ...
"""

from __future__ import annotations

import time
import traceback
from contextlib import contextmanager
from typing import Any, Generator

from .schemas import QualitySignals, TaskOutcome, TaskRecord
from .task_store import get_task_store
from .preference_engine import get_preference_engine
from .agent_scorer import AgentScorer, ScoreBreakdown, get_agent_scorer
from .evolution_engine import EvolutionEngine, EvolutionResult, get_evolution_engine
from .knowledge_store import KnowledgeStore, get_knowledge_store


class TaskContext:
    """
    Mutable task context passed into the Brain's execute method.
    Collects agents_used, skills_used, outcome, and category
    without the Brain needing to know anything about storage.
    """

    def __init__(self, raw_input: str) -> None:
        self.raw_input = raw_input
        self.agents_used: list[str] = []
        self.skills_used: list[str] = []
        self.outcome: TaskOutcome = TaskOutcome.FAILED
        self.task_category: str = ""
        self.error_log: str | None = None
        self.quality_signals: QualitySignals = QualitySignals()

        self._start_ms: float = time.monotonic() * 1000
        self._task_id: str = ""      # filled by TaskLogger after record creation
        self._score_breakdowns: dict = {}    # agent_name → ScoreBreakdown
        self._evolution_results: dict = {}   # agent_name → EvolutionResult

    # ── convenience setters ───────────────────────────────────────────────────

    def add_agent(self, name: str) -> None:
        """Call once per agent invoked, in order."""
        if name and name not in self.agents_used:
            self.agents_used.append(name)

    def add_skill(self, name: str) -> None:
        if name and name not in self.skills_used:
            self.skills_used.append(name)

    def set_outcome(self, outcome: str | TaskOutcome) -> None:
        self.outcome = TaskOutcome(outcome)

    def set_category(self, category: str) -> None:
        """file | search | code | browser | mixed"""
        self.task_category = category

    def set_error(self, error: str | Exception) -> None:
        self.error_log = str(error)
        self.outcome = TaskOutcome.FAILED

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def elapsed_ms(self) -> int:
        return int(time.monotonic() * 1000 - self._start_ms)

    @property
    def score_breakdowns(self) -> dict:
        """agent_name → ScoreBreakdown (available after context exits)."""
        return self._score_breakdowns

    @property
    def evolution_results(self) -> dict:
        """agent_name → EvolutionResult (available after context exits)."""
        return self._evolution_results


class TaskLogger:
    """
    Context-manager-based task logger.

    Typical usage
    -------------
        logger = TaskLogger()

        with logger.task("search for latest AI news") as ctx:
            ctx.add_agent("search_agent")
            ctx.add_skill("web_search")
            result = search_agent.run(...)
            ctx.set_outcome("success")
            ctx.set_category("search")
        # record is written here automatically

    If the body raises an exception, outcome is set to FAILED and
    the traceback is stored in error_log before the record is written.
    """

    def __init__(self) -> None:
        self._store     = get_task_store()
        self._engine    = get_preference_engine()
        self._scorer    = get_agent_scorer()
        self._evo       = get_evolution_engine()
        self._knowledge = get_knowledge_store()

    @contextmanager
    def task(self, raw_input: str) -> Generator[TaskContext, None, None]:
        ctx = TaskContext(raw_input)
        try:
            yield ctx
        except Exception as exc:
            ctx.set_error(traceback.format_exc())
            raise
        finally:
            self._commit(ctx)

    def _commit(self, ctx: TaskContext) -> None:
        record = TaskRecord(
            raw_input=ctx.raw_input,
            agents_used=ctx.agents_used,
            skills_used=ctx.skills_used,
            outcome=ctx.outcome,
            duration_ms=ctx.elapsed_ms,
            correction_issued=False,  # updated later via correction window
            quality_signals=ctx.quality_signals,
            error_log=ctx.error_log,
            task_category=ctx.task_category,
        )
        ctx._task_id = record.task_id
        self._store.insert(record)

        # ── score + evolve each agent used in this task ───────────────────
        output_non_empty = ctx.outcome == TaskOutcome.SUCCESS
        for agent_name in ctx.agents_used:
            try:
                breakdown = self._scorer.score_task(
                    record,
                    agent_name=agent_name,
                    output_non_empty=output_non_empty,
                )
                ctx._score_breakdowns[agent_name] = breakdown
                evo_result = self._evo.evaluate(agent_name)
                ctx._evolution_results[agent_name] = evo_result
            except Exception:
                pass   # scoring must never crash the Brain

        # ── preference profile refresh ────────────────────────────────────
        self._engine.maybe_refresh()

    # ── correction window helper ──────────────────────────────────────────────

    def flag_correction(self, task_id: str) -> bool:
        """
        Call this when the user says 'that was wrong' within 60 s.
        Returns True if the task was found and updated.
        """
        return self._store.mark_correction(task_id)

    # ── Brain context helpers ─────────────────────────────────────────────────

    def get_brain_context(self) -> str:
        """
        Returns a short user-profile string the Brain can prepend to its
        planning prompt for personalised routing.
        """
        return self._engine.get_profile().as_brain_context()

    def get_full_brain_context(self, task_input: str) -> str:
        """
        Returns combined context for the Brain planning prompt:
          1. User preference profile (from PreferenceEngine)
          2. Relevant knowledge base snippets (from KnowledgeStore)

        The Brain prepends this to its planning prompt so routing decisions
        are informed by past preferences and personal knowledge.

        Empty string sections are omitted so the prompt stays clean.
        """
        parts: list[str] = []

        profile_ctx = self._engine.get_profile().as_brain_context()
        if profile_ctx:
            parts.append(profile_ctx)

        try:
            knowledge_ctx = self._knowledge.search_as_context(task_input, n=3)
            if knowledge_ctx:
                parts.append(knowledge_ctx)
        except Exception:
            pass   # knowledge retrieval must never block the Brain

        return "\n\n".join(parts)


# ── module-level singleton ────────────────────────────────────────────────────
_logger: TaskLogger | None = None


def get_task_logger() -> TaskLogger:
    global _logger
    if _logger is None:
        _logger = TaskLogger()
    return _logger
