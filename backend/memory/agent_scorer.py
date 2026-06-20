"""
AEGIS Phase 2 Week 2 — Agent Scorer
=====================================
Implements the composite performance score defined in SOW v2 section 5.1.

Signal          Weight  How measured
──────────────  ──────  ────────────────────────────────────────────────────
Success         40%     outcome == success → 1, else 0
No correction   25%     correction_issued == False → 1, else 0
Speed           15%     duration_ms ≤ 2× agent median → 1, else 0
                        (skipped for first 5 tasks → treated as 1.0)
Non-empty       20%     agent produced actual output (not empty/null) → 1

Score = weighted sum × 100, rolling average over last 20 tasks per agent.
Range: 0–100. New agents start at 50 (set by Agent Factory at registration).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .schemas import TaskOutcome, TaskRecord
from .task_store import TaskStore, get_task_store

if TYPE_CHECKING:
    pass

# ── weights (must sum to 1.0) ──────────────────────────────────────────────
_W_SUCCESS    = 0.40
_W_NO_CORRECT = 0.25
_W_SPEED      = 0.15
_W_NON_EMPTY  = 0.20

_ROLLING_WINDOW   = 20   # tasks per agent
_SPEED_MIN_TASKS  = 5    # skip speed signal below this count


@dataclass
class ScoreBreakdown:
    """Full score breakdown for one task evaluation."""
    agent_name: str
    task_id: str

    # raw signals (0.0 or 1.0)
    success_signal:     float = 0.0
    no_correct_signal:  float = 0.0
    speed_signal:       float = 1.0   # default pass when skipped
    non_empty_signal:   float = 0.0

    # per-task weighted score (0–100)
    task_score: float = 0.0

    # rolling average after this task (0–100)
    rolling_score: float = 0.0

    # diagnostics
    speed_skipped: bool = False
    tasks_evaluated: int = 0

    def as_dict(self) -> dict:
        return {
            "agent_name":       self.agent_name,
            "task_id":          self.task_id,
            "success_signal":   self.success_signal,
            "no_correct_signal":self.no_correct_signal,
            "speed_signal":     self.speed_signal,
            "non_empty_signal": self.non_empty_signal,
            "task_score":       round(self.task_score, 2),
            "rolling_score":    round(self.rolling_score, 2),
            "speed_skipped":    self.speed_skipped,
            "tasks_evaluated":  self.tasks_evaluated,
        }


class AgentScorer:
    """
    Calculates and returns agent performance scores.

    Does NOT write to the agent registry itself — that is the
    EvolutionEngine's responsibility. AgentScorer only reads
    task history and computes numbers.

    Usage
    -----
        scorer = AgentScorer()
        breakdown = scorer.score_task(task_record, agent_name="search_agent")
        print(breakdown.rolling_score)   # e.g. 72.5
    """

    def __init__(self, store: TaskStore | None = None) -> None:
        self._store = store or get_task_store()

    # ── public API ────────────────────────────────────────────────────────────

    def score_task(
        self,
        task: TaskRecord,
        agent_name: str,
        output_non_empty: bool = True,
    ) -> ScoreBreakdown:
        """
        Evaluate a single completed task for one agent and return the
        full score breakdown including updated rolling average.

        Parameters
        ----------
        task:             The completed TaskRecord.
        agent_name:       Which agent to score (task may have used multiple).
        output_non_empty: Caller tells us whether the agent produced output.
                          Defaults True; set False when result was empty/null.
        """
        bd = ScoreBreakdown(agent_name=agent_name, task_id=task.task_id)

        # ── signal 1: success ────────────────────────────────────────────────
        bd.success_signal = 1.0 if task.outcome == TaskOutcome.SUCCESS else 0.0

        # ── signal 2: no correction ──────────────────────────────────────────
        bd.no_correct_signal = 0.0 if task.correction_issued else 1.0

        # ── signal 3: speed ──────────────────────────────────────────────────
        agent_tasks = self._store.by_agent(agent_name)
        prior_tasks = [t for t in agent_tasks if t.task_id != task.task_id]

        if len(prior_tasks) < _SPEED_MIN_TASKS:
            bd.speed_signal  = 1.0   # not enough history → pass
            bd.speed_skipped = True
        else:
            median = self._store.agent_median_duration(agent_name)
            if median and median > 0 and task.duration_ms > 0:
                bd.speed_signal = 1.0 if task.duration_ms <= 2 * median else 0.0
            else:
                bd.speed_signal  = 1.0
                bd.speed_skipped = True

        # ── signal 4: non-empty output ───────────────────────────────────────
        bd.non_empty_signal = 1.0 if output_non_empty else 0.0

        # ── per-task weighted score ──────────────────────────────────────────
        bd.task_score = (
            bd.success_signal    * _W_SUCCESS    +
            bd.no_correct_signal * _W_NO_CORRECT +
            bd.speed_signal      * _W_SPEED      +
            bd.non_empty_signal  * _W_NON_EMPTY
        ) * 100

        # ── rolling average over last _ROLLING_WINDOW tasks ──────────────────
        bd.rolling_score = self._compute_rolling(agent_name, bd.task_score)
        bd.tasks_evaluated = len(agent_tasks)

        return bd

    def rolling_score(self, agent_name: str) -> float:
        """
        Current rolling score for an agent based on existing task history.
        Returns 50.0 (new-agent default) when no history exists.
        """
        tasks = self._store.by_agent(agent_name)
        if not tasks:
            return 50.0

        scores = [self._task_score_from_record(t, agent_name) for t in tasks]
        window = scores[-_ROLLING_WINDOW:]
        return round(sum(window) / len(window), 2)

    # ── private helpers ───────────────────────────────────────────────────────

    def _compute_rolling(self, agent_name: str, latest_score: float) -> float:
        """Rolling average including the latest (not-yet-stored) score."""
        tasks = self._store.by_agent(agent_name)
        historical = [self._task_score_from_record(t, agent_name) for t in tasks]
        window = (historical + [latest_score])[-_ROLLING_WINDOW:]
        return round(sum(window) / len(window), 2)

    def _task_score_from_record(self, task: TaskRecord, agent_name: str) -> float:
        """Re-derive a task score from a stored record (no output signal available)."""
        # success
        s1 = 1.0 if task.outcome == TaskOutcome.SUCCESS else 0.0
        # no correction
        s2 = 0.0 if task.correction_issued else 1.0
        # speed — can't know median without recursion; use 1.0 as safe default
        s3 = 1.0
        # non-empty — assume non-empty for stored successful tasks
        s4 = 1.0 if task.outcome == TaskOutcome.SUCCESS else 0.0

        return (s1 * _W_SUCCESS + s2 * _W_NO_CORRECT + s3 * _W_SPEED + s4 * _W_NON_EMPTY) * 100


# ── module-level singleton ────────────────────────────────────────────────────
_scorer: AgentScorer | None = None


def get_agent_scorer() -> AgentScorer:
    global _scorer
    if _scorer is None:
        _scorer = AgentScorer()
    return _scorer
