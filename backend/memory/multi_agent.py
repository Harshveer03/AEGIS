"""
AEGIS Phase 2 Week 3 — Multi-Agent Workflow Orchestrator
==========================================================
Implements SOW Phase 2: "multi-agent workflows — parallel agent collaboration
on complex tasks."

Design
------
The Brain decomposes a complex task into sub-tasks. Each sub-task is assigned
to one agent. Sub-tasks that don't depend on each other run in parallel via
ThreadPoolExecutor. The Brain then synthesises the combined results.

This module is intentionally decoupled from the Brain's LangGraph loop —
it is a utility the Brain calls from _node_execute when routing detects a
multi-agent pattern.

Dependency model
----------------
Sub-tasks carry an optional `depends_on` list of step indices. Steps with no
dependencies run in parallel in the first wave; steps that depend on earlier
results run after those complete. Simple two-wave model covers the vast
majority of real cases without the complexity of a full DAG scheduler.

Usage
-----
    orchestrator = WorkflowOrchestrator(agent_runner=my_runner)

    plan = WorkflowPlan(steps=[
        WorkflowStep(agent="search_agent", goal="find latest AI news"),
        WorkflowStep(agent="file_agent",   goal="save results to notes.txt",
                     depends_on=[0]),
    ])

    result = orchestrator.run(plan)
    print(result.combined_output)
    print(result.success)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class StepStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    SUCCESS  = "success"
    FAILED   = "failed"
    SKIPPED  = "skipped"   # dependency failed → skip this step


@dataclass
class WorkflowStep:
    """Single step in a multi-agent workflow."""
    agent:      str                         # agent name to invoke
    goal:       str                         # natural-language goal for this step
    context:    dict = field(default_factory=dict)   # extra context for the agent
    depends_on: list[int] = field(default_factory=list)  # step indices (0-based)

    # filled in after execution
    status:      StepStatus = StepStatus.PENDING
    result:      dict       = field(default_factory=dict)
    duration_ms: int        = 0
    error:       str | None = None


@dataclass
class WorkflowPlan:
    """
    Complete multi-agent workflow plan produced by the Brain.

    Parameters
    ----------
    steps:       Ordered list of WorkflowStep objects.
    task_input:  Original user input (for context injection).
    """
    steps:      list[WorkflowStep]
    task_input: str = ""


@dataclass
class WorkflowResult:
    """Outcome of a completed workflow."""
    steps:          list[WorkflowStep]
    success:        bool
    combined_output: str
    agents_used:    list[str]
    total_ms:       int
    failed_steps:   list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "success":        self.success,
            "combined_output": self.combined_output,
            "agents_used":    self.agents_used,
            "total_ms":       self.total_ms,
            "failed_steps":   self.failed_steps,
            "steps": [
                {
                    "agent":       s.agent,
                    "goal":        s.goal,
                    "status":      s.status,
                    "duration_ms": s.duration_ms,
                    "error":       s.error,
                }
                for s in self.steps
            ],
        }


# ── Agent runner protocol ──────────────────────────────────────────────────────
# The Brain passes its own _run_agent method as the runner so WorkflowOrchestrator
# stays decoupled from the Brain implementation.
#
# Signature: runner(agent_name: str, goal: str, context: dict) -> dict
# Returns:   {"success": bool, "message": str, "data": dict, ...}

AgentRunner = Callable[[str, str, dict], dict]


class WorkflowOrchestrator:
    """
    Runs a WorkflowPlan, executing independent steps in parallel.

    Parameters
    ----------
    agent_runner: Callable with signature (agent_name, goal, context) -> dict.
                  The Brain passes self._run_agent here.
    max_workers:  Thread pool size for parallel execution (default 4).
    synthesiser:  Optional callable(steps, task_input) -> str for LLM synthesis.
                  If None, a simple rule-based summary is produced.
    """

    def __init__(
        self,
        agent_runner: AgentRunner,
        max_workers: int = 4,
        synthesiser: Callable[[list[WorkflowStep], str], str] | None = None,
    ) -> None:
        self._runner      = agent_runner
        self._max_workers = max_workers
        self._synthesiser = synthesiser

    # ── public API ─────────────────────────────────────────────────────────────

    def run(self, plan: WorkflowPlan) -> WorkflowResult:
        """Execute the workflow plan and return the combined result."""
        start = time.monotonic()

        if not plan.steps:
            return WorkflowResult(
                steps=[], success=False,
                combined_output="No steps in workflow plan.",
                agents_used=[], total_ms=0,
            )

        # Separate into waves: independent steps first, dependent steps after
        wave1, wave2 = self._partition_waves(plan.steps)

        # Execute wave 1 in parallel
        self._run_wave(wave1)

        # Inject wave 1 results as context into wave 2 steps;
        # steps whose dependency failed are marked SKIPPED here and
        # excluded from the wave so _run_wave never touches them.
        self._inject_prior_results(plan.steps, wave2)
        runnable_wave2 = [s for s in wave2 if s.status != StepStatus.SKIPPED]

        # Execute wave 2 (may be empty)
        self._run_wave(runnable_wave2)

        total_ms = int((time.monotonic() - start) * 1000)

        # Synthesise
        combined = self._synthesise(plan.steps, plan.task_input)
        failed   = [i for i, s in enumerate(plan.steps) if s.status == StepStatus.FAILED]
        agents   = list(dict.fromkeys(s.agent for s in plan.steps))  # ordered unique

        return WorkflowResult(
            steps=plan.steps,
            success=len(failed) == 0,
            combined_output=combined,
            agents_used=agents,
            total_ms=total_ms,
            failed_steps=failed,
        )

    # ── wave execution ─────────────────────────────────────────────────────────

    def _partition_waves(
        self, steps: list[WorkflowStep]
    ) -> tuple[list[WorkflowStep], list[WorkflowStep]]:
        wave1 = [s for s in steps if not s.depends_on]
        wave2 = [s for s in steps if s.depends_on]
        return wave1, wave2

    def _run_wave(self, steps: list[WorkflowStep]) -> None:
        if not steps:
            return

        if len(steps) == 1:
            self._execute_step(steps[0])
            return

        with ThreadPoolExecutor(max_workers=min(self._max_workers, len(steps))) as pool:
            futures = {pool.submit(self._execute_step, step): step for step in steps}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    step = futures[future]
                    step.status = StepStatus.FAILED
                    step.error  = str(exc)

    def _execute_step(self, step: WorkflowStep) -> None:
        """Run a single step, updating its status and result in place."""
        step.status = StepStatus.RUNNING
        t0 = time.monotonic()
        try:
            result = self._runner(step.agent, step.goal, step.context)
            step.result      = result
            step.duration_ms = int((time.monotonic() - t0) * 1000)
            step.status      = StepStatus.SUCCESS if result.get("success") else StepStatus.FAILED
            if not result.get("success"):
                step.error = result.get("error", "Agent returned failure")
        except Exception as exc:
            step.duration_ms = int((time.monotonic() - t0) * 1000)
            step.status      = StepStatus.FAILED
            step.error       = str(exc)
            step.result      = {"success": False, "error": str(exc)}

    def _inject_prior_results(
        self,
        all_steps: list[WorkflowStep],
        wave2: list[WorkflowStep],
    ) -> None:
        """
        For each wave-2 step, inject results of its dependencies into context.
        Skips the step if any dependency failed.
        """
        for step in wave2:
            dep_results = {}
            skip = False
            for dep_idx in step.depends_on:
                dep = all_steps[dep_idx]
                if dep.status == StepStatus.FAILED:
                    step.status = StepStatus.SKIPPED
                    step.error  = f"Skipped: dependency step {dep_idx} failed."
                    skip = True
                    break
                dep_results[f"step_{dep_idx}_result"] = dep.result.get("message", "")
            if not skip:
                step.context.update({"prior_results": dep_results})

    # ── synthesis ──────────────────────────────────────────────────────────────

    def _synthesise(self, steps: list[WorkflowStep], task_input: str) -> str:
        if self._synthesiser:
            try:
                return self._synthesiser(steps, task_input)
            except Exception:
                pass   # fall through to rule-based

        return self._rule_based_summary(steps, task_input)

    @staticmethod
    def _rule_based_summary(steps: list[WorkflowStep], task_input: str) -> str:
        """Simple deterministic summary when no LLM synthesiser is provided."""
        lines: list[str] = []
        succeeded = [s for s in steps if s.status == StepStatus.SUCCESS]
        failed    = [s for s in steps if s.status == StepStatus.FAILED]
        skipped   = [s for s in steps if s.status == StepStatus.SKIPPED]

        if succeeded:
            for s in succeeded:
                msg = s.result.get("message", "")
                if msg:
                    lines.append(f"[{s.agent}] {msg}")

        if failed:
            lines.append("")
            for s in failed:
                lines.append(f"[{s.agent}] Failed: {s.error or 'unknown error'}")

        if skipped:
            for s in skipped:
                lines.append(f"[{s.agent}] Skipped: {s.error or 'dependency failed'}")

        if not lines:
            return "Workflow completed with no output."

        return "\n".join(lines)


# ── Brain-side helpers ─────────────────────────────────────────────────────────

def build_plan_from_routing(routing: dict, task_input: str) -> WorkflowPlan | None:
    """
    Convert the Brain's routing dict into a WorkflowPlan when the routing
    signals a multi-agent task.

    The Brain's router sets routing["multi_agent"] = True and provides
    routing["steps"] = [{"agent": ..., "goal": ..., "depends_on": [...]}, ...]

    Returns None if routing is not a multi-agent plan.
    """
    if not routing.get("multi_agent"):
        return None

    steps_raw = routing.get("steps", [])
    if not steps_raw:
        return None

    steps = [
        WorkflowStep(
            agent=s["agent"],
            goal=s.get("goal", task_input),
            context=s.get("context", {}),
            depends_on=s.get("depends_on", []),
        )
        for s in steps_raw
    ]

    return WorkflowPlan(steps=steps, task_input=task_input)
