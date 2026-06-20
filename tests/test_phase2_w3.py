"""
AEGIS Phase 2 — Week 3 Acceptance Tests
Knowledge store (KAG retrieval) + multi-agent workflows

Run from project root:
    python -m pytest tests/test_phase2_w3.py -v

All tests are isolated — nothing touches real ChromaDB or task store.
"""

import sys
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.memory.knowledge_store import KnowledgeStore, SearchResult
from backend.memory.multi_agent import (
    WorkflowOrchestrator, WorkflowPlan, WorkflowStep,
    WorkflowResult, StepStatus, build_plan_from_routing,
)
from backend.memory.task_store import TaskStore
from backend.memory.preference_engine import PreferenceEngine
from backend.memory.agent_scorer import AgentScorer
from backend.memory.evolution_engine import EvolutionEngine
from backend.memory.brain_integration import TaskLogger
from backend.memory.schemas import TaskOutcome


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_knowledge(tmp_path):
    """KnowledgeStore backed by a temp ChromaDB dir — no Ollama required."""
    return KnowledgeStore(persist_dir=tmp_path / "chroma", use_ollama=False)


@pytest.fixture
def tmp_logger(tmp_path):
    store = TaskStore(path=tmp_path / "tasks.json")
    pref  = PreferenceEngine(store=store)
    pref._profile_path = tmp_path / "profile.json"
    scorer = AgentScorer(store=store)
    evo    = EvolutionEngine(llm_call=None, store=store, scorer=scorer,
                              registry_path=tmp_path / "agents.json")
    kb     = KnowledgeStore(persist_dir=tmp_path / "chroma", use_ollama=False)

    # Bypass TaskLogger.__init__ to avoid touching global singletons
    # (get_knowledge_store() would fail without Ollama/ONNX in this environment)
    logger = object.__new__(TaskLogger)
    logger._store     = store
    logger._engine    = pref
    logger._scorer    = scorer
    logger._evo       = evo
    logger._knowledge = kb
    return logger, kb


# ══════════════════════════════════════════════════════════════════════════════
# 1. KnowledgeStore — ingestion
# ══════════════════════════════════════════════════════════════════════════════

class TestKnowledgeStoreIngestion:

    def test_ingest_text_increases_count(self, tmp_knowledge):
        assert tmp_knowledge.count() == 0
        tmp_knowledge.ingest_text("AEGIS is an AI orchestration platform.", source="test")
        assert tmp_knowledge.count() > 0

    def test_ingest_returns_doc_ids(self, tmp_knowledge):
        ids = tmp_knowledge.ingest_text("Some content here.", source="test")
        assert isinstance(ids, list)
        assert len(ids) >= 1
        assert all(isinstance(i, str) for i in ids)

    def test_ingest_multiple_sources(self, tmp_knowledge):
        tmp_knowledge.ingest_text("Note about Python.", source="notes/python.md")
        tmp_knowledge.ingest_text("Note about LangGraph.", source="notes/langgraph.md")
        sources = tmp_knowledge.list_sources()
        assert "notes/python.md"    in sources
        assert "notes/langgraph.md" in sources

    def test_ingest_long_text_creates_multiple_chunks(self, tmp_knowledge):
        long_text = " ".join([f"word{i}" for i in range(1000)])
        ids = tmp_knowledge.ingest_text(long_text, source="long_doc")
        assert len(ids) > 1   # should be chunked

    def test_ingest_empty_text_returns_empty_list(self, tmp_knowledge):
        ids = tmp_knowledge.ingest_text("", source="empty")
        assert ids == []

    def test_ingest_txt_file(self, tmp_knowledge, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("AEGIS uses ChromaDB for vector search.", encoding="utf-8")
        ids = tmp_knowledge.ingest_file(f)
        assert len(ids) >= 1

    def test_ingest_md_file(self, tmp_knowledge, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# AEGIS\nAI orchestration platform.", encoding="utf-8")
        ids = tmp_knowledge.ingest_file(f)
        assert len(ids) >= 1

    def test_ingest_missing_file_raises(self, tmp_knowledge):
        with pytest.raises(FileNotFoundError):
            tmp_knowledge.ingest_file("/nonexistent/path/file.txt")

    def test_ingest_unsupported_type_raises(self, tmp_knowledge, tmp_path):
        f = tmp_path / "data.xyz"
        f.write_text("some data")
        with pytest.raises(ValueError, match="Unsupported file type"):
            tmp_knowledge.ingest_file(f)

    def test_upsert_same_source_does_not_duplicate(self, tmp_knowledge):
        tmp_knowledge.ingest_text("Version 1 content.", source="doc")
        count_after_first = tmp_knowledge.count()
        tmp_knowledge.ingest_text("Version 2 content.", source="doc")
        # upsert by stable ID — count should stay the same
        assert tmp_knowledge.count() == count_after_first

    def test_ingest_task_summary(self, tmp_knowledge):
        doc_id = tmp_knowledge.ingest_task_summary(
            task_id="abc-123",
            summary="Searched for AI news using search_agent.",
            agents=["search_agent"],
        )
        assert doc_id == "task:abc-123"
        assert tmp_knowledge.count() >= 1


# ══════════════════════════════════════════════════════════════════════════════
# 2. KnowledgeStore — retrieval
# ══════════════════════════════════════════════════════════════════════════════

class TestKnowledgeStoreRetrieval:

    def _seed(self, store):
        store.ingest_text(
            "ChromaDB is a vector database for semantic search.",
            source="tech/chroma.md",
        )
        store.ingest_text(
            "LangGraph is used for building multi-agent workflows in AEGIS.",
            source="tech/langgraph.md",
        )
        store.ingest_text(
            "Python is the primary language for AEGIS backend development.",
            source="tech/python.md",
        )

    def test_search_returns_results(self, tmp_knowledge):
        self._seed(tmp_knowledge)
        results = tmp_knowledge.search("vector database", n=2)
        assert len(results) >= 1
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_result_has_required_fields(self, tmp_knowledge):
        self._seed(tmp_knowledge)
        results = tmp_knowledge.search("chromadb", n=1)
        r = results[0]
        assert isinstance(r.text, str) and r.text
        assert isinstance(r.source, str)
        assert isinstance(r.score, float)
        assert isinstance(r.doc_id, str)

    def test_search_respects_n_limit(self, tmp_knowledge):
        self._seed(tmp_knowledge)
        results = tmp_knowledge.search("AEGIS", n=2)
        assert len(results) <= 2

    def test_search_empty_store_returns_empty(self, tmp_knowledge):
        results = tmp_knowledge.search("anything")
        assert results == []

    def test_search_as_context_returns_string(self, tmp_knowledge):
        self._seed(tmp_knowledge)
        ctx = tmp_knowledge.search_as_context("vector database")
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_search_as_context_empty_when_no_docs(self, tmp_knowledge):
        ctx = tmp_knowledge.search_as_context("anything")
        assert ctx == ""

    def test_search_as_context_respects_max_chars(self, tmp_knowledge):
        # Ingest a lot of text
        big = " ".join([f"word{i}" for i in range(2000)])
        tmp_knowledge.ingest_text(big, source="big_doc")
        ctx = tmp_knowledge.search_as_context("word500", n=5, max_chars=200)
        assert len(ctx) <= 500   # header + a bit of content, not the full 2000 words

    def test_delete_source_removes_chunks(self, tmp_knowledge):
        self._seed(tmp_knowledge)
        before = tmp_knowledge.count()
        deleted = tmp_knowledge.delete_source("tech/chroma.md")
        assert deleted >= 1
        assert tmp_knowledge.count() < before

    def test_list_sources_accurate(self, tmp_knowledge):
        self._seed(tmp_knowledge)
        sources = tmp_knowledge.list_sources()
        assert "tech/chroma.md"     in sources
        assert "tech/langgraph.md"  in sources
        assert "tech/python.md"     in sources

    def test_reset_clears_all_docs(self, tmp_knowledge):
        self._seed(tmp_knowledge)
        assert tmp_knowledge.count() > 0
        tmp_knowledge.reset()
        assert tmp_knowledge.count() == 0


# ══════════════════════════════════════════════════════════════════════════════
# 3. MultiAgent — workflow execution
# ══════════════════════════════════════════════════════════════════════════════

def _make_runner(responses: dict[str, dict]):
    """
    Build a mock agent runner.
    responses: {agent_name: result_dict}
    Agents not in dict return a generic success.
    """
    def runner(agent_name: str, goal: str, context: dict) -> dict:
        return responses.get(agent_name, {"success": True, "message": f"{agent_name} done"})
    return runner


class TestWorkflowOrchestrator:

    def test_single_step_success(self):
        runner = _make_runner({"search_agent": {"success": True, "message": "Found results"}})
        orch = WorkflowOrchestrator(agent_runner=runner)
        plan = WorkflowPlan(steps=[WorkflowStep(agent="search_agent", goal="find AI news")])
        result = orch.run(plan)
        assert result.success is True
        assert "search_agent" in result.agents_used

    def test_parallel_independent_steps(self):
        """Wave 1 steps with no dependencies run in parallel."""
        call_times: list[float] = []

        def slow_runner(agent, goal, ctx):
            call_times.append(time.monotonic())
            time.sleep(0.05)
            return {"success": True, "message": f"{agent} done"}

        orch = WorkflowOrchestrator(agent_runner=slow_runner, max_workers=4)
        plan = WorkflowPlan(steps=[
            WorkflowStep(agent="search_agent", goal="step 1"),
            WorkflowStep(agent="file_agent",   goal="step 2"),
            WorkflowStep(agent="code_agent",   goal="step 3"),
        ])
        start = time.monotonic()
        result = orch.run(plan)
        elapsed = time.monotonic() - start

        assert result.success is True
        # parallel execution: 3×50ms serial = 150ms, parallel should be ~50ms
        assert elapsed < 0.13   # generous bound for CI/test runners

    def test_sequential_dependency_respected(self):
        """Wave 2 step only runs after its dependency completes."""
        execution_order: list[str] = []

        def ordered_runner(agent, goal, ctx):
            execution_order.append(agent)
            return {"success": True, "message": f"{agent} done"}

        orch = WorkflowOrchestrator(agent_runner=ordered_runner)
        plan = WorkflowPlan(steps=[
            WorkflowStep(agent="search_agent", goal="search first"),        # step 0
            WorkflowStep(agent="file_agent",   goal="save results",
                         depends_on=[0]),                                    # step 1
        ])
        orch.run(plan)
        assert execution_order.index("search_agent") < execution_order.index("file_agent")

    def test_dependency_result_injected_into_context(self):
        """Wave 2 step receives wave 1 results in its context."""
        received_context: list[dict] = []

        def capturing_runner(agent, goal, ctx):
            received_context.append(dict(ctx))
            return {"success": True, "message": f"{agent} result"}

        orch = WorkflowOrchestrator(agent_runner=capturing_runner)
        plan = WorkflowPlan(steps=[
            WorkflowStep(agent="search_agent", goal="step 0"),
            WorkflowStep(agent="file_agent",   goal="step 1", depends_on=[0]),
        ])
        orch.run(plan)

        # file_agent (wave 2) should have prior_results in its context
        file_ctx = received_context[1]
        assert "prior_results" in file_ctx
        assert "step_0_result" in file_ctx["prior_results"]

    def test_failed_step_marks_result_failed(self):
        runner = _make_runner({"search_agent": {"success": False, "error": "network error"}})
        orch = WorkflowOrchestrator(agent_runner=runner)
        plan = WorkflowPlan(steps=[WorkflowStep(agent="search_agent", goal="search")])
        result = orch.run(plan)
        assert result.success is False
        assert 0 in result.failed_steps

    def test_failed_dependency_skips_downstream(self):
        runner = _make_runner({
            "search_agent": {"success": False, "error": "search failed"},
            "file_agent":   {"success": True,  "message": "saved"},
        })
        orch = WorkflowOrchestrator(agent_runner=runner)
        plan = WorkflowPlan(steps=[
            WorkflowStep(agent="search_agent", goal="search"),
            WorkflowStep(agent="file_agent",   goal="save", depends_on=[0]),
        ])
        result = orch.run(plan)
        assert plan.steps[1].status == StepStatus.SKIPPED

    def test_agent_exception_does_not_crash_orchestrator(self):
        def exploding_runner(agent, goal, ctx):
            raise RuntimeError("agent exploded")

        orch = WorkflowOrchestrator(agent_runner=exploding_runner)
        plan = WorkflowPlan(steps=[WorkflowStep(agent="broken_agent", goal="do it")])
        result = orch.run(plan)
        assert result.success is False
        assert plan.steps[0].status == StepStatus.FAILED

    def test_empty_plan_returns_failure(self):
        orch = WorkflowOrchestrator(agent_runner=lambda *a: {})
        result = orch.run(WorkflowPlan(steps=[]))
        assert result.success is False

    def test_combined_output_contains_agent_messages(self):
        runner = _make_runner({
            "search_agent": {"success": True, "message": "Found 5 articles"},
            "file_agent":   {"success": True, "message": "Saved to notes.txt"},
        })
        orch = WorkflowOrchestrator(agent_runner=runner)
        plan = WorkflowPlan(steps=[
            WorkflowStep(agent="search_agent", goal="search"),
            WorkflowStep(agent="file_agent",   goal="save"),
        ])
        result = orch.run(plan)
        assert "Found 5 articles" in result.combined_output
        assert "Saved to notes.txt" in result.combined_output

    def test_custom_synthesiser_called(self):
        synthesiser_called = []

        def custom_synth(steps, task_input):
            synthesiser_called.append(True)
            return "Custom synthesis result"

        runner = _make_runner({"search_agent": {"success": True, "message": "ok"}})
        orch = WorkflowOrchestrator(agent_runner=runner, synthesiser=custom_synth)
        plan = WorkflowPlan(steps=[WorkflowStep(agent="search_agent", goal="search")])
        result = orch.run(plan)
        assert synthesiser_called
        assert result.combined_output == "Custom synthesis result"

    def test_duration_recorded(self):
        def slow_runner(agent, goal, ctx):
            time.sleep(0.03)
            return {"success": True, "message": "done"}

        orch = WorkflowOrchestrator(agent_runner=slow_runner)
        plan = WorkflowPlan(steps=[WorkflowStep(agent="search_agent", goal="search")])
        result = orch.run(plan)
        assert result.total_ms >= 25

    def test_agents_used_list_correct(self):
        runner = _make_runner({
            "search_agent": {"success": True, "message": "ok"},
            "code_agent":   {"success": True, "message": "ok"},
        })
        orch = WorkflowOrchestrator(agent_runner=runner)
        plan = WorkflowPlan(steps=[
            WorkflowStep(agent="search_agent", goal="step1"),
            WorkflowStep(agent="code_agent",   goal="step2"),
        ])
        result = orch.run(plan)
        assert set(result.agents_used) == {"search_agent", "code_agent"}


# ══════════════════════════════════════════════════════════════════════════════
# 4. build_plan_from_routing helper
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildPlanFromRouting:

    def test_non_multi_agent_returns_none(self):
        routing = {"agent": "search_agent", "goal": "search"}
        assert build_plan_from_routing(routing, "find news") is None

    def test_multi_agent_returns_plan(self):
        routing = {
            "multi_agent": True,
            "steps": [
                {"agent": "search_agent", "goal": "search AI news"},
                {"agent": "file_agent",   "goal": "save results", "depends_on": [0]},
            ]
        }
        plan = build_plan_from_routing(routing, "find and save AI news")
        assert plan is not None
        assert len(plan.steps) == 2
        assert plan.steps[0].agent == "search_agent"
        assert plan.steps[1].depends_on == [0]

    def test_empty_steps_returns_none(self):
        routing = {"multi_agent": True, "steps": []}
        assert build_plan_from_routing(routing, "task") is None


# ══════════════════════════════════════════════════════════════════════════════
# 5. Brain integration — get_full_brain_context
# ══════════════════════════════════════════════════════════════════════════════

class TestBrainContextWithKnowledge:

    def test_full_context_returns_string(self, tmp_logger):
        logger, kb = tmp_logger
        ctx_str = logger.get_full_brain_context("find AI news")
        assert isinstance(ctx_str, str)

    def test_full_context_includes_knowledge_when_relevant(self, tmp_logger):
        logger, kb = tmp_logger
        kb.ingest_text(
            "AEGIS uses LangGraph for its Brain orchestrator.",
            source="docs/architecture.md",
        )
        ctx_str = logger.get_full_brain_context("how does the Brain work?")
        assert "Knowledge base" in ctx_str or "LangGraph" in ctx_str

    def test_full_context_empty_when_no_data(self, tmp_logger):
        logger, kb = tmp_logger
        # No tasks logged, no knowledge ingested, profile below minimum
        ctx_str = logger.get_full_brain_context("do something")
        # Should return empty string or minimal content — must not raise
        assert isinstance(ctx_str, str)

    def test_knowledge_retrieval_failure_does_not_crash(self, tmp_logger):
        logger, kb = tmp_logger
        # Simulate broken knowledge store
        logger._knowledge = MagicMock()
        logger._knowledge.search_as_context.side_effect = RuntimeError("chroma down")
        # Must not raise
        ctx_str = logger.get_full_brain_context("any task")
        assert isinstance(ctx_str, str)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import subprocess
    subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        check=False,
    )
