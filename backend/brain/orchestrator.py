import json
import uuid
import time
from typing import TypedDict, Optional, Any
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from config import settings
from backend.brain.session import SessionContext
from backend.brain.router import AgentRouter
from backend.brain.planner import TaskPlanner
from backend.brain.prompts import (
    BRAIN_CLARIFY_PROMPT,
    BRAIN_SYNTHESISE_PROMPT,
    BRAIN_NO_AGENT_PROMPT,
)
from backend.factory.skill_factory import get_skill_factory
from backend.factory.agent_factory import get_agent_factory
from backend.models.task import Task, TaskOutcome, TaskStatus, FailureType
from backend.registry.agent_registry import agent_registry
from backend.registry.skill_registry import skill_registry
from backend.agents.file_agent import create_file_agent
from backend.agents.search_agent import create_search_agent
from backend.agents.browser_agent import create_browser_agent
from backend.agents.code_agent import create_code_agent
from backend.memory.task_store import task_store
from backend.memory.brain_integration import get_task_logger
from backend.memory.multi_agent import WorkflowOrchestrator, build_plan_from_routing
import structlog

log = structlog.get_logger()


# ── LangGraph state ──────────────────────────────────────────────
class BrainState(TypedDict):
    task: Task
    session: SessionContext
    routing: dict
    agent_result: dict
    final_response: str
    needs_clarification: bool
    clarification: dict
    needs_factory: bool
    error: Optional[str]


class Brain:

    def __init__(self):
        self.llm = self._init_llm()
        self.router = AgentRouter()
        self.planner = TaskPlanner()
        self.graph = self._build_graph()
        self._agent_map = { ... }
        self.skill_factory = get_skill_factory(self._llm_call)
        self.agent_factory = get_agent_factory(self._llm_call)
        self._task_logger = get_task_logger()   # ← add this line
        log.info("Brain initialised", provider=settings.llm_provider)

    # ── LLM initialisation ───────────────────────────────────────
    def _init_llm(self):
        if settings.is_ollama:
            return OllamaLLM(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=0.1,
            )
        raise NotImplementedError("Only Ollama supported in Phase 1")

    def _llm_call(self, prompt: str) -> str:
        return self.llm.invoke(prompt)

    # ── LangGraph nodes ──────────────────────────────────────────
    def _node_clarify(self, state: BrainState) -> BrainState:
        task = state["task"]
        log.info("Brain: clarify node", task_id=task.task_id)
        task.status = TaskStatus.PLANNING

        ambiguity = self.planner.assess_ambiguity(task.raw_input)

        if ambiguity:
            prompt = BRAIN_CLARIFY_PROMPT.format(
                task=task.raw_input,
                ambiguity_type=ambiguity,
            )
            try:
                response = self._llm_call(prompt)
                clarification = self._parse_json(response) or {
                    "question": f"Could you clarify your task? It seems {ambiguity.replace('_', ' ')}.",
                    "options": [],
                    "ambiguity_type": ambiguity,
                }
            except Exception as e:
                clarification = {
                    "question": "Could you provide more detail about what you'd like to do?",
                    "options": [],
                    "ambiguity_type": ambiguity,
                }

            state["needs_clarification"] = True
            state["clarification"] = clarification
            log.info("Clarification needed", ambiguity=ambiguity)
        else:
            state["needs_clarification"] = False
            state["clarification"] = {}

        return state

    def _node_plan(self, state: BrainState) -> BrainState:
        task = state["task"]
        session = state["session"]
        log.info("Brain: plan node", task_id=task.task_id)
        brain_context = self._task_logger.get_full_brain_context(task.raw_input)

        routing = self.router.route(
            task.raw_input,
            session.get_recent_context(),
            self._llm_call,
            brain_context=brain_context,
        )

        plan = self.planner.build_plan(routing)
        task.execution_plan = plan
        state["routing"] = plan
        state["needs_factory"] = plan.get("needs_factory", False)

        return state

    def _node_execute(self, state: BrainState) -> BrainState:
        task = state["task"]
        routing = state["routing"]
        log.info("Brain: execute node", task_id=task.task_id)
        
        plan = build_plan_from_routing(routing, task.raw_input)
        if plan:
            orchestrator = WorkflowOrchestrator(agent_runner=self._run_agent_for_workflow)
            result = orchestrator.run(plan)
            state["agent_result"] = {
                "success": result.success,
                "message": result.combined_output,
                "data": result.as_dict(),
            }
            task.outcome = TaskOutcome.SUCCESS if result.success else TaskOutcome.FAILED
            return state

        agent_name = routing.get("agent")
        goal = routing.get("goal", task.raw_input)
        context = routing.get("context", {})

        if not agent_name:
            state["agent_result"] = {
                "success": False,
                "error": "No agent selected",
                "failure_type": FailureType.BRAIN,
            }
            task.failure_type = FailureType.BRAIN
            return state

        # Get agent instance
        agent = self._get_agent(agent_name)
        if not agent:
            state["agent_result"] = {
                "success": False,
                "error": f"Agent '{agent_name}' could not be instantiated",
                "failure_type": FailureType.BRAIN,
            }
            task.failure_type = FailureType.BRAIN
            return state

        # Execute
        result = agent.execute(task, goal, context)
        state["agent_result"] = result

        # Update task outcome
        if result.get("success"):
            task.outcome = TaskOutcome.SUCCESS
        elif result.get("awaiting_confirmation"):
            task.status = TaskStatus.AWAITING_CONFIRMATION
            task.outcome = TaskOutcome.PENDING
        else:
            task.outcome = TaskOutcome.FAILED
            if result.get("failure_type"):
                task.failure_type = result["failure_type"]

        return state

    def _node_synthesise(self, state: BrainState) -> BrainState:
        task = state["task"]
        routing = state["routing"]
        agent_result = state["agent_result"]
        session = state["session"]
        log.info("Brain: synthesise node", task_id=task.task_id)

        task.status = TaskStatus.SYNTHESISING

        # Build a concise result summary for the LLM
        result_summary = {
            "success": agent_result.get("success"),
            "message": agent_result.get("message", ""),
            "error": agent_result.get("error", ""),
            "data_keys": list(agent_result.get("data", {}).keys()),
        }
        if agent_result.get("awaiting_confirmation"):
            result_summary["awaiting_confirmation"] = agent_result.get("message")

        prompt = BRAIN_SYNTHESISE_PROMPT.format(
            task=task.raw_input,
            agent=routing.get("agent", "unknown"),
            result=json.dumps(result_summary, indent=2),
        )

        try:
            response = self._llm_call(prompt)
            final_response = response.strip()
        except Exception as e:
            if agent_result.get("success"):
                final_response = agent_result.get("message", "Task completed.")
            else:
                final_response = f"Task failed: {agent_result.get('error', 'Unknown error')}"

        state["final_response"] = final_response
        task.status = TaskStatus.COMPLETED
        task.result = {"response": final_response, "agent_data": agent_result.get("data", {})}
        task.mark_updated()
        task_store.save(task)  # ← add this line

        # Update session
        session.add_turn(
            task_id=task.task_id,
            agent=routing.get("agent", "unknown"),
            goal=routing.get("goal", task.raw_input),
            result=agent_result,
        )

        log.info("Brain: synthesis complete", task_id=task.task_id)
        return state

    # ── Routing conditions ───────────────────────────────────────
    def _should_clarify(self, state: BrainState) -> str:
        if state.get("needs_clarification"):
            return "clarify"
        return "plan"

    def _after_plan(self, state: BrainState) -> str:
        if state.get("needs_factory"):
            return "factory"
        return "execute"

    def _node_factory(self, state: BrainState) -> BrainState:
        task = state["task"]
        log.info("Brain: factory node", task_id=task.task_id)

        def on_agent_created(success, message, spec):
            if success:
              log.info("Factory: agent ready", name=spec.name)
            else:
              log.error("Factory: agent creation failed", reason=message)

        self.agent_factory.create_async(task.raw_input, callback=on_agent_created)

        state["final_response"] = (
            "I don't have an agent for that task yet. "
            "I'm building one now — this usually takes about 30 seconds. "
            "Please try again shortly."
        )
        task.status = TaskStatus.COMPLETED
        task.outcome = TaskOutcome.PARTIAL
        task.mark_updated()
        task_store.save(task)
        return state

    # ── Graph construction ───────────────────────────────────────
    def _build_graph(self):
        graph = StateGraph(BrainState)

        graph.add_node("clarify",    self._node_clarify)
        graph.add_node("plan",       self._node_plan)
        graph.add_node("execute",    self._node_execute)
        graph.add_node("synthesise", self._node_synthesise)
        graph.add_node("factory",    self._node_factory)   # ← add this

        graph.set_entry_point("clarify")

        graph.add_conditional_edges(
            "clarify",
            self._should_clarify,
            {"clarify": END, "plan": "plan"},
        )

        graph.add_conditional_edges(
            "plan",
            self._after_plan,
            {"execute": "execute", "factory": "factory"},  # ← update this
        )

        graph.add_edge("factory",    END)                  # ← add this
        graph.add_edge("execute",    "synthesise")
        graph.add_edge("synthesise", END)

        return graph.compile()
    # ── Main entry point ─────────────────────────────────────────
    def process(self, task_input: str, session: Optional[SessionContext] = None) -> dict:
        if not session:
            session = SessionContext(session_id=str(uuid.uuid4()))

        task = Task(raw_input=task_input)
        log.info("Brain processing task", task_id=task.task_id, input=task_input[:80])

        initial_state: BrainState = {
            "task":                task,
            "session":             session,
            "routing":             {},
            "agent_result":        {},
            "final_response":      "",
            "needs_clarification": False,
            "clarification":       {},
            "needs_factory":       False,
            "error":               None,
        }

        start = time.time()
        with self._task_logger.task(task_input) as ctx:
            final_state = self.graph.invoke(initial_state)
            duration = (time.time() - start) * 1000

            # populate context from final state
            agent_name = final_state["routing"].get("agent", "")
            if agent_name:
                ctx.add_agent(agent_name)

            outcome = task.outcome
            if outcome == TaskOutcome.SUCCESS:
                ctx.set_outcome("success")
            elif outcome == TaskOutcome.PARTIAL:
                ctx.set_outcome("partial")
            elif str(outcome) == "pending":
                ctx.set_outcome("partial")
            else:
                ctx.set_outcome("failed")

            ctx.set_category(final_state["routing"].get("category", ""))
        # ── logger commits the record here automatically ─────────────

        log.info("Brain task complete", task_id=task.task_id,
             duration_ms=round(duration, 1), outcome=task.outcome)


        # Build response
        if final_state.get("needs_clarification"):
            return {
                "status": "clarification_needed",
                "task_id": task.task_id,
                "clarification": final_state["clarification"],
                "session": session,
            }

        if final_state.get("needs_factory"):
            return {
                "status": "factory_needed",
                "task_id": task.task_id,
                "message": "No agent available for this task. Factory activation required.",
                "session": session,
            }

        return {
            "status": "completed" if task.outcome == TaskOutcome.SUCCESS else "failed",
            "task_id": task.task_id,
            "response": final_state.get("final_response", ""),
            "agent": final_state["routing"].get("agent"),
            "outcome": task.outcome,
            "awaiting_confirmation": final_state["agent_result"].get("awaiting_confirmation", False),
            "confirmation_message": final_state["agent_result"].get("message", ""),
            "data": final_state["agent_result"].get("data", {}),
            "session": session,
            "duration_ms": round(duration, 1),
        }

    # ── Agent instantiation ──────────────────────────────────────
    def _get_agent(self, name: str):
        # Try factory map first
        if name in self._agent_map:
            return self._agent_map[name]()

        # Try loading from registry (for factory-generated agents)
        spec = agent_registry.get(name)
        if spec:
            log.warning("Dynamic agent loading not yet implemented", agent=name)

        return None
    
    def _run_agent_for_workflow(self, agent_name: str, goal: str, context: dict) -> dict:
        """Adapter so WorkflowOrchestrator can call agents the same way _node_execute does."""
        agent = self._get_agent(agent_name)
        if not agent:
            return {"success": False, "error": f"Agent '{agent_name}' could not be instantiated"}

        fake_task = Task(raw_input=goal)
        return agent.execute(fake_task, goal, context)

    def _parse_json(self, text: str):
        try:
            text = text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])
            return json.loads(text)
        except Exception:
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    return None
            return None


# ── Singleton ────────────────────────────────────────────────────
brain = Brain()