import json
import threading
from pathlib import Path
from config import settings
from backend.models.agent import AgentSpec, CapabilitySchema
from backend.registry.agent_registry import agent_registry
from backend.registry.skill_registry import skill_registry
from backend.factory.validator import validate_agent_spec
import structlog

log = structlog.get_logger()

AGENT_CREATION_PROMPT = """You are designing a specialist AI agent for the AEGIS system.

Task that needs handling: {task}

Available skills:
{skills}

Design an agent that can handle this task using the available skills.

Respond with JSON only — no explanation, no markdown:
{{
  "name": "<snake_case_agent_name>",
  "description": "<one sentence — what this agent does>",
  "domain": "<domain>",
  "task_types": ["<task_type_1>", "<task_type_2>"],
  "system_prompt": "<complete system prompt for this agent>",
  "skills": ["<skill_name_1>", "<skill_name_2>"],
  "reasoning": "<why these skills handle the task>"
}}

Rules:
- Only use skills from the available list
- Name must be snake_case ending in _agent
- System prompt must be specific and actionable
- Pick only the skills actually needed"""


class AgentFactory:

    def __init__(self, llm_call):
        self.llm_call = llm_call
        self._output_path = Path(settings.generated_agents_path)
        self._output_path.mkdir(parents=True, exist_ok=True)
        self._pending: dict[str, str] = {}

    def create_async(self, task_input: str, callback=None):
        thread = threading.Thread(
            target=self._create_worker,
            args=(task_input, callback),
            daemon=True,
        )
        thread.start()
        log.info("Agent Factory: async creation started", task=task_input[:60])
        return thread

    def _create_worker(self, task_input: str, callback=None):
        success, message, spec = self.create(task_input)
        if callback:
            callback(success, message, spec)

    def create(self, task_input: str) -> tuple[bool, str, AgentSpec | None]:
        log.info("Agent Factory: creating agent", task=task_input[:60])

        # ── Step 1: Build skills summary ─────────────────────────
        skills = skill_registry.get_all()
        skills_desc = "\n".join([
            f"- {s.name}: {s.description}" for s in skills
        ])

        # ── Step 2: Generate agent spec ──────────────────────────
        prompt = AGENT_CREATION_PROMPT.format(
            task=task_input,
            skills=skills_desc,
        )

        try:
            response = self.llm_call(prompt).strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])
            spec_dict = json.loads(response)
        except json.JSONDecodeError as e:
            import re
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                try:
                    spec_dict = json.loads(match.group(0))
                except Exception:
                    return False, f"Could not parse agent spec: {e}", None
            else:
                return False, f"LLM returned invalid JSON: {e}", None
        except Exception as e:
            return False, f"LLM call failed: {e}", None

        # ── Step 3: Validate spec ────────────────────────────────
        ok, msg = validate_agent_spec(spec_dict)
        if not ok:
            return False, f"Agent spec invalid: {msg}", None

        # ── Step 4: Validate skills exist ────────────────────────
        requested_skills = spec_dict.get("skills", [])
        missing = [s for s in requested_skills if not skill_registry.exists(s)]
        if missing:
            # Remove missing skills rather than fail
            requested_skills = [s for s in requested_skills if skill_registry.exists(s)]
            log.warning("Agent Factory: removed missing skills", missing=missing)
            if not requested_skills:
                return False, f"No valid skills available for agent", None

        # ── Step 5: Get predecessor briefing if retiring ─────────
        predecessor_briefing = self._extract_predecessor_briefing(
            spec_dict.get("name", "")
        )

        # ── Step 6: Build AgentSpec ──────────────────────────────
        agent_spec = AgentSpec(
            name=spec_dict["name"],
            description=spec_dict["description"],
            capability_schema=CapabilitySchema(
                domain=spec_dict.get("domain", "custom"),
                task_types=spec_dict.get("task_types", []),
                input_formats=["text"],
                output_formats=["text"],
            ),
            system_prompt=spec_dict["system_prompt"],
            skills=requested_skills,
            is_generated=True,
            predecessor_briefing=predecessor_briefing,
        )

        # ── Step 7: Register ─────────────────────────────────────
        agent_registry.register(agent_spec)
        log.info(
            "Agent Factory: agent registered",
            name=agent_spec.name,
            skills=agent_spec.skills,
        )

        return True, f"Agent '{agent_spec.name}' created and registered", agent_spec

    def _extract_predecessor_briefing(self, agent_name: str) -> str | None:
        existing = agent_registry.get(agent_name)
        if not existing or existing.is_active:
            return None
        if existing.failure_count == 0:
            return None
        return (
            f"Your predecessor '{agent_name}' was retired after {existing.task_count} tasks "
            f"with {existing.failure_count} failures. "
            f"Failure breakdown: {existing.failure_attribution}. "
            f"Common issues to avoid based on failure history."
        )


# ── Factory instance ─────────────────────────────────────────────
_agent_factory: AgentFactory | None = None


def get_agent_factory(llm_call) -> AgentFactory:
    global _agent_factory
    if _agent_factory is None:
        _agent_factory = AgentFactory(llm_call)
    return _agent_factory