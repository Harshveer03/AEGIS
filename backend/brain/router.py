import json
from typing import Optional
from backend.models.agent import AgentSpec
from backend.registry.agent_registry import agent_registry
import structlog

log = structlog.get_logger()


class AgentRouter:

    def route(
        self,
        task_input: str,
        session_context: list[dict],
        llm_call,
        brain_context: str = "",
    ) -> dict:
        active_agents = agent_registry.get_active()

        if not active_agents:
            return {
                "agent": None,
                "goal": task_input,
                "context": {},
                "confidence": 0.0,
                "reasoning": "No active agents available",
                "needs_factory": True,
            }

        agents_desc = "\n".join([
            f"- {a.name} (score: {a.performance_score}): {a.description}"
            for a in active_agents
        ])

        context_desc = "\n".join([
            f"  Turn {t['turn']}: [{t['agent']}] {t['goal'][:60]} → {'OK' if t['success'] else 'FAILED'}"
            for t in session_context[-3:]
        ]) or "No prior context"

        from backend.brain.prompts import BRAIN_ROUTER_PROMPT
        prompt = BRAIN_ROUTER_PROMPT.format(
            agents=agents_desc,
            task=task_input,
            context=context_desc,
        )

        if brain_context:
            prompt = f"{brain_context}\n\n{prompt}"

        try:
            response = llm_call(prompt)
            routing = self._parse_json(response)

            if not routing:
                return self._fallback_route(task_input, active_agents)

            agent_name = routing.get("agent", "none")
            confidence = routing.get("confidence", 0.0)

            # Validate agent exists
            if agent_name != "none" and not agent_registry.exists(agent_name):
                log.warning("LLM routed to unknown agent", agent=agent_name)
                return self._fallback_route(task_input, active_agents)

            # Score-boost: if confidence is low, prefer highest-scoring capable agent
            if confidence < 0.5 and agent_name != "none":
                domain_agents = agent_registry.find_by_task_type(agent_name)
                if domain_agents:
                    best = max(domain_agents, key=lambda a: a.performance_score)
                    if best.performance_score > agent_registry.get(agent_name).performance_score:
                        routing["agent"] = best.name
                        routing["reasoning"] += f" (score-boosted to {best.name})"

            needs_factory = agent_name == "none"
            routing["needs_factory"] = needs_factory

            log.info(
                "Task routed",
                agent=agent_name,
                confidence=confidence,
                needs_factory=needs_factory,
            )
            return routing

        except Exception as e:
            log.error("Router failed", error=str(e))
            return self._fallback_route(task_input, active_agents)

    def _fallback_route(self, task_input: str, agents: list[AgentSpec]) -> dict:
        task_lower = task_input.lower()

        # Simple keyword fallback
        if any(w in task_lower for w in ["file", "read", "write", "folder", "directory", "path"]):
            agent = next((a for a in agents if a.name == "file_agent"), None)
        elif any(w in task_lower for w in ["search", "find", "news", "research", "look up"]):
            agent = next((a for a in agents if a.name == "search_agent"), None)
        elif any(w in task_lower for w in ["http", "url", "website", "webpage", "fetch"]):
            agent = next((a for a in agents if a.name == "browser_agent"), None)
        elif any(w in task_lower for w in ["code", "python", "run", "execute", "script"]):
            agent = next((a for a in agents if a.name == "code_agent"), None)
        else:
            agent = max(agents, key=lambda a: a.performance_score) if agents else None

        return {
            "agent": agent.name if agent else None,
            "goal": task_input,
            "context": {},
            "confidence": 0.3,
            "reasoning": "Fallback keyword routing",
            "needs_factory": agent is None,
        }

    def _parse_json(self, text: str) -> Optional[dict]:
        try:
            # Strip markdown fences if present
            text = text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])
            return json.loads(text)
        except Exception:
            # Try extracting JSON from mixed text
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    return None
            return None