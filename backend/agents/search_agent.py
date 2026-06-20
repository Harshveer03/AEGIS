from backend.models.agent import AgentSpec, CapabilitySchema
from backend.models.task import Task, FailureType
from backend.agents.base_agent import BaseAgent


class SearchAgent(BaseAgent):

    def run(self, task: Task, goal: str, context: dict) -> dict:
        goal_lower = goal.lower()
        query = context.get("query") or self._extract_query(goal)

        if not query:
            return self.failure("No search query could be extracted from the goal")

        # ── News search ─────────────────────────────────────────
        if any(w in goal_lower for w in ["news", "latest", "recent", "today", "headlines"]):
            max_results = context.get("max_results", 5)
            result = self.use_skill("news_search", {"query": query, "max_results": max_results}, task)
            if result.success:
                return self.success(result.data, f"Found {result.data.get('count', 0)} news results for '{query}'")
            return self.failure(result.error or "News search failed", FailureType.SKILL)

        # ── Web search ──────────────────────────────────────────
        max_results = context.get("max_results", 5)
        result = self.use_skill("web_search", {"query": query, "max_results": max_results}, task)
        if result.success:
            return self.success(result.data, f"Found {result.data.get('count', 0)} results for '{query}'")
        return self.failure(result.error or "Web search failed", FailureType.SKILL)

    def _extract_query(self, goal: str) -> str:
        import re
        # Strip common instruction words to get the query
        stopwords = [
            "search for", "search", "find", "look up", "look for",
            "get me", "get", "show me", "show", "what is", "what are",
            "tell me about", "tell me", "news about", "news on",
            "latest on", "latest about", "research",
        ]
        query = goal.lower()
        for word in stopwords:
            query = query.replace(word, "").strip()
        return query.strip() or goal


def create_search_agent() -> SearchAgent:
    spec = AgentSpec(
        name="search_agent",
        description="Searches the web and news for information, answers research questions, and retrieves current information from the internet",
        capability_schema=CapabilitySchema(
            domain="search",
            task_types=["web_search", "news_search", "research", "find_information"],
            input_formats=["text", "query"],
            output_formats=["text", "list", "structured"],
        ),
        system_prompt=(
            "You are the Search Agent, a specialist in web and news retrieval. "
            "You find accurate, relevant information from the internet. "
            "You always return structured results with titles, URLs, and summaries. "
            "You prefer recent sources and flag when information may be outdated."
        ),
        skills=["web_search", "news_search"],
    )
    return SearchAgent(spec)