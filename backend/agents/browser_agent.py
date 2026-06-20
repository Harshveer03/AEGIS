from backend.models.agent import AgentSpec, CapabilitySchema
from backend.models.task import Task, FailureType
from backend.agents.base_agent import BaseAgent


class BrowserAgent(BaseAgent):

    def run(self, task: Task, goal: str, context: dict) -> dict:
        goal_lower = goal.lower()
        url = context.get("url") or self._extract_url(goal)

        if not url:
            return self.failure("No URL could be extracted from the goal")

        # ── Screenshot ──────────────────────────────────────────
        if any(w in goal_lower for w in ["screenshot", "capture", "snap"]):
            save_path = context.get("save_path", "screenshot.png")
            result = self.use_skill("take_screenshot", {"url": url, "save_path": save_path}, task)
            if result.success:
                return self.success(result.data, f"Screenshot saved to {save_path}")
            return self.failure(result.error or "Screenshot failed", FailureType.SKILL)

        # ── Extract links ────────────────────────────────────────
        if any(w in goal_lower for w in ["links", "urls", "hrefs", "extract link"]):
            result = self.use_skill("extract_links", {"url": url}, task)
            if result.success:
                return self.success(result.data, f"Extracted {result.data.get('count', 0)} links from {url}")
            return self.failure(result.error or "Link extraction failed", FailureType.SKILL)

        # ── Fetch page (default) ────────────────────────────────
        result = self.use_skill("fetch_page", {"url": url}, task)
        if result.success:
            return self.success(result.data, f"Fetched page: {result.data.get('title', url)}")
        return self.failure(result.error or "Page fetch failed", FailureType.SKILL)

    def _extract_url(self, text: str) -> str:
        import re
        match = re.search(r"https?://[^\s]+", text)
        return match.group(0) if match else ""


def create_browser_agent() -> BrowserAgent:
    spec = AgentSpec(
        name="browser_agent",
        description="Interacts with websites using a headless browser — fetches page content, extracts links, and takes screenshots",
        capability_schema=CapabilitySchema(
            domain="browser",
            task_types=["fetch_page", "take_screenshot", "extract_links", "web_interaction"],
            input_formats=["url", "text"],
            output_formats=["text", "image", "list"],
        ),
        system_prompt=(
            "You are the Browser Agent, a specialist in web interaction. "
            "You fetch web pages, extract links, and take screenshots using a headless browser. "
            "You handle timeouts gracefully and always return meaningful content."
        ),
        skills=["fetch_page", "take_screenshot", "extract_links"],
    )
    return BrowserAgent(spec)