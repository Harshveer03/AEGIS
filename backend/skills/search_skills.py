from ddgs import DDGS
from backend.models.skill import SkillResult


def web_search(query: str, max_results: int = 5) -> SkillResult:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        formatted = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
        ]
        return SkillResult(success=True, data={"results": formatted, "count": len(formatted), "query": query})
    except Exception as e:
        return SkillResult(success=False, error=str(e))


def news_search(query: str, max_results: int = 5) -> SkillResult:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=max_results))
        formatted = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("body", ""),
                "date": r.get("date", ""),
                "source": r.get("source", ""),
            }
            for r in results
        ]
        return SkillResult(success=True, data={"results": formatted, "count": len(formatted), "query": query})
    except Exception as e:
        return SkillResult(success=False, error=str(e))