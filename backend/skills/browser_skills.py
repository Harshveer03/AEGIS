from playwright.sync_api import sync_playwright
from backend.models.skill import SkillResult


def fetch_page(url: str) -> SkillResult:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=15000)
            content = page.inner_text("body")
            title = page.title()
            browser.close()
        return SkillResult(success=True, data={"url": url, "title": title, "content": content[:5000]})
    except Exception as e:
        return SkillResult(success=False, error=str(e))


def take_screenshot(url: str, save_path: str = "screenshot.png") -> SkillResult:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=15000)
            page.screenshot(path=save_path, full_page=True)
            browser.close()
        return SkillResult(success=True, data={"url": url, "saved_to": save_path})
    except Exception as e:
        return SkillResult(success=False, error=str(e))


def extract_links(url: str) -> SkillResult:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=15000)
            links = page.eval_on_selector_all(
                "a[href]",
                "elements => elements.map(e => ({text: e.innerText.trim(), href: e.href}))"
            )
            browser.close()
        return SkillResult(success=True, data={"url": url, "links": links, "count": len(links)})
    except Exception as e:
        return SkillResult(success=False, error=str(e))