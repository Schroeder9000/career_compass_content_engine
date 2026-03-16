"""Competitor content fetcher — fetches and summarizes top-ranking pages.

Retrieves the actual body content from SERP-ranked competitor URLs,
strips HTML boilerplate, and returns a summarized view for the research
agent. This gives the agent real competitor angles rather than 50-word
snippets, enabling genuinely differentiated topic selection.

Security note: Only fetches URLs returned by the Serper SERP API.
All network requests have strict timeouts and content-size limits.

Usage
-----
    from career_compass_content_engine.tools.competitor_fetcher import CompetitorFetcher
    from career_compass_content_engine.tools.serper_client import SerperClient

    serper = SerperClient(api_key="your-key")
    results = serper.fetch_serp_snapshot(["early career manager tips"], num_results=3)
    urls = [r["link"] for r in results[:3]]

    fetcher = CompetitorFetcher()
    summary = fetcher.summarize_urls(urls)
    print(summary)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests


# Max bytes to read per page — prevents runaway downloads
_MAX_CONTENT_BYTES = 200_000
# Truncate extracted text to this many characters before returning
_MAX_TEXT_CHARS = 3_000
# Domains that reliably block or return unhelpful content for scraping
_BLOCKED_DOMAINS = frozenset(
    {
        "linkedin.com",
        "facebook.com",
        "instagram.com",
        "twitter.com",
        "x.com",
        "reddit.com",
        "quora.com",
        "pinterest.com",
        "youtube.com",
        "tiktok.com",
    }
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; CareerCompassBot/1.0; +https://careercompassapp.io)"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


def _is_allowed_url(url: str) -> bool:
    """Return True only for http/https URLs not in the blocked-domain list."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname or ""
        # Block exact match or any subdomain of a blocked domain
        for blocked in _BLOCKED_DOMAINS:
            if hostname == blocked or hostname.endswith(f".{blocked}"):
                return False
        return True
    except Exception:
        return False


def _strip_html(html: str) -> str:
    """Extract readable text from HTML without a heavy dependency."""
    # Remove <script> and <style> blocks entirely
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove all remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_main_content(text: str, url: str) -> str:
    """Heuristically extract the article body from stripped page text."""
    # Text before the first 200 chars is usually nav/header — skip it
    body = text[200:] if len(text) > 200 else text
    # Don't return more than _MAX_TEXT_CHARS
    return body[:_MAX_TEXT_CHARS]


class CompetitorFetcher:
    """Fetches and summarizes competitor article content from SERP URLs."""

    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)

    def fetch_url(self, url: str) -> Optional[str]:
        """Fetch a single URL and return extracted plain text, or None on failure."""
        if not _is_allowed_url(url):
            return None
        try:
            resp = self.session.get(
                url,
                timeout=self.timeout,
                stream=True,
                allow_redirects=True,
            )
            # Only process HTML responses
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                return None
            # Read up to _MAX_CONTENT_BYTES to avoid huge page downloads
            raw_bytes = b""
            for chunk in resp.iter_content(chunk_size=8192):
                raw_bytes += chunk
                if len(raw_bytes) >= _MAX_CONTENT_BYTES:
                    break
            html = raw_bytes.decode("utf-8", errors="replace")
            text = _strip_html(html)
            return _extract_main_content(text, url)
        except requests.RequestException:
            return None

    def fetch_urls(self, urls: List[str], max_urls: int = 3) -> List[Dict[str, str]]:
        """Fetch multiple URLs and return a list of result dicts.

        Returns a list of dicts with keys:
            url, text (extracted body), error (if failed)
        """
        results = []
        for url in urls[:max_urls]:
            text = self.fetch_url(url)
            if text:
                results.append({"url": url, "text": text, "error": ""})
            else:
                results.append({"url": url, "text": "", "error": "fetch failed or blocked"})
        return results

    def summarize_urls(self, urls: List[str], max_urls: int = 3) -> str:
        """Return a prompt-ready competitive content summary for the given URLs.

        Designed for injection into the research agent so it can identify
        the specific angles, frameworks, and depth competitors use — then
        choose a more specific, contrarian, or deeper take.
        """
        if not urls:
            return "(No competitor URLs provided.)"

        results = self.fetch_urls(urls, max_urls=max_urls)
        successful = [r for r in results if r["text"]]

        if not successful:
            return "[Competitor Fetcher] Could not retrieve any competitor content."

        lines = [
            f"=== Competitor Content Analysis — {len(successful)} page(s) fetched ===",
            "",
            "INSTRUCTIONS FOR USE:",
            "- Identify the angles, frameworks, and depth each competitor uses.",
            "- Do NOT repeat these angles. Choose something more specific, contrarian, or deeper.",
            "- Note what's missing (emotions, exact scripts, real examples, contrarian takes).",
            "",
        ]

        for i, r in enumerate(successful, start=1):
            domain = urlparse(r["url"]).hostname or r["url"]
            lines.append(f"--- Competitor {i}: {domain} ---")
            lines.append(f"URL: {r['url']}")
            lines.append("")
            # Show a clean excerpt; trim trailing mid-word cut
            excerpt = r["text"].strip()
            last_period = excerpt.rfind(".")
            if last_period > len(excerpt) - 200:
                excerpt = excerpt[: last_period + 1]
            lines.append(excerpt)
            lines.append("")

        return "\n".join(lines)
