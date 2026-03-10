"""Serper.dev client for fetching real-time Google SERP data.

Setup
-----
1. Create a free account at https://serper.dev/
2. Copy your API key and set SERPER_API_KEY in your .env file.

Usage
-----
    from career_compass_content_engine.tools.serper_client import SerperClient
    client = SerperClient(api_key="your-key")
    summary = client.competitor_summary(["early career manager tips", "first management role advice"])
    print(summary)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import requests


_SERPER_ENDPOINT = "https://google.serper.dev/search"


class SerperClient:
    """Thin wrapper around the Serper.dev Google Search API."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("SERPER_API_KEY is required.")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update(
            {"X-API-KEY": api_key, "Content-Type": "application/json"}
        )

    def search(
        self,
        query: str,
        num: int = 5,
        gl: str = "us",
        hl: str = "en",
    ) -> Dict[str, Any]:
        """Run a single Google search and return the raw Serper response."""
        payload = {"q": query, "num": num, "gl": gl, "hl": hl}
        try:
            resp = self.session.post(
                _SERPER_ENDPOINT,
                data=json.dumps(payload),
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            return {"error": str(exc), "organic": []}

    def fetch_serp_snapshot(
        self,
        queries: List[str],
        num_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Fetch SERP results for a list of queries.

        Returns a list of result dicts with keys:
            query, position, title, link, snippet
        """
        results: List[Dict[str, Any]] = []
        for query in queries:
            data = self.search(query, num=num_results)
            for item in data.get("organic", []):
                results.append(
                    {
                        "query": query,
                        "position": item.get("position", 0),
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                    }
                )
        return results

    def competitor_summary(
        self,
        queries: List[str],
        num_results: int = 5,
    ) -> str:
        """Return a prompt-ready string summarising top SERP results for a topic.

        Designed to give the research agent competitive context so it can:
        - See what angles competitors are using
        - Identify underserved angles or gaps
        - Calibrate headline differentiation
        """
        if not queries:
            return "(No SERP queries provided.)"

        raw = self.fetch_serp_snapshot(queries[:3], num_results=num_results)  # cap at 3 queries to limit API cost

        if not raw:
            return "[Serper] No results returned."

        lines: List[str] = [
            f"=== Live SERP Competitive Snapshot — {len(queries[:3])} query(ies) ===",
            "",
        ]

        current_query: Optional[str] = None
        for r in raw:
            if r["query"] != current_query:
                current_query = r["query"]
                lines.append(f'Query: "{current_query}"')
            lines.append(
                f"  #{r['position']}  {r['title']}"
            )
            if r["snippet"]:
                # Trim snippet to one line to keep prompt concise
                snippet = r["snippet"].replace("\n", " ").strip()[:160]
                lines.append(f"        → {snippet}")

        lines += [
            "",
            "Use this to: (a) avoid repeating the same angles, "
            "(b) choose a more specific or contrarian take, "
            "(c) identify underserved sub-questions.",
        ]
        return "\n".join(lines)
