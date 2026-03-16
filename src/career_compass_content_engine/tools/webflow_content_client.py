"""Webflow CMS content client for fetching published post metadata.

Queries the live Webflow CMS to give the research and writing agents
accurate, up-to-date information about published posts — enabling
real internal linking suggestions instead of guesses based on a
static snapshot.

Setup
-----
Reuses the WEBFLOW_API_TOKEN and WEBFLOW_COLLECTION_ID already required
by the publishing pipeline. No additional credentials needed.

Usage
-----
    from career_compass_content_engine.tools.webflow_content_client import WebflowContentClient

    client = WebflowContentClient(
        api_token="your-webflow-api-token",
        collection_id="your-collection-id",
    )
    summary = client.published_content_summary()
    print(summary)  # prompt-ready string for the research agent
"""

from __future__ import annotations

from typing import Any, Dict, List

import requests


_WEBFLOW_API_BASE = "https://api.webflow.com/v2"
_PAGE_SIZE = 100  # Webflow max per request


class WebflowContentClient:
    """Fetches published CMS items from Webflow for internal-link grounding."""

    def __init__(self, api_token: str, collection_id: str) -> None:
        if not api_token:
            raise ValueError("WEBFLOW_API_TOKEN is required.")
        if not collection_id:
            raise ValueError("WEBFLOW_COLLECTION_ID is required.")
        self.collection_id = collection_id
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_token}",
                "accept-version": "1.0.0",
                "Content-Type": "application/json",
            }
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_items_page(self, offset: int = 0) -> Dict[str, Any]:
        url = f"{_WEBFLOW_API_BASE}/collections/{self.collection_id}/items"
        params = {"limit": _PAGE_SIZE, "offset": offset}
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            return {"error": str(exc), "items": [], "pagination": {"total": 0}}

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_published_items(self) -> List[Dict[str, Any]]:
        """Fetch all published CMS items, paginating through results.

        Returns a list of simplified dicts with keys:
            id, name, slug, summary, tags, published_on
        """
        all_items: List[Dict[str, Any]] = []
        offset = 0

        while True:
            page = self._get_items_page(offset=offset)
            if "error" in page:
                break
            raw_items = page.get("items", [])
            for item in raw_items:
                field_data = item.get("fieldData", {})
                # Only include items that are published (not draft/archived)
                if item.get("isDraft") or item.get("isArchived"):
                    continue
                all_items.append(
                    {
                        "id": item.get("id", ""),
                        "name": field_data.get("name", ""),
                        "slug": field_data.get("slug", ""),
                        "summary": field_data.get("post-summary", "")
                        or field_data.get("summary", "")
                        or field_data.get("meta-description", ""),
                        "tags": field_data.get("tags", []),
                        "published_on": item.get("lastPublished", ""),
                    }
                )
            pagination = page.get("pagination", {})
            total = pagination.get("total", 0)
            offset += len(raw_items)
            if not raw_items or offset >= total:
                break

        return all_items

    def published_content_summary(self, max_items: int = 50) -> str:
        """Return a prompt-ready string of published post titles and slugs.

        Designed for injecting into the research agent so it can:
        - Avoid topic duplication across real published content
        - Recommend accurate internal linking targets by slug
        - Identify content clusters and gaps from the live corpus
        """
        items = self.fetch_published_items()
        if not items:
            return "(Webflow CMS: no published items found or fetch failed.)"

        # Most recently published first
        items = sorted(items, key=lambda x: x.get("published_on") or "", reverse=True)[:max_items]

        lines = [
            f"=== Live Published Content — {len(items)} post(s) from Webflow CMS ===",
            "",
        ]
        for item in items:
            title = item["name"] or "(untitled)"
            slug = item["slug"] or "(no-slug)"
            summary = item.get("summary", "").strip()
            tags = ", ".join(item.get("tags", [])) if item.get("tags") else ""
            lines.append(f"• {title}")
            lines.append(f"  slug: /{slug}")
            if summary:
                lines.append(f"  summary: {summary[:140]}")
            if tags:
                lines.append(f"  tags: {tags}")

        lines += [
            "",
            "Use this to: (a) avoid topic duplication, "
            "(b) recommend accurate internal links by exact slug, "
            "(c) identify content cluster gaps.",
        ]
        return "\n".join(lines)
