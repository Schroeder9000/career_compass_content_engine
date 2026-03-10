"""Google Search Console client for fetching query-level performance data.

Setup
-----
1. Create a Google Cloud service account with the Search Console API enabled.
2. Download the JSON key file and set the path in GSC_SERVICE_ACCOUNT_JSON.
3. Add the service account email as a verified user in Search Console
   (Settings → Users and permissions → Add user).
4. Set GSC_SITE_URL to the verified property URL (e.g. "https://yoursite.com/").

Usage
-----
    from career_compass_content_engine.tools.gsc_client import GSCClient
    client = GSCClient(service_account_json="path/to/key.json", site_url="https://...")
    summary = client.fetch_performance_summary(days=90, row_limit=50)
    print(summary)  # formatted string ready for injection into the research prompt
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


class GSCClient:
    """Thin wrapper around the Google Search Console Search Analytics API v3."""

    def __init__(self, service_account_json: str, site_url: str) -> None:
        self.service_account_json = service_account_json
        # Domain properties use "sc-domain:example.com" format — no trailing slash.
        # URL-prefix properties require a trailing slash (e.g. "https://example.com/").
        if site_url.startswith("sc-domain:"):
            self.site_url = site_url.rstrip("/")
        else:
            self.site_url = site_url.rstrip("/") + "/"
        self._service = None  # lazy-loaded

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_service(self):
        """Build and cache the GSC API service client."""
        if self._service is not None:
            return self._service
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise ImportError(
                "google-api-python-client and google-auth are required for GSC. "
                "Run: pip install google-api-python-client google-auth"
            ) from exc

        key_path = Path(self.service_account_json)
        if not key_path.exists():
            raise FileNotFoundError(
                f"GSC service account key not found: {key_path}. "
                "Set GSC_SERVICE_ACCOUNT_JSON to the path of your JSON key file."
            )

        creds = service_account.Credentials.from_service_account_file(
            str(key_path),
            scopes=["https://www.googleapis.com/auth/webmasters.readonly"],
        )
        self._service = build("searchconsole", "v1", credentials=creds, cache_discovery=False)
        return self._service

    # ── Public API ─────────────────────────────────────────────────────────────

    def fetch_top_queries(
        self,
        days: int = 90,
        row_limit: int = 50,
        start_row: int = 0,
    ) -> List[Dict[str, Any]]:
        """Fetch top queries ranked by impressions for the trailing N days.

        Returns a list of dicts with keys:
            query, clicks, impressions, ctr, position
        """
        service = self._get_service()
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=days)

        body = {
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "dimensions": ["query"],
            "rowLimit": row_limit,
            "startRow": start_row,
            "dimensionFilterGroups": [],
        }

        response = (
            service.searchanalytics()
            .query(siteUrl=self.site_url, body=body)
            .execute()
        )
        rows = response.get("rows", [])
        return [
            {
                "query": row["keys"][0],
                "clicks": row.get("clicks", 0),
                "impressions": row.get("impressions", 0),
                "ctr": round(row.get("ctr", 0) * 100, 2),
                "position": round(row.get("position", 0), 1),
            }
            for row in rows
        ]

    def fetch_top_pages(
        self,
        days: int = 90,
        row_limit: int = 30,
    ) -> List[Dict[str, Any]]:
        """Fetch top pages ranked by clicks for the trailing N days.

        Returns a list of dicts with keys:
            page, clicks, impressions, ctr, position
        """
        service = self._get_service()
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=days)

        body = {
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "dimensions": ["page"],
            "rowLimit": row_limit,
        }

        response = (
            service.searchanalytics()
            .query(siteUrl=self.site_url, body=body)
            .execute()
        )
        rows = response.get("rows", [])
        return [
            {
                "page": row["keys"][0],
                "clicks": row.get("clicks", 0),
                "impressions": row.get("impressions", 0),
                "ctr": round(row.get("ctr", 0) * 100, 2),
                "position": round(row.get("position", 0), 1),
            }
            for row in rows
        ]

    def fetch_performance_summary(
        self,
        days: int = 90,
        query_limit: int = 30,
        page_limit: int = 20,
    ) -> str:
        """Return a formatted summary string ready for injection into an LLM prompt.

        Covers:
        - Top queries by impressions with CTR and avg position
        - Queries with high impressions but position > 10 (opportunity gap)
        - Top pages by clicks
        """
        try:
            queries = self.fetch_top_queries(days=days, row_limit=query_limit)
            pages = self.fetch_top_pages(days=days, row_limit=page_limit)
        except Exception as exc:
            return f"[GSC] Data unavailable: {exc}"

        if not queries and not pages:
            return "[GSC] No data returned — verify site URL and service account permissions."

        lines: List[str] = [
            f"=== Google Search Console — last {days} days ===",
            "",
            "TOP QUERIES BY IMPRESSIONS:",
        ]
        for q in queries[:20]:
            flag = " ⚠ low CTR" if q["ctr"] < 2.0 and q["impressions"] > 100 else ""
            lines.append(
                f"  '{q['query']}' — {q['impressions']:,} impr, {q['clicks']} clicks, "
                f"{q['ctr']}% CTR, pos {q['position']}{flag}"
            )

        # Opportunity: high impressions, position 4–20 (ranking but not top 3)
        opportunities = [
            q for q in queries if 4 <= q["position"] <= 20 and q["impressions"] >= 50
        ]
        if opportunities:
            lines += ["", "RANKING OPPORTUNITIES (position 4–20, worth improving):"]
            for q in opportunities[:10]:
                lines.append(
                    f"  '{q['query']}' — pos {q['position']}, {q['impressions']:,} impr"
                )

        lines += ["", "TOP PAGES BY CLICKS:"]
        for p in pages[:10]:
            lines.append(
                f"  {p['page'].split('/')[-2] or p['page']} — {p['clicks']} clicks, pos {p['position']}"
            )

        return "\n".join(lines)
