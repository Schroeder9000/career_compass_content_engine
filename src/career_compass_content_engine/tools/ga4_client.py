"""Google Analytics 4 client for fetching page-level engagement data.

Setup
-----
1. Enable the Google Analytics Data API in Google Cloud for your project.
2. Add the service account email as a Viewer on the GA4 property
   (GA4 Admin → Account Access Management → Add users).
3. Set GA4_PROPERTY_ID to your numeric GA4 property ID (e.g. "123456789").
   Find it in GA4 Admin → Property Settings.
4. Reuse the same GSC_SERVICE_ACCOUNT_JSON key file (different scope added automatically).

Usage
-----
    from career_compass_content_engine.tools.ga4_client import GA4Client
    client = GA4Client(service_account_json="path/to/key.json", property_id="123456789")
    summary = client.fetch_engagement_summary(days=90)
    print(summary)
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List


class GA4Client:
    """Thin wrapper around the Google Analytics Data API v1beta."""

    def __init__(self, service_account_json: str, property_id: str) -> None:
        self.service_account_json = service_account_json
        self.property_id = property_id.lstrip("properties/")  # accept both forms
        self._client = None  # lazy-loaded

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from google.analytics.data_v1beta import BetaAnalyticsDataClient
            from google.oauth2 import service_account
        except ImportError as exc:
            raise ImportError(
                "google-analytics-data and google-auth are required for GA4. "
                "Run: pip install google-analytics-data google-auth"
            ) from exc

        key_path = Path(self.service_account_json)
        if not key_path.exists():
            raise FileNotFoundError(
                f"GA4 service account key not found: {key_path}. "
                "Set GSC_SERVICE_ACCOUNT_JSON to the path of your JSON key file."
            )

        creds = service_account.Credentials.from_service_account_file(
            str(key_path),
            scopes=["https://www.googleapis.com/auth/analytics.readonly"],
        )
        self._client = BetaAnalyticsDataClient(credentials=creds)
        return self._client

    # ── Public API ─────────────────────────────────────────────────────────────

    def fetch_page_engagement(
        self,
        days: int = 90,
        row_limit: int = 30,
    ) -> List[Dict[str, Any]]:
        """Fetch per-page engagement metrics for the trailing N days.

        Returns a list of dicts with keys:
            page_path, sessions, avg_engagement_seconds, scroll_percentage, bounce_rate
        """
        try:
            from google.analytics.data_v1beta.types import (
                DateRange,
                Dimension,
                Metric,
                RunReportRequest,
            )
        except ImportError as exc:
            raise ImportError(
                "google-analytics-data is required. "
                "Run: pip install google-analytics-data"
            ) from exc

        client = self._get_client()
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=days)

        request = RunReportRequest(
            property=f"properties/{self.property_id}",
            dimensions=[Dimension(name="pagePath")],
            metrics=[
                Metric(name="sessions"),
                Metric(name="averageSessionDuration"),
                Metric(name="scrolledUsers"),
                Metric(name="bounceRate"),
            ],
            date_ranges=[
                DateRange(
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                )
            ],
            limit=row_limit,
        )

        response = client.run_report(request)
        results = []
        for row in response.rows:
            page = row.dimension_values[0].value
            sessions = int(row.metric_values[0].value or 0)
            avg_dur = float(row.metric_values[1].value or 0)
            scrolled = float(row.metric_values[2].value or 0)
            bounce = float(row.metric_values[3].value or 0)
            results.append(
                {
                    "page_path": page,
                    "sessions": sessions,
                    "avg_engagement_seconds": round(avg_dur, 0),
                    "scrolled_users": round(scrolled, 0),
                    "bounce_rate": round(bounce * 100, 1),
                }
            )
        # Sort by sessions descending
        results.sort(key=lambda x: x["sessions"], reverse=True)
        return results

    def fetch_engagement_summary(
        self,
        days: int = 90,
        row_limit: int = 30,
    ) -> str:
        """Return a formatted summary string ready for injection into an LLM prompt.

        Flags:
        - High-traffic pages with low engagement (update candidates)
        - Low-traffic pages that may have topic gaps
        """
        try:
            pages = self.fetch_page_engagement(days=days, row_limit=row_limit)
        except Exception as exc:
            return f"[GA4] Data unavailable: {exc}"

        if not pages:
            return "[GA4] No page data returned — verify property ID and service account access."

        lines = [f"=== Google Analytics 4 — last {days} days ===", ""]
        lines.append("PAGE ENGAGEMENT (sessions, avg engagement time, scroll %):")
        for p in pages[:20]:
            slug = p["page_path"].strip("/").split("/")[-1] or "homepage"
            flags = []
            if p["avg_engagement_seconds"] < 60 and p["sessions"] > 50:
                flags.append("⚠ low engagement")
            if p["bounce_rate"] > 70:
                flags.append("⚠ high bounce")
            flag_str = "  " + ", ".join(flags) if flags else ""
            lines.append(
                f"  /{slug} — {p['sessions']} sessions, "
                f"{int(p['avg_engagement_seconds'])}s avg, "
                f"{p['scrolled_users']:.0f} scrolled"
                f"{flag_str}"
            )

        # Highlight update candidates: top traffic + poor engagement
        update_candidates = [
            p for p in pages
            if p["sessions"] > 30 and p["avg_engagement_seconds"] < 60
        ]
        if update_candidates:
            lines += ["", "UPDATE CANDIDATES (traffic but poor engagement — consider refreshing):"]
            for p in update_candidates[:5]:
                slug = p["page_path"].strip("/").split("/")[-1] or "homepage"
                lines.append(f"  /{slug}")

        return "\n".join(lines)
