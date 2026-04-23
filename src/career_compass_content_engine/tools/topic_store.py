"""Persistent topic history store.

Stores a JSON log of every completed run at .cache/topic_history.json.
Provides:
- Deduplication of previously covered topics for the research agent
- Per-run outcome tracking (score, revision count, Webflow item ID)
- A formatted history string for injecting into the research prompt

File schema (JSON array of objects):
    [
        {
            "run_id": "20260310_143022",
            "timestamp": "2026-03-10T14:30:22",
            "topic": "graceful resignation",
            "slug": "exiting-gracefully-7-steps-to-quit-without-burning-bridges",
            "suggested_title": "Exiting Gracefully: 7 Steps to Quit Without Burning Bridges",
            "webflow_item_id": "abc123",
            "final_score": 8.7,
            "revision_count": 1,
            "evaluation_passed": true
        },
        ...
    ]
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


_DEFAULT_PATH = Path(".cache/topic_history.json")


class TopicStore:
    """Read/write interface for the run history JSON log."""

    def __init__(self, path: Path = _DEFAULT_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ── I/O helpers ──────────────────────────────────────────────────────────

    def _load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self, records: List[Dict[str, Any]]) -> None:
        self.path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def append(
        self,
        *,
        topic: str,
        slug: str,
        suggested_title: str,
        final_score: float,
        revision_count: int,
        evaluation_passed: bool,
        webflow_item_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Append a completed run record and persist to disk."""
        now = datetime.now()
        record: Dict[str, Any] = {
            "run_id": now.strftime("%Y%m%d_%H%M%S"),
            "timestamp": now.isoformat(timespec="seconds"),
            "topic": topic,
            "slug": slug,
            "suggested_title": suggested_title,
            "webflow_item_id": webflow_item_id,
            "final_score": round(final_score, 2),
            "revision_count": revision_count,
            "evaluation_passed": evaluation_passed,
        }
        records = self._load()
        records.append(record)
        self._save(records)
        return record

    def all_records(self) -> List[Dict[str, Any]]:
        """Return all records, newest first."""
        return list(reversed(self._load()))

    def covered_topics(self) -> List[str]:
        """Return a list of previously published slugs and titles for deduplication."""
        return [
            f"{r['suggested_title']} (slug: {r['slug']})"
            for r in self._load()
            if r.get("suggested_title")
        ]

    def format_history_for_prompt(self, limit: int = 30) -> str:
        """Return a compact prompt-ready string listing recent posts and scores.

        Used by the research agent to:
        1. Avoid recommending already-covered topics.
        2. Detect patterns in what scores high vs. needs multiple revisions.
        """
        records = self._load()
        if not records:
            return "(No prior run history.)"

        # De-duplicate by suggested_title (keep the most recent record for each title).
        seen_titles: set = set()
        deduped: List[Dict[str, Any]] = []
        for r in reversed(records):  # newest first so we keep the latest occurrence
            title = r.get("suggested_title") or r.get("topic") or ""
            if title and title not in seen_titles:
                seen_titles.add(title)
                deduped.append(r)

        recent = deduped[:limit]  # cap to limit unique titles

        lines = [f"=== Content History — {len(recent)} unique post(s) covered ===", ""]

        for r in recent:  # already newest-first
            status = "✅" if r.get("evaluation_passed") else "⚠"
            score_str = f"{r.get('final_score', '?')}/10"
            rev_str = f"rev={r.get('revision_count', '?')}"
            lines.append(
                f"  {status} '{r.get('suggested_title', r.get('topic', 'Unknown'))}' "
                f"— score {score_str}, {rev_str}, {r.get('timestamp', '')[:10]}"
            )

        # Highlight topics that required revisions — useful signal for the researcher
        struggling = [r for r in records if r.get("revision_count", 0) >= 2]
        if struggling:
            lines += ["", "TOPICS THAT NEEDED MAX REVISIONS (hard to write — consider framing differently):"]
            for r in struggling[-5:]:
                lines.append(f"  '{r.get('suggested_title', r.get('topic', 'Unknown'))}'")

        # Build an explicit banned list for the LLM to check against.
        banned_titles = sorted(seen_titles)
        lines += [
            "",
            "⛔ BANNED TOPICS — you MUST NOT recommend any of these. This is a hard constraint:",
        ]
        for t in banned_titles:
            lines.append(f"  - {t}")
        lines += [
            "",
            "Your recommendation MUST be a topic not present in the banned list above.",
            "If the topic you are considering appears anywhere in the banned list, choose a different one.",
        ]
        return "\n".join(lines)
