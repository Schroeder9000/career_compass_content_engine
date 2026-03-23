"""Unit tests for TopicStore — the persistent run-history JSON log."""
from __future__ import annotations

from pathlib import Path

import pytest

from career_compass_content_engine.tools.topic_store import TopicStore


@pytest.fixture
def store(tmp_path: Path) -> TopicStore:
    """Return a TopicStore backed by a temp file that is wiped between tests."""
    return TopicStore(path=tmp_path / "topic_history.json")


class TestTopicStoreAppendAndLoad:
    def test_append_creates_file(self, store: TopicStore):
        store.append(
            topic="salary negotiation",
            slug="salary-negotiation",
            suggested_title="How to Negotiate Salary",
            final_score=9.0,
            revision_count=1,
            evaluation_passed=True,
        )
        assert store.path.exists()

    def test_appended_record_is_readable(self, store: TopicStore):
        store.append(
            topic="career pivot",
            slug="career-pivot",
            suggested_title="Making Your Career Pivot",
            final_score=8.7,
            revision_count=0,
            evaluation_passed=True,
        )
        records = store._load()
        assert len(records) == 1
        assert records[0]["topic"] == "career pivot"
        assert records[0]["final_score"] == 8.7

    def test_multiple_appends_accumulate(self, store: TopicStore):
        for i in range(3):
            store.append(
                topic=f"topic {i}",
                slug=f"slug-{i}",
                suggested_title=f"Title {i}",
                final_score=8.0 + i * 0.1,
                revision_count=0,
                evaluation_passed=True,
            )
        assert len(store._load()) == 3

    def test_append_with_webflow_id(self, store: TopicStore):
        store.append(
            topic="negotiation",
            slug="negotiation",
            suggested_title="Negotiate",
            final_score=9.5,
            revision_count=0,
            evaluation_passed=True,
            webflow_item_id="abc123",
        )
        record = store._load()[0]
        assert record["webflow_item_id"] == "abc123"

    def test_load_returns_empty_list_when_file_missing(self, store: TopicStore):
        assert store._load() == []


class TestTopicStoreFormatHistory:
    def test_empty_store_returns_empty_string_or_header(self, store: TopicStore):
        result = store.format_history_for_prompt()
        # Either empty string or a header with no items — must not raise
        assert isinstance(result, str)

    def test_format_includes_topic_titles(self, store: TopicStore):
        store.append(
            topic="performance reviews",
            slug="performance-reviews",
            suggested_title="Surviving Performance Reviews",
            final_score=9.1,
            revision_count=1,
            evaluation_passed=True,
        )
        history = store.format_history_for_prompt()
        assert "performance reviews" in history.lower() or "surviving performance reviews" in history.lower()
