"""Unit tests for the _safe_parse helper in graph.py."""
from __future__ import annotations

import json
import pytest

from career_compass_content_engine.graph import _safe_parse


class TestSafeParse:
    def test_valid_json_object(self):
        data = {"topic": "salary negotiation", "score": 9.0}
        assert _safe_parse(json.dumps(data)) == data

    def test_json_wrapped_in_code_fence(self):
        payload = {"key": "value"}
        raw = f"```json\n{json.dumps(payload)}\n```"
        result = _safe_parse(raw)
        assert result == payload

    def test_json_embedded_in_prose(self):
        payload = {"title": "Test", "outline": ["section 1"]}
        raw = f"Here is the result:\n{json.dumps(payload)}\nEnd of response."
        assert _safe_parse(raw) == payload

    def test_empty_string_returns_empty_dict(self):
        assert _safe_parse("") == {}

    def test_none_like_empty_returns_empty_dict(self):
        # Internally the function guards on `if not raw`
        assert _safe_parse("   ") == {}  # whitespace-only → json.loads fails → regex finds nothing

    def test_invalid_json_with_no_braces_returns_empty_dict(self):
        assert _safe_parse("This is just a plain sentence.") == {}

    def test_nested_object_preserved(self):
        data = {"scores": {"a": 9.0, "b": 8.5}, "passes": True}
        assert _safe_parse(json.dumps(data)) == data
