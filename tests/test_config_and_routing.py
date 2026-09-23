"""Unit tests for graph routing functions and Settings/config utilities."""
from __future__ import annotations

import os
from unittest.mock import patch

from career_compass_content_engine.config import Settings, configure_logging
from career_compass_content_engine.supervisor import WorkflowResult


# ── Settings.configure_tracing ────────────────────────────────────────────────

class TestConfigureTracing:
    def test_sets_env_vars_when_key_present(self):
        settings = Settings(
            openai_api_key="sk-test",
            langsmith_api_key="ls__test",
            langsmith_project="my-project",
            langsmith_tracing=True,
        )
        with patch.dict(os.environ, {}, clear=False):
            # Remove vars to start clean
            os.environ.pop("LANGCHAIN_TRACING_V2", None)
            os.environ.pop("LANGCHAIN_API_KEY", None)
            os.environ.pop("LANGCHAIN_PROJECT", None)
            settings.configure_tracing()
            assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
            assert os.environ.get("LANGCHAIN_API_KEY") == "ls__test"
            assert os.environ.get("LANGCHAIN_PROJECT") == "my-project"

    def test_no_op_when_key_absent(self):
        settings = Settings(openai_api_key="sk-test", langsmith_api_key=None)
        before = os.environ.get("LANGCHAIN_TRACING_V2")
        settings.configure_tracing()
        assert os.environ.get("LANGCHAIN_TRACING_V2") == before

    def test_validate_raises_without_openai_key(self):
        settings = Settings(openai_api_key="")
        try:
            settings.validate()
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "OPENAI_API_KEY" in str(exc)

    def test_validate_passes_with_openai_key(self):
        settings = Settings(openai_api_key="sk-valid")
        settings.validate()  # should not raise


# ── configure_logging ─────────────────────────────────────────────────────────

class TestConfigureLogging:
    def test_does_not_raise(self):
        configure_logging("DEBUG")
        configure_logging("INFO")
        configure_logging("WARNING")

    def test_invalid_level_falls_back_to_info(self):
        # getattr(logging, "BADLEVEL", logging.INFO) should return INFO
        configure_logging("BADLEVEL")  # must not raise


# ── WorkflowResult defaults ───────────────────────────────────────────────────

class TestWorkflowResult:
    def test_default_token_fields_are_zero(self):
        result = WorkflowResult(
            research={},
            blog_post="",
            image_plan={},
            publish_result={},
        )
        assert result.total_tokens == 0
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_cost_usd == 0.0

    def test_run_id_is_auto_generated(self):
        r1 = WorkflowResult(research={}, blog_post="", image_plan={}, publish_result={})
        r2 = WorkflowResult(research={}, blog_post="", image_plan={}, publish_result={})
        assert r1.run_id != r2.run_id  # UUIDs are unique

    def test_explicit_run_id_preserved(self):
        result = WorkflowResult(
            research={}, blog_post="", image_plan={}, publish_result={}, run_id="fixed-id"
        )
        assert result.run_id == "fixed-id"
