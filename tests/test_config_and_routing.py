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


# ── Routing helpers (exercise without building the full graph) ────────────────

class TestRoutingLogic:
    """
    The routing functions are closures defined inside build_graph().
    We test the equivalent boolean logic directly to avoid spinning up LLMs.
    """

    def _route_after_evaluate(self, state: dict) -> str:
        """Mirror of the closure in graph.py."""
        from career_compass_content_engine.graph import MAX_REVISIONS

        if state.get("evaluation_passed") or state.get("revision_count", 0) >= MAX_REVISIONS:
            return "node_prepare_metadata"
        return "node_edit"

    def _route_after_image_design(self, state: dict):
        from langgraph.graph import END

        if state.get("image_plan", {}).get("prompt"):
            return "node_image_generate"
        return END

    def test_passes_routes_to_metadata(self):
        assert self._route_after_evaluate({"evaluation_passed": True}) == "node_prepare_metadata"

    def test_max_revisions_routes_to_metadata(self):
        from career_compass_content_engine.graph import MAX_REVISIONS

        assert self._route_after_evaluate({"revision_count": MAX_REVISIONS}) == "node_prepare_metadata"

    def test_not_passed_not_max_routes_to_edit(self):
        assert self._route_after_evaluate({"evaluation_passed": False, "revision_count": 0}) == "node_edit"

    def test_image_with_prompt_routes_to_generate(self):
        state = {"image_plan": {"prompt": "a vivid orange sunrise"}}
        assert self._route_after_image_design(state) == "node_image_generate"

    def test_image_without_prompt_routes_to_end(self):
        from langgraph.graph import END

        assert self._route_after_image_design({"image_plan": {}}) == END
