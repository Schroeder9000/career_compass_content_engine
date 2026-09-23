"""Unit tests for EvaluationAgent parsing, scoring, and feedback formatting."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from career_compass_content_engine.agents.evaluation_agent import (
    EvaluationAgent,
    EvaluationResult,
)


def _make_agent() -> EvaluationAgent:
    mock_llm = MagicMock()
    return EvaluationAgent(
        llm=mock_llm,
        name="Test Evaluator",
        system_prompt="You evaluate content.",
        score_threshold=8.5,
    )


PASSING_JSON = {
    "scores": {
        "word_count_depth": 9.0,
        "specific_examples": 9.0,
        "brand_voice": 9.0,
        "emotional_depth": 9.0,
        "actionability": 9.0,
        "writing_quality": 9.0,
    },
    "average_score": 9.0,
    "passes_threshold": True,
    "feedback": {
        "strengths": ["Great depth"],
        "critical_issues": [],
        "improvement_suggestions": [],
    },
    "recommended_action": "PUBLISH",
}

FAILING_JSON = {
    "scores": {
        "word_count_depth": 6.0,
        "specific_examples": 5.0,
        "brand_voice": 7.0,
        "emotional_depth": 6.0,
        "actionability": 7.0,
        "writing_quality": 6.0,
    },
    "average_score": 6.17,
    "passes_threshold": False,
    "feedback": {
        "strengths": [],
        "critical_issues": ["Too short", "Generic voice"],
        "improvement_suggestions": ["Add examples"],
    },
    "recommended_action": "REVISE",
}


class TestEvaluationAgentParsing:
    def test_parse_valid_json(self):
        agent = _make_agent()
        result = agent._parse_evaluation(json.dumps(PASSING_JSON))
        assert result["average_score"] == 9.0
        assert result["passes_threshold"] is True

    def test_parse_json_in_code_fence(self):
        agent = _make_agent()
        raw = f"```json\n{json.dumps(PASSING_JSON)}\n```"
        result = agent._parse_evaluation(raw)
        assert result["recommended_action"] == "PUBLISH"

    def test_parse_empty_returns_default(self):
        agent = _make_agent()
        result = agent._parse_evaluation("")
        assert result["average_score"] == 5.0
        assert result["passes_threshold"] is False

    def test_parse_garbage_returns_default(self):
        agent = _make_agent()
        result = agent._parse_evaluation("not json at all")
        assert "scores" in result
        assert result["recommended_action"] == "REVISE"


class TestEvaluationAgentEvaluate:
    def test_passing_content(self):
        agent = _make_agent()
        with patch.object(agent, "run") as mock_run:
            mock_run.return_value = MagicMock(raw_response=json.dumps(PASSING_JSON))
            result = agent.evaluate("some content")
        assert isinstance(result, EvaluationResult)
        assert result.passes_threshold is True
        assert result.average_score == 9.0
        assert result.recommended_action == "PUBLISH"

    def test_failing_content(self):
        agent = _make_agent()
        with patch.object(agent, "run") as mock_run:
            mock_run.return_value = MagicMock(raw_response=json.dumps(FAILING_JSON))
            result = agent.evaluate("short content")
        assert result.passes_threshold is False
        assert result.recommended_action == "REVISE"
        assert len(result.feedback["critical_issues"]) == 2

    def test_scores_dict_present(self):
        agent = _make_agent()
        with patch.object(agent, "run") as mock_run:
            mock_run.return_value = MagicMock(raw_response=json.dumps(PASSING_JSON))
            result = agent.evaluate("content")
        assert set(result.scores.keys()) == {
            "word_count_depth",
            "specific_examples",
            "brand_voice",
            "emotional_depth",
            "actionability",
            "writing_quality",
        }


class TestFormatFeedback:
    def test_format_contains_score_header(self):
        agent = _make_agent()
        with patch.object(agent, "run") as mock_run:
            mock_run.return_value = MagicMock(raw_response=json.dumps(FAILING_JSON))
            result = agent.evaluate("content")
        feedback_str = agent.format_feedback_for_editor(result)
        assert "EVALUATION FEEDBACK" in feedback_str
        assert "CRITICAL ISSUES" in feedback_str
        assert "Too short" in feedback_str

    def test_format_passes_does_not_raise(self):
        agent = _make_agent()
        with patch.object(agent, "run") as mock_run:
            mock_run.return_value = MagicMock(raw_response=json.dumps(PASSING_JSON))
            result = agent.evaluate("content")
        feedback_str = agent.format_feedback_for_editor(result)
        assert "PASSES" in feedback_str


@pytest.mark.parametrize("mutation", [
    lambda x: x.update(average_score=10, scores={k: 2 for k in x["scores"]}),
    lambda x: x["scores"].update(writing_quality=float("nan")),
    lambda x: x["scores"].update(writing_quality=True),
    lambda x: x["scores"].pop("writing_quality"),
    lambda x: x["feedback"].update(critical_issues=["Unsupported product claim"]),
    lambda x: x.update(recommended_action="REVISE"),
])
def test_evaluation_does_not_trust_claimed_average_or_approval(mutation):
    from copy import deepcopy
    payload = deepcopy(PASSING_JSON)
    mutation(payload)
    agent = _make_agent()
    with patch.object(agent, "run", return_value=MagicMock(raw_response=json.dumps(payload))):
        assert agent.evaluate("article").passes_threshold is False


@pytest.mark.parametrize("payload", [[], None, {"scores": []}, "not an evaluation"])
def test_invalid_evaluation_structure_is_rejected(payload):
    agent = _make_agent()
    with patch.object(agent, "run", return_value=MagicMock(raw_response=json.dumps(payload))):
        result = agent.evaluate("article")
    assert result.passes_threshold is False
    assert result.feedback["critical_issues"]
