import asyncio
import json
from collections import Counter
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from career_compass_content_engine.editorial_policy import (
    ADVANCEMENT, PILLARS, article_editorial_issues, brief_for,
    choose_editorial_brief, published_history, research_plan_issues,
)
from career_compass_content_engine.agents.base import AgentContext, AgentResult
from career_compass_content_engine.config import Settings
from tests.test_publication_safety import BODY, COMPLETE_ENDING, METADATA


def record(pillar, n, status="success"):
    return {"slug": f"article-{n}", "webflow_item_id": f"id-{n}", "editorial_pillar": pillar,
            "publish_status": status, "suggested_title": f"Useful article {n}"}


def test_rotation_covers_all_pillars_and_limits_advancement_in_every_window():
    history = []
    for n in range(100):
        brief = choose_editorial_brief(history)
        assert not (brief["pillar"] == ADVANCEMENT and history and history[0]["editorial_pillar"] == ADVANCEMENT)
        history.insert(0, record(brief["pillar"], n))
        assert Counter(r["editorial_pillar"] for r in history[:10])[ADVANCEMENT] <= 2
    assert set(r["editorial_pillar"] for r in history) == set(PILLARS)


def test_drafts_errors_and_duplicate_runs_do_not_consume_editorial_slots():
    history = [record("first_jobs", 1)]
    polluted = [record(ADVANCEMENT, 2, "draft"), record("skills_and_learning", 3, "error"),
                *history, *history]
    assert published_history(polluted) == history
    assert choose_editorial_brief(polluted) == choose_editorial_brief(history)


def test_legacy_promotion_streak_cannot_schedule_another_promotion_post():
    history = [{"suggested_title": f"Build your promotion case {i}", "slug": f"promotion-{i}",
                "webflow_item_id": str(i), "evaluation_passed": True} for i in range(10)]
    for i in range(8):
        brief = choose_editorial_brief(history)
        assert brief["pillar"] != ADVANCEMENT
        history.insert(0, record(brief["pillar"], 100+i))


def test_research_must_stay_in_the_assigned_broad_pillar():
    brief = brief_for("first_jobs")
    assert research_plan_issues({"topic": "promotion", "suggested_title": "Build Your Promotion Case"}, brief)
    assert not research_plan_issues({"topic": "asking questions", "suggested_title": "Ask Better Questions in a New Job"}, brief)


def test_healthy_ending_without_brand_or_product_is_accepted():
    from career_compass_content_engine.graph import _check_required_ending
    article = BODY + COMPLETE_ENDING
    assert not _check_required_ending(article)
    assert not article_editorial_issues(article, brief_for("first_jobs"))


@pytest.mark.parametrize("pitch", [
    "Promotion Case helps you prepare your next conversation.",
    "Try [this tool](https://careeraiapp.com/promotion-case).",
    "Try [the app](https://apps.apple.com/us/app/promotion-case-career-compass/id6670568018).",
])
def test_product_pitch_rejected_in_a_broad_article(pitch):
    assert article_editorial_issues(BODY + COMPLETE_ENDING + "\n\n" + pitch, brief_for("work_habits"))


def test_relevant_short_product_connection_is_optional_and_bounded():
    brief = brief_for(ADVANCEMENT)
    assert not article_editorial_issues(BODY + COMPLETE_ENDING, brief)
    pitch = "Promotion Case by Career Compass can help organize evidence for your conversation."
    assert not article_editorial_issues(BODY + COMPLETE_ENDING + "\n\n" + pitch, brief)
    assert article_editorial_issues(BODY + COMPLETE_ENDING + "\n\n" + pitch + " details"*70, brief)
    assert article_editorial_issues(BODY + "\n\n" + pitch + "\n\n" + pitch, brief)


@pytest.mark.parametrize("scenario", ["no_brand", "unrepaired_ending", "repair_inserts_pitch"])
def test_compiled_graph_accepts_independent_articles_and_blocks_bad_repairs(monkeypatch, tmp_path, scenario):
    from career_compass_content_engine import graph as module
    monkeypatch.setenv("TOPIC_HISTORY_PATH", str(tmp_path / "history.json"))
    monkeypatch.setattr("career_compass_content_engine.supervisor.build_llm", lambda *args: MagicMock())
    answer = lambda text: AgentResult(content={"message": text}, raw_response=text)
    monkeypatch.setattr(module.ResearchAgent, "recommend", lambda *args, **kwargs: answer(json.dumps(
        {"topic": "learning a new role", "suggested_title": "Ask Better Questions in a New Job"})))
    original = BODY + COMPLETE_ENDING if scenario == "no_brand" else BODY
    monkeypatch.setattr(module.WritingAgent, "draft", lambda *args: answer(original))
    def edit(*args, **kwargs):
        if scenario == "repair_inserts_pitch" and "HARD REQUIREMENTS" in (kwargs.get("feedback") or ""):
            return answer(BODY + COMPLETE_ENDING + "\n\nPromotion Case by Career Compass can help.")
        return answer(original)
    monkeypatch.setattr(module.EditingAgent, "refine", edit)
    monkeypatch.setattr(module.ImageAgent, "design", lambda *args, **kwargs: answer("{}"))
    monkeypatch.setattr(module.EvaluationAgent, "evaluate", lambda *args, **kwargs: SimpleNamespace(
        average_score=9.0, passes_threshold=True, scores={"writing_quality": 9.0},
        feedback={"critical_issues": []}, raw_json={}))
    metadata = MagicMock(return_value=answer(json.dumps(METADATA)))
    monkeypatch.setattr(module.PublishingAgent, "prepare_metadata", metadata)
    upload = MagicMock(return_value={"status":"success", "slug":"new-job", "webflow_item_id":"new-item"})
    monkeypatch.setattr(module.PublishingAgent, "upload_to_webflow", upload)
    rag = MagicMock(); rag.as_image_context_with_sources.return_value = ("", []); rag.as_text_context.return_value = ""
    graph = module.build_graph(Settings(openai_api_key="test"), rag)
    result = asyncio.run(graph.ainvoke({"existing_content":"", "context":AgentContext("", "", "", "")},
                                      config={"configurable":{"thread_id":"editorial-check"}}))
    if scenario == "no_brand":
        assert result["publish_result"]["status"] == "success"
        assert result["edited_content"] == original
        upload.assert_called_once()
        saved_history = json.loads((tmp_path / "history.json").read_text())
        assert saved_history[0]["editorial_pillar"] == "first_jobs"
    else:
        assert result["publish_result"]["status"] == "blocked"
        upload.assert_not_called(); metadata.assert_not_called()
        if scenario == "unrepaired_ending":
            assert result["edited_content"] == original
            assert "Promotion Case" not in result["edited_content"]
