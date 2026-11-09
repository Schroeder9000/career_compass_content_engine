"""Exercise the real publisher boundary with mocked HTTP and real graph routing."""
import asyncio
import json
from datetime import datetime, timezone
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

from career_compass_content_engine.agents.base import AgentContext, AgentResult
from career_compass_content_engine.agents.publishing_agent import PublishingAgent, WebflowCMS
from career_compass_content_engine.config import Settings
from career_compass_content_engine.publication_policy import metadata_blockers

METADATA = {"title": "Prepare a clear case", "slug": "prepare-a-clear-case",
            "summary": "Gather clear examples of your contribution and results before your next career conversation.",
            "content_ready": True, "blocking_issues": []}
BODY = "# Prepare a clear case\n\n## Start with evidence\n\nWrite down the result and your contribution."
SCHEMA = [{"slug": slug, "type": kind, "isRequired": kind == "Image"} for slug, kind in {
    "name": "PlainText", "slug": "PlainText", "summary": "PlainText",
    "blog-post-richt-text": "RichText", "blog-post-featured-image-photo": "Image",
    "blog-post-thumbnail-image-photo": "Image", "blog-post-author": "Reference",
    "blog-post-category": "Reference",
}.items()]


def publisher(monkeypatch, mode="live"):
    cms = MagicMock(spec=WebflowCMS)
    cms.get_collection_fields.return_value = deepcopy(SCHEMA)
    cms.find_item_by_slug.return_value = None
    cms.create_draft_blog_item.return_value = {"id": "new-item", "isDraft": True,
        "fieldData": {"slug": METADATA["slug"]}}
    cms.publish_items.return_value = {"publishedItemIds": ["new-item"], "errors": []}
    cms._request_with_retries.return_value.json.return_value = {"id": "new-item", "isDraft": False}
    monkeypatch.setattr("career_compass_content_engine.agents.publishing_agent.WebflowCMS", lambda token: cms)
    agent = PublishingAgent(name="QA", system_prompt="", llm=MagicMock(), settings=Settings(
        openai_api_key="test", webflow_api_token="test", webflow_collection_id="collection",
        webflow_site_id="site", publish_mode=mode))
    return agent, cms


@pytest.mark.parametrize("mode,expected", [("live", "success"), ("draft", "draft")])
def test_item_boundary_and_preserved_cms_fields(monkeypatch, mode, expected):
    agent, cms = publisher(monkeypatch, mode)
    result = agent.upload_to_webflow(METADATA, BODY, image_url="https://example.com/approved.png")
    assert result["status"] == expected
    kwargs = cms.create_draft_blog_item.call_args.kwargs
    assert kwargs["field_slug_article"] == "blog-post-richt-text"
    assert kwargs["field_slug_author"] == "blog-post-author"
    assert kwargs["author_item_id"] == "6759b39c1464ba09dc7d783d"
    assert kwargs["extra_field_data"]["blog-post-category"] == "6759b39c1464ba09dc7d785e"
    assert "<h1" not in kwargs["article_html"]
    assert "<h2>Start with evidence</h2>" in kwargs["article_html"]
    assert result["url"] == "https://careeraiapp.com/blog/prepare-a-clear-case"
    if mode == "live":
        cms.publish_items.assert_called_once_with("collection", ["new-item"])
    else:
        cms.publish_items.assert_not_called()
    assert not hasattr(WebflowCMS, "publish_site")


def test_generated_image_publishes_responsive_webp_and_preserves_item_boundary(monkeypatch, tmp_path):
    from PIL import Image
    agent, cms = publisher(monkeypatch)
    cms.get_collection_fields.return_value.extend([
        {"slug": key, "type": "PlainText"}
        for key in ("featured-image-srcset", "thumbnail-image-srcset")])
    cms.get_collection_fields.return_value.extend([
        {"slug": f"{prefix}-image-{dimension}", "type": "Number"}
        for prefix in ("featured", "thumbnail") for dimension in ("width", "height")])
    source = tmp_path / "generated.png"
    Image.new("RGB", (1536, 1024), "orange").save(source)
    cms.upload_asset.side_effect = lambda site, path: {"url": "https://cdn.example.com/" + path.name}
    result = agent.upload_to_webflow(METADATA, BODY, image_file_path=str(source))
    assert result["status"] == "success"
    assert cms.upload_asset.call_count == 4
    args = cms.create_draft_blog_item.call_args.kwargs
    assert args["main_image_url"]["url"].endswith("generated.webp")
    assert args["thumb_image_url"]["url"].endswith("-800w.webp")
    assert "1536w" in args["extra_field_data"]["featured-image-srcset"]
    assert args["extra_field_data"]["featured-image-srcset"] == args["extra_field_data"]["thumbnail-image-srcset"]
    assert args["extra_field_data"]["featured-image-width"] == 1536
    assert args["extra_field_data"]["featured-image-height"] == 1024
    assert args["extra_field_data"]["thumbnail-image-width"] == 800
    assert args["extra_field_data"]["thumbnail-image-height"] == 533
    cms.publish_items.assert_called_once_with("collection", ["new-item"])


def test_missing_responsive_schema_prevents_image_upload_and_publication(monkeypatch, tmp_path):
    agent, cms = publisher(monkeypatch)
    result = agent.upload_to_webflow(METADATA, BODY, image_file_path=str(tmp_path / "generated.png"))
    assert result["status"] == "error"
    cms.upload_asset.assert_not_called()
    cms.create_draft_blog_item.assert_not_called()
    cms.publish_items.assert_not_called()


def test_webp_upload_mime_is_explicit_even_without_os_mime_database(monkeypatch, tmp_path):
    source = tmp_path / "optimized.webp"
    source.write_bytes(b"synthetic upload fixture")
    cms = WebflowCMS("test")
    cms.create_asset_metadata = MagicMock(return_value={
        "uploadUrl": "https://upload.example.com/",
        "uploadDetails": {"key": "asset"},
        "hostedUrl": "https://s3.amazonaws.com/webflow-prod-assets/site/optimized.webp"})
    post = MagicMock()
    monkeypatch.setattr("career_compass_content_engine.agents.publishing_agent.requests.post", post)
    monkeypatch.setattr("career_compass_content_engine.agents.publishing_agent.mimetypes.guess_type", lambda p: (None, None))
    result = cms.upload_asset("site", source)
    assert post.call_args.kwargs["files"]["file"][2] == "image/webp"
    assert result["url"] == "https://cdn.prod.website-files.com/site/optimized.webp"


@pytest.mark.parametrize("update", [
    {"content_ready": False}, {"content_ready": "true"}, {"content_ready": None},
    {"blocking_issues": ["Unsupported product claim"]}, {"blocking_issues": None},
    {"title": ""}, {"summary": ""},
])
def test_failed_metadata_never_writes_cms(monkeypatch, update):
    agent, cms = publisher(monkeypatch)
    result = agent.upload_to_webflow({**METADATA, **update}, BODY)
    assert result["status"] == "blocked"
    cms.get_collection_fields.assert_not_called()
    cms.create_draft_blog_item.assert_not_called()
    cms.publish_items.assert_not_called()


@pytest.mark.parametrize("fault", ["schema", "lookup", "duplicate", "image"])
def test_uncertain_or_invalid_input_does_not_create(monkeypatch, fault):
    agent, cms = publisher(monkeypatch)
    if fault == "schema":
        cms.get_collection_fields.return_value = []
    elif fault == "lookup":
        cms.find_item_by_slug.side_effect = RuntimeError("unavailable")
    elif fault == "duplicate":
        cms.find_item_by_slug.return_value = {"id": "existing"}
    result = agent.upload_to_webflow(METADATA, BODY,
        image_url=None if fault == "image" else "https://example.com/image.png")
    assert result["status"] in {"error", "blocked"}
    cms.create_draft_blog_item.assert_not_called()
    cms.publish_items.assert_not_called()


def test_create_timeout_does_not_retry_non_idempotent_post():
    cms = WebflowCMS("test")
    cms.session = MagicMock()
    cms.session.request.side_effect = requests.Timeout("uncertain write")
    with pytest.raises(RuntimeError):
        cms._request_with_retries("POST", "https://api.webflow.com/v2/collections/test/items")
    assert cms.session.request.call_count == 1


@pytest.mark.parametrize("mode", ["live", "draft"])
def test_publication_date_is_only_assigned_for_live_articles(monkeypatch, mode):
    agent, cms = publisher(monkeypatch, mode)
    cms.get_collection_fields.return_value.append({"slug": "publication-date", "type": "DateTime"})
    before = datetime.now(timezone.utc)
    result = agent.upload_to_webflow(METADATA, BODY, image_url="https://example.com/image.png")
    assert result["status"] == ("success" if mode == "live" else "draft")
    extra = cms.create_draft_blog_item.call_args.kwargs["extra_field_data"]
    if mode == "live":
        assert before <= datetime.fromisoformat(extra["publication-date"]) <= datetime.now(timezone.utc)
    else:
        assert "publication-date" not in extra


def test_incompatible_publication_date_field_blocks_write(monkeypatch):
    agent, cms = publisher(monkeypatch)
    cms.get_collection_fields.return_value.append({"slug": "publication-date", "type": "PlainText"})
    result = agent.upload_to_webflow(METADATA, BODY, image_url="https://example.com/image.png")
    assert result["status"] == "error"
    assert "publication-date" in result["reason"]
    cms.create_draft_blog_item.assert_not_called()


def test_publish_api_partial_failure_is_error():
    cms = WebflowCMS("test")
    response = MagicMock(status_code=202)
    response.json.return_value = {"publishedItemIds": [], "errors": ["not published"]}
    cms._request_with_retries = MagicMock(return_value=response)
    with pytest.raises(RuntimeError, match="missing_published_ids"):
        cms.publish_items("collection", ["new-item"])


@pytest.mark.parametrize("score,ready,expected", [(6.0, True, "blocked"), (9.0, False, "blocked"),
                                                   (9.0, True, "success")])
def test_compiled_graph_gates_real_publication_route(monkeypatch, tmp_path, score, ready, expected):
    from career_compass_content_engine import graph as module
    monkeypatch.setenv("TOPIC_HISTORY_PATH", str(tmp_path / "history.json"))
    monkeypatch.setattr("career_compass_content_engine.supervisor.build_llm", lambda *args: MagicMock())
    draft = module._append_fallback_closing(BODY)
    answer = lambda content: AgentResult(content={"message": content}, raw_response=content)
    monkeypatch.setattr(module.ResearchAgent, "recommend", lambda *args, **kwargs: answer(
        json.dumps({"topic": "evidence", "suggested_title": METADATA["title"]})))
    monkeypatch.setattr(module.WritingAgent, "draft", lambda *args: answer(draft))
    monkeypatch.setattr(module.EditingAgent, "refine", lambda *args, **kwargs: answer(draft))
    monkeypatch.setattr(module.ImageAgent, "design", lambda *args, **kwargs: answer("{}"))
    evaluate = MagicMock(return_value=SimpleNamespace(average_score=score, passes_threshold=score >= 8.5,
        scores={"writing_quality": score}, feedback={"critical_issues": [] if score >= 8.5 else ["Revise"]}, raw_json={}))
    monkeypatch.setattr(module.EvaluationAgent, "evaluate", evaluate)
    monkeypatch.setattr(module.EvaluationAgent, "format_feedback_for_editor", lambda *args: "Revise")
    metadata = MagicMock(return_value=answer(json.dumps({**METADATA, "content_ready": ready})))
    monkeypatch.setattr(module.PublishingAgent, "prepare_metadata", metadata)
    upload = MagicMock(return_value={"status": "success", "slug": "real-slug", "webflow_item_id": "new-item"})
    monkeypatch.setattr(module.PublishingAgent, "upload_to_webflow", upload)
    rag = MagicMock()
    rag.as_image_context_with_sources.return_value = ("", [])
    rag.as_text_context.return_value = ""
    graph = module.build_graph(Settings(openai_api_key="test"), rag)
    result = asyncio.run(graph.ainvoke({"existing_content": "", "context": AgentContext("", "", "", "")},
        config={"configurable": {"thread_id": "test"}}))
    assert result["publish_result"]["status"] == expected
    if expected == "blocked":
        upload.assert_not_called()
        saved = list((tmp_path / "blocked").glob("*.json"))
        assert len(saved) == 1
        assert json.loads(saved[0].read_text())["article"] == draft
    else:
        upload.assert_called_once()
    assert evaluate.call_count == (3 if score < 8.5 else 1)
    if score < 8.5:
        metadata.assert_not_called()
