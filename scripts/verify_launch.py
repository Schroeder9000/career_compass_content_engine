"""Explicit launch checks. No article workflow or OpenAI requests are invoked.

Run from the repository root:
    PYTHONPATH=src modal run scripts/verify_launch.py
Creates one clearly named unpublished CMS verification draft, once per slug.
"""
import modal
from career_compass_content_engine.modal_app import image

app = modal.App("career-compass-launch-verification")


@app.function(image=image.add_local_dir("tests", "/qa/tests", copy=True), timeout=600)
def verify_tests():
    import subprocess
    # This function has no production secrets attached.
    completed = subprocess.run(["python", "-m", "pytest", "/qa/tests", "-q"],
                               cwd="/app", check=True)
    return {"tests_passed": completed.returncode == 0}


@app.function(image=image, secrets=[modal.Secret.from_name("careeer_compass_langchain_content_engine")],
              timeout=180)
def verify_draft():
    from dataclasses import replace
    from career_compass_content_engine.config import get_settings
    from career_compass_content_engine.agents.publishing_agent import PublishingAgent, WebflowCMS

    settings = replace(get_settings(), publish_mode="draft")
    cms = WebflowCMS(settings.webflow_api_token)
    slug = "launch-verification-promotion-case-2026-09-23"
    existing = cms.find_item_by_slug(settings.webflow_collection_id, slug)
    if existing:
        assert existing.get("isDraft") is True, "Verification item must remain an unpublished draft"
        return {"status": "draft", "item_id": existing["id"], "slug": slug, "already_exists": True}
    source = next(item for item in cms.list_items(settings.webflow_collection_id)["items"]
                  if not item.get("isDraft") and not item.get("isArchived")
                  and item.get("fieldData", {}).get("blog-post-featured-image-photo"))
    source_image = source["fieldData"]["blog-post-featured-image-photo"]["url"]
    agent = PublishingAgent(name="Launch verification", system_prompt="", llm=None, settings=settings)
    result = agent.upload_to_webflow(
        {"title": "Launch verification: Promotion Case", "slug": slug,
         "summary": "An unpublished verification draft for the Promotion Case website launch. It validates CMS fields and individual-item publishing safeguards.",
         "content_ready": True, "blocking_issues": [], "image_alt": "Existing editorial illustration reused for unpublished verification"},
        "# Launch verification: Promotion Case\n\n"
        "## Unpublished operational check\n\n"
        "This is a synthetic verification draft, not an article for publication. "
        "Promotion Case by Career Compass prepares a manager-facing Promotion Case PDF "
        "and a separate private Conversation Guide PDF. The US purchase is $24.99 once "
        "for one case per account. No promotion or raise is guaranteed.",
        image_url=source_image,
    )
    assert result["status"] == "draft", result.get("reason", "Draft check failed")
    assert result["webflow_item"]["isDraft"] is True
    assert "webflow_publish" not in result
    return {"status": result["status"], "item_id": result["webflow_item_id"], "slug": result["slug"],
            "image_attached": result["image_attached"], "publish_called": False}


@app.local_entrypoint()
def main():
    print(verify_tests.remote())
    print(verify_draft.remote())
