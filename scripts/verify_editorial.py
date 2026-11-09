"""Read-only production-history check and unpublished editorial model preview.

Run from the repository root with PYTHONPATH=src modal run scripts/verify_editorial.py.
Reuses the existing Modal secret. Never calls CMS writes, image generation,
publication, history append, or notification functions.
"""
import json
from pathlib import Path

import modal
from career_compass_content_engine.modal_app import image, cache_volume

app = modal.App("career-compass-editorial-verification")


@app.function(image=image.add_local_dir("tests", "/qa/tests", copy=True), timeout=600)
def verify_tests():
    import subprocess
    completed = subprocess.run(["python", "-m", "pytest", "/qa/tests", "-q"],
                               cwd="/app", check=True, capture_output=True, text=True)
    return {"tests_passed": completed.returncode == 0, "output": completed.stdout}


@app.function(image=image, secrets=[modal.Secret.from_name("careeer_compass_langchain_content_engine")],
              volumes={"/app/.cache": cache_volume}, timeout=900)
def preview():
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime, timezone
    from career_compass_content_engine.agents.base import AgentContext
    from career_compass_content_engine.agents.research_agent import ResearchAgent
    from career_compass_content_engine.agents.writing_agent import WritingAgent
    from career_compass_content_engine.agents.editing_agent import EditingAgent
    from career_compass_content_engine.agents.evaluation_agent import EvaluationAgent
    from career_compass_content_engine.agents.publishing_agent import PublishingAgent
    from career_compass_content_engine.config import get_settings
    from career_compass_content_engine.editorial_policy import (
        brief_for, choose_editorial_brief, published_history,
        research_plan_issues, article_editorial_issues,
    )
    from career_compass_content_engine.graph import _safe_parse, _check_required_ending
    from career_compass_content_engine.publication_policy import metadata_blockers
    from career_compass_content_engine.supervisor import build_llm
    from career_compass_content_engine.tools.topic_store import TopicStore

    settings = get_settings()
    history_path = Path("/app/.cache/topic_history.json")
    before = history_path.read_bytes() if history_path.exists() else None
    history = TopicStore(history_path).all_records()
    next_brief = choose_editorial_brief(history)
    context = AgentContext(
        brand_voice="Supportive, practical, and optimistic career guidance with tactical steps.",
        brand_colors="Orange, dark grey and white",
        target_audience="Early-career professionals facing a broad range of workplace challenges.",
        distribution_channels="Webflow blog, LinkedIn and email newsletter",
    )
    existing = json.dumps([{k: r.get(k) for k in ("suggested_title", "topic", "slug")}
                           for r in published_history(history)[:30]], ensure_ascii=False)
    briefs = [next_brief, brief_for("work_habits"), brief_for("workplace_change"),
              brief_for("advancement_and_pay")]

    def recommend(brief):
        researcher = ResearchAgent(llm=build_llm(settings, "research"), name="Preview researcher",
                                   system_prompt="Identify useful, distinctive early-career topics.")
        plan = _safe_parse(researcher.recommend(existing, context, editorial_brief=brief).raw_response)
        issues = research_plan_issues(plan, brief)
        assert not issues, issues
        plan["editorial_brief"] = brief
        return plan

    with ThreadPoolExecutor(max_workers=4) as pool:
        proposals = list(pool.map(recommend, briefs))
    print("Topic proposals checked: " + "; ".join(p["suggested_title"] for p in proposals), flush=True)
    plan = proposals[0]
    writer = WritingAgent(llm=build_llm(settings, "writing"), name="Preview writer",
                          system_prompt="Craft useful, distinctive Career Compass articles.")
    draft = writer.draft(plan, "", context).raw_response
    editor = EditingAgent(llm=build_llm(settings, "editing"), name="Preview editor",
                          system_prompt="Polish drafts to publication quality.")
    article = editor.refine(draft, editorial_brief=next_brief).raw_response
    issues = article_editorial_issues(article, next_brief) + _check_required_ending(article)
    assert not issues, issues
    print(f"Independent article checked: {len(article.split())} words; no forced product pitch", flush=True)
    evaluator = EvaluationAgent(llm=build_llm(settings, "evaluation"), name="Preview evaluator",
                                system_prompt="Evaluate publication quality.")
    evaluation = evaluator.evaluate(article, editorial_brief=next_brief)
    publisher = PublishingAgent(llm=build_llm(settings, "metadata"), name="Preview metadata QA",
                                system_prompt="Check metadata and publication readiness.", settings=settings)
    metadata = _safe_parse(publisher.prepare_metadata(article, editorial_brief=next_brief).raw_response)
    metadata_issues = metadata_blockers(metadata)
    assert (history_path.read_bytes() if history_path.exists() else None) == before, "Preview changed history"
    return {"checked_at": datetime.now(timezone.utc).isoformat(), "next_brief": next_brief,
            "published_history_count": len(published_history(history)), "proposals": proposals,
            "article": article, "word_count": len(article.split()), "editorial_issues": issues,
            "evaluation": evaluation.raw_json, "metadata": metadata, "metadata_issues": metadata_issues,
            "history_unchanged": True, "cms_writes": False, "published": False}


@app.local_entrypoint()
def main(output: str = "/tmp/career-compass-editorial-preview.json"):
    tests = verify_tests.remote()
    print(tests["output"])
    result = preview.remote()
    result["test_result"] = tests
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": output, "next_coverage": result["next_brief"]["coverage"],
                      "editorial_issues": result["editorial_issues"],
                      "evaluation_passed": result["evaluation"]["passes_threshold"],
                      "metadata_issues": result["metadata_issues"], "published": False}))
