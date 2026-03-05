"""
Career Compass Content Pipeline — LangGraph StateGraph

Graph topology
--------------
                              ┌─── node_write ─── node_edit ─┐
                              │                               ▼
START ── node_research ───────┤                         node_evaluate
                              │                         (loop / done)
                              └─── node_image_design                │
                                         │                          ▼
                                   [has prompt?]           node_prepare_metadata
                                         │                          │
                               node_image_generate → END    node_publish → END

Parallel branches
-----------------
After node_research, the CONTENT branch (write → edit → evaluate) and the IMAGE
branch (image_design → image_generate) are queued in the same superstep, so they
run back-to-back without waiting for each other. The image branch is always shorter
(one design call + one generation call) and will finish before the content loop's
first evaluation cycle completes. By the time node_prepare_metadata fires, both
edited_content and image_file_path are already in shared state.

To unlock true I/O-concurrent execution (both branches making LLM calls at the
same time), define nodes as `async def` and call `graph.ainvoke()` instead of
`graph.invoke()`.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from .agents.editing_agent import EditingAgent
from .agents.evaluation_agent import EvaluationAgent
from .agents.image_agent import ImageAgent
from .agents.image_generator import ImageGenerator
from .agents.publishing_agent import PublishingAgent
from .agents.research_agent import ResearchAgent
from .agents.writing_agent import WritingAgent
from .config import Settings, get_settings
from .rag import RAGPipeline
from .state import ContentState

# ── Pipeline constants (single source of truth) ───────────────────────────────
MAX_REVISIONS: int = 2        # Max evaluation cycles; 1 edit revision will occur
SCORE_THRESHOLD: float = 8.5  # Average score (0–10) required to pass without revision


# ── Internal helpers ──────────────────────────────────────────────────────────
def _safe_parse(raw: str) -> Dict[str, Any]:
    """Parse a JSON string, stripping code fences and falling back to regex."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}


def _augment_image_prompt(image_plan: Dict[str, Any]) -> str:
    """Append spatial/composition instructions to the raw image prompt."""
    prompt_text = image_plan.get("prompt", "")
    if image_plan.get("aspect_ratio"):
        return (
            f"WIDE CINEMATIC PANORAMA in {image_plan.get('aspect_ratio')} format. "
            "COMPOSITION STRUCTURE: Divide the canvas into THREE ZONES - "
            "LEFT THIRD has visual elements, CENTER THIRD has visual elements, "
            "RIGHT THIRD has visual elements. "
            "Elements flow and connect across all three zones creating a continuous "
            "horizontal narrative. "
            f"{prompt_text} "
            "CRITICAL SPATIAL REQUIREMENTS: Visual elements must extend from the "
            "FAR LEFT EDGE all the way to the FAR RIGHT EDGE. "
            "Use repeating patterns, flowing lines, or distributed subjects that span "
            "the FULL HORIZONTAL WIDTH. "
            "Think ULTRA-WIDE landscape photography or cinema aspect ratio - "
            "NEVER a centered square subject with empty sides."
        )
    return (
        f"FULL-FRAME COMPOSITION: {prompt_text} "
        "CRITICAL: Create a full-frame composition that fills the entire canvas "
        "edge-to-edge with no borders, padding, or empty space around the illustration."
    )


def _aspect_ratio_to_size(image_plan: Dict[str, Any]) -> str:
    """Map an aspect_ratio string to a DALL-E 3 supported size string."""
    if image_plan.get("size"):
        return image_plan["size"]
    ar = str(image_plan.get("aspect_ratio", "")).strip()
    return {
        "16:9": "1792x1024", "16x9": "1792x1024",
        "4:3": "1792x1024",  "4x3": "1792x1024",
        "1:1": "1024x1024",  "1x1": "1024x1024",
    }.get(ar, "1024x1024")


# ── Graph factory ─────────────────────────────────────────────────────────────
def build_graph(
    settings: Optional[Settings] = None,
    rag_pipeline: Optional[RAGPipeline] = None,
):
    """
    Build and compile the content pipeline StateGraph.

    All agent instances are held in closure scope so the compiled graph is fully
    self-contained and thread-safe (each call to graph.invoke() creates a fresh
    execution context while sharing the agent objects).
    """
    from .supervisor import build_llm  # imported here to avoid circular import at module level

    settings = settings or get_settings()
    rag = rag_pipeline or RAGPipeline.from_examples()

    # ── Agent instantiation ────────────────────────────────────────────────────
    research_agent = ResearchAgent(
        llm=build_llm(settings, "research"),
        name="SEO Researcher",
        system_prompt="Strategic planner who identifies high-impact Career Compass topics.",
    )
    writer_agent = WritingAgent(
        llm=build_llm(settings, "writing"),
        name="Content Writer",
        system_prompt="Principal writer who crafts concise, actionable Career Compass blogs.",
    )
    editor_agent = EditingAgent(
        llm=build_llm(settings, "editing"),
        name="Senior Editor",
        system_prompt="Senior editor who polishes drafts to publish-ready quality.",
    )
    revision_editor_agent = EditingAgent(
        llm=build_llm(settings, "revision"),
        name="Senior Editor (Revision)",
        system_prompt="Senior editor tasked with bold, substantial rewrites to address specific quality issues.",
    )
    evaluation_agent = EvaluationAgent(
        llm=build_llm(settings, "evaluation"),
        name="Chief Content Officer",
        system_prompt="Chief Content Officer who evaluates content quality against publishing standards.",
        score_threshold=SCORE_THRESHOLD,
    )
    image_agent = ImageAgent(
        llm=build_llm(settings, "image"),
        name="Visual Designer",
        system_prompt="Designer who produces on-brand imagery prompts for Career Compass.",
    )
    publisher = PublishingAgent(
        llm=build_llm(settings, "metadata"),
        name="Publishing QA",
        system_prompt="Ensures metadata and publishing quality for Career Compass posts.",
        settings=settings,
    )

    # ── Node functions ─────────────────────────────────────────────────────────

    def node_research(state: ContentState) -> dict:
        print("🔵 [research] starting...")
        result = research_agent.recommend(state["existing_content"], state["context"])
        research_payload = _safe_parse(result.raw_response)
        topic = research_payload.get("topic", "Career Compass")
        print(f"📋 [research] topic: {research_payload.get('suggested_title', topic)}")
        print("✅ [research] done")
        return {
            "research_payload": research_payload,
            "text_context": rag.as_text_context(query=topic),
            "image_context": rag.as_image_context(query=topic),
        }

    def node_write(state: ContentState) -> dict:
        print("🔵 [write] starting...")
        result = writer_agent.draft(
            state["research_payload"],
            state["text_context"],
            state["context"],
        )
        blog_draft = result.content.get("message", str(result.raw_response))
        print("✅ [write] done")
        return {"blog_draft": blog_draft}

    def node_image_design(state: ContentState) -> dict:
        """
        Image design runs in a parallel branch alongside the write/edit/evaluate
        loop. In synchronous mode the branches execute back-to-back in the same
        superstep; switch to async node functions + graph.ainvoke() for true
        concurrent I/O.
        """
        print("🔵 [image_design] starting...")
        result = image_agent.design(
            state["research_payload"],
            state["image_context"],
            state["context"],
        )
        if isinstance(result.content, dict) and "error" in result.content:
            print(f"⚠️  [image_design] failed: {result.content.get('error')}")
            image_plan: Dict[str, Any] = {}
        else:
            image_plan = _safe_parse(result.raw_response)
        print(f"📸 [image_design] plan generated: {bool(image_plan.get('prompt'))}")
        print("✅ [image_design] done")
        return {"image_plan": image_plan}

    def node_image_generate(state: ContentState) -> dict:
        print("🔵 [image_generate] starting...")
        image_plan = state.get("image_plan", {})
        prompt_text = _augment_image_prompt(image_plan)
        # Blog hero images are always 16:9 landscape — hardcode to avoid
        # falling back to the square 1024x1024 DALL-E 3 default when
        # the image_plan is missing an aspect_ratio key.
        size = "1792x1024"
        print(f"📐 [image_generate] size: {size}")
        image_file_path: Optional[str] = None
        try:
            gen = ImageGenerator()
            gen.openai_api_key = settings.openai_api_key
            img_path = gen.generate(prompt_text, size=size)
            image_file_path = str(img_path)
            print("✅ [image_generate] done")
        except Exception as exc:
            print(f"⚠️  [image_generate] failed: {exc}")
        return {"image_file_path": image_file_path}

    def node_edit(state: ContentState) -> dict:
        # On first pass use blog_draft; on revisions use the previous edited_content
        source = state.get("edited_content") or state.get("blog_draft", "")
        feedback = state.get("feedback_text") or None
        is_revision = bool(feedback)

        if is_revision:
            print("🔵 [edit/revision] starting — applying evaluation feedback with bold rewrites...")
            result = revision_editor_agent.refine(source, feedback=feedback)
        else:
            print("🔵 [edit] starting — first-pass polish...")
            result = editor_agent.refine(source, feedback=None)

        edited_content = result.content.get("message", str(result.raw_response))
        print("✅ [edit] done")
        return {"edited_content": edited_content, "feedback_text": ""}

    def node_evaluate(state: ContentState) -> dict:
        print("🔵 [evaluate] starting...")
        evaluation = evaluation_agent.evaluate(state["edited_content"])
        revision_count = state.get("revision_count", 0) + 1
        passes = evaluation.average_score >= SCORE_THRESHOLD

        # Always print per-criterion scores for full visibility
        score_lines = "  ".join(
            f"{k.replace('_', ' ').title()}: {v:.1f}"
            for k, v in evaluation.scores.items()
        )
        print(f"📊 [evaluate] scores — {score_lines}")
        print(f"📊 [evaluate] average: {evaluation.average_score:.1f}/10  (threshold: {SCORE_THRESHOLD})")

        if evaluation.feedback.get("critical_issues"):
            print("📊 [evaluate] critical issues:")
            for issue in evaluation.feedback["critical_issues"]:
                print(f"   ✗ {issue}")

        if passes:
            print(f"✅ [evaluate] approved after {revision_count} cycle(s)")
        elif revision_count >= MAX_REVISIONS:
            print(f"⚠️  [evaluate] below threshold after {MAX_REVISIONS} revision(s). Proceeding to publish.")
        else:
            print(f"🔄 [evaluate] queuing revision {revision_count + 1}...")

        feedback_text = "" if passes else evaluation_agent.format_feedback_for_editor(evaluation)
        return {
            "evaluation_scores": evaluation.scores,
            "evaluation_avg": evaluation.average_score,
            "evaluation_passed": passes,
            "feedback_text": feedback_text,
            "revision_count": revision_count,
        }

    def node_prepare_metadata(state: ContentState) -> dict:
        print("🔵 [prepare_metadata] starting...")
        result = publisher.prepare_metadata(state["edited_content"])
        metadata_payload = _safe_parse(result.raw_response)
        if not metadata_payload.get("content_ready", True):
            issues = metadata_payload.get("blocking_issues", [])
            print(f"⚠️  [prepare_metadata] content_ready=False. Blocking issues: {issues}")
        print("✅ [prepare_metadata] done")
        return {"metadata_payload": metadata_payload}

    def node_publish(state: ContentState) -> dict:
        print("🔵 [publish] starting...")
        result = publisher.upload_to_webflow(
            state.get("metadata_payload", {}),
            state["edited_content"],
            image_file_path=state.get("image_file_path"),
        )
        status = result.get("status", "unknown")
        print(f"✅ [publish] done — status: {status}")
        return {"publish_result": result}

    # ── Routing (conditional edges) ────────────────────────────────────────────

    def route_after_evaluate(state: ContentState) -> str:
        """Loop back to editing or proceed to publishing."""
        if state.get("evaluation_passed") or state.get("revision_count", 0) >= MAX_REVISIONS:
            return "node_prepare_metadata"
        return "node_edit"

    def route_after_image_design(state: ContentState) -> str:
        """Skip image generation if the design step produced no prompt."""
        if state.get("image_plan", {}).get("prompt"):
            return "node_image_generate"
        return END

    # ── Graph assembly ─────────────────────────────────────────────────────────
    builder = StateGraph(ContentState)

    # Register nodes
    builder.add_node("node_research", node_research)
    builder.add_node("node_write", node_write)
    builder.add_node("node_image_design", node_image_design)
    builder.add_node("node_image_generate", node_image_generate)
    builder.add_node("node_edit", node_edit)
    builder.add_node("node_evaluate", node_evaluate)
    builder.add_node("node_prepare_metadata", node_prepare_metadata)
    builder.add_node("node_publish", node_publish)

    # Entry point
    builder.add_edge(START, "node_research")

    # Fan-out: research → content branch AND image branch (run back-to-back in
    # the same superstep; truly concurrent when using async node functions)
    builder.add_edge("node_research", "node_write")
    builder.add_edge("node_research", "node_image_design")

    # Content branch: write → edit → evaluate (with optional revision loop)
    builder.add_edge("node_write", "node_edit")
    builder.add_edge("node_edit", "node_evaluate")
    builder.add_conditional_edges("node_evaluate", route_after_evaluate)

    # Image branch: design → [generate if prompt exists] → END
    # image_file_path is written to shared state and read by node_prepare_metadata
    builder.add_conditional_edges("node_image_design", route_after_image_design)
    builder.add_edge("node_image_generate", END)

    # Publishing branch (triggered by content loop completion)
    builder.add_edge("node_prepare_metadata", "node_publish")
    builder.add_edge("node_publish", END)

    return builder.compile()
