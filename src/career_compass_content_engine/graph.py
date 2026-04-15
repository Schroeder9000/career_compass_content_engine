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
All nodes are ``async def``.  After node_research the CONTENT branch
(write → edit → evaluate) and the IMAGE branch (image_design → image_generate)
run as truly concurrent async tasks via ``graph.ainvoke()``.
By the time node_prepare_metadata fires both edited_content and image_file_path
are already in shared state.

Checkpointing
-------------
When ``settings.checkpointer_path`` is set (default: ``.cache/checkpoints.db``)
a LangGraph ``SqliteSaver`` is attached.  A failed run can be resumed from the
last completed node by passing the same ``thread_id`` in the RunnableConfig.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import re
from typing import Any, Callable, Dict, Optional

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
from .tools.topic_store import TopicStore
from .tools.gsc_client import GSCClient
from .tools.ga4_client import GA4Client
from .tools.serper_client import SerperClient
from .tools.webflow_content_client import WebflowContentClient
from .tools.competitor_fetcher import CompetitorFetcher
from .tools.cliche_screener import ClicheScreener
from .tools.readability_scorer import ReadabilityScorer

logger = logging.getLogger(__name__)

# ── Pipeline constants (single source of truth) ───────────────────────────────
MAX_REVISIONS: int = 3        # Max evaluation cycles before forcing publish
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


async def _run_sync(fn: Callable[[], Any]) -> Any:
    """Run a blocking callable in the default thread-pool executor while
    propagating the current *contextvars* context (e.g. LangChain callbacks)
    into the worker thread.  This is required for ``get_openai_callback()``
    to capture token usage from calls that run on the executor.
    """
    ctx = contextvars.copy_context()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: ctx.run(fn))


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
    """Map an aspect_ratio string to a gpt-image-1 supported size string."""
    if image_plan.get("size"):
        return image_plan["size"]
    ar = str(image_plan.get("aspect_ratio", "16:9")).strip()
    return {
        "16:9": "1536x1024", "16x9": "1536x1024",
        "4:3": "1536x1024",  "4x3": "1536x1024",
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
    self-contained and thread-safe (each call to graph.ainvoke() creates a fresh
    execution context while sharing the agent objects).

    When ``settings.checkpointer_path`` is non-empty a ``SqliteSaver`` is
    attached, enabling mid-run resume.  Pass a ``thread_id`` in the
    ``RunnableConfig`` to identify a run; re-using the same ID resumes from
    the last completed checkpoint.
    """
    from .supervisor import build_llm  # imported here to avoid circular import at module level

    settings = settings or get_settings()
    rag = rag_pipeline or RAGPipeline.from_examples()

    # ── Data-enrichment tools (all optional — gracefully disabled if creds absent) —
    topic_store = TopicStore()

    gsc_client: Optional[GSCClient] = None
    if settings.gsc_service_account_json and settings.gsc_site_url:
        try:
            gsc_client = GSCClient(
                service_account_json=settings.gsc_service_account_json,
                site_url=settings.gsc_site_url,
            )
            logger.info("[research tools] GSC client ready")
        except Exception as exc:
            logger.warning("[research tools] GSC client init failed: %s", exc)

    ga4_client: Optional[GA4Client] = None
    if settings.gsc_service_account_json and settings.ga4_property_id:
        try:
            ga4_client = GA4Client(
                service_account_json=settings.gsc_service_account_json,
                property_id=settings.ga4_property_id,
            )
            logger.info("[research tools] GA4 client ready")
        except Exception as exc:
            logger.warning("[research tools] GA4 client init failed: %s", exc)

    serper_client: Optional[SerperClient] = None
    if settings.serper_api_key:
        try:
            serper_client = SerperClient(api_key=settings.serper_api_key)
            logger.info("[research tools] Serper client ready")
        except Exception as exc:
            logger.warning("[research tools] Serper client init failed: %s", exc)

    webflow_content_client: Optional[WebflowContentClient] = None
    if settings.webflow_api_token and settings.webflow_collection_id:
        try:
            webflow_content_client = WebflowContentClient(
                api_token=settings.webflow_api_token,
                collection_id=settings.webflow_collection_id,
            )
            logger.info("[research tools] Webflow content client ready")
        except Exception as exc:
            logger.warning("[research tools] Webflow content client init failed: %s", exc)

    competitor_fetcher = CompetitorFetcher()
    cliche_screener = ClicheScreener()
    readability_scorer = ReadabilityScorer()

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

    async def node_research(state: ContentState) -> dict:
        logger.info("🔵 [research] starting...")

        # ── 1. Build performance_context from GSC + GA4 ───────────────────────────────────
        perf_parts = []
        gsc_queries: list = []  # reused for Serper targeting
        if gsc_client:
            try:
                logger.info("📁 [research] fetching GSC data...")
                gsc_summary = gsc_client.fetch_performance_summary(days=90, query_limit=30)
                perf_parts.append(gsc_summary)
                # Extract opportunity queries (position 4–20) for Serper
                raw_queries = gsc_client.fetch_top_queries(days=90, row_limit=30)
                gsc_queries = [
                    q["query"] for q in raw_queries
                    if 4 <= q["position"] <= 20 and q["impressions"] >= 50
                ][:3]
                logger.info("✅ [research] GSC loaded (%d queries)", len(raw_queries))
            except Exception as exc:
                logger.warning("⚠️  [research] GSC fetch failed: %s", exc)

        if ga4_client:
            try:
                logger.info("📁 [research] fetching GA4 data...")
                ga4_summary = ga4_client.fetch_engagement_summary(days=90)
                perf_parts.append(ga4_summary)
                logger.info("✅ [research] GA4 loaded")
            except Exception as exc:
                logger.warning("⚠️  [research] GA4 fetch failed: %s", exc)

        performance_context = "\n\n".join(perf_parts) if perf_parts else None

        # ── 2. Build competitor_context from Serper + PAA + live page fetch ─────────
        competitor_context = None
        if serper_client and gsc_queries:
            try:
                logger.info("🔍 [research] running Serper on %d opportunity quer(ies)...", len(gsc_queries))
                serp_summary = serper_client.competitor_summary(gsc_queries)
                paa_summary = serper_client.paa_summary(gsc_queries)

                # Fetch and summarize actual competitor page content (top 3 URLs)
                top_urls: list = []
                try:
                    raw_serp = serper_client.fetch_serp_snapshot(gsc_queries[:1], num_results=3)
                    top_urls = [r["link"] for r in raw_serp if r.get("link")][:3]
                except Exception:
                    pass
                fetched_content = ""
                if top_urls:
                    logger.info("🌐 [research] fetching %d competitor page(s)...", len(top_urls))
                    fetched_content = competitor_fetcher.summarize_urls(top_urls)

                competitor_context = "\n\n".join(
                    part for part in [serp_summary, paa_summary, fetched_content] if part
                )
                logger.info("✅ [research] Serper SERP + PAA + competitor content loaded")
            except Exception as exc:
                logger.warning("⚠️  [research] Serper fetch failed: %s", exc)

        # ── 3. Append topic history + live Webflow corpus to existing_content ─────────
        history_block = topic_store.format_history_for_prompt()
        live_cms_block = ""
        if webflow_content_client:
            try:
                logger.info("📚 [research] fetching live Webflow CMS corpus...")
                live_cms_block = webflow_content_client.published_content_summary()
                logger.info("✅ [research] Webflow CMS corpus loaded")
            except Exception as exc:
                logger.warning("⚠️  [research] Webflow CMS fetch failed: %s", exc)

        enriched_existing = "\n\n".join(
            part for part in [
                state.get("existing_content", ""),
                history_block,
                live_cms_block,
            ]
            if part
        )

        # ── 4. Run research agent ───────────────────────────────────────────────────────────
        logger.debug(
            "[research] enriched_existing snippet: %.200s", enriched_existing
        )
        result = await _run_sync(
            lambda: research_agent.recommend(
                enriched_existing,
                state["context"],
                performance_context=performance_context,
                competitor_context=competitor_context,
            )
        )
        research_payload = _safe_parse(result.raw_response)
        logger.debug("[research] raw response snippet: %.300s", result.raw_response)
        topic = research_payload.get("topic", "Career Compass")
        logger.info("📋 [research] topic: %s", research_payload.get("suggested_title", topic))
        logger.info("✅ [research] done")
        return {
            "research_payload": research_payload,
            "text_context": rag.as_text_context(query=topic),
            "image_context": rag.as_image_context(query=topic),
        }

    async def node_write(state: ContentState) -> dict:
        logger.info("🔵 [write] starting...")
        result = await _run_sync(
            lambda: writer_agent.draft(
                state["research_payload"],
                state["text_context"],
                state["context"],
            )
        )
        blog_draft = result.content.get("message", str(result.raw_response))
        logger.debug("[write] draft word count: ~%d", len(blog_draft.split()))
        logger.info("✅ [write] done")
        return {"blog_draft": blog_draft}

    async def node_image_design(state: ContentState) -> dict:
        """
        Image design runs in a parallel branch alongside the write/edit/evaluate
        loop. As an async node this branch runs concurrently with node_write when
        graph.ainvoke() is used.
        """
        logger.info("🔵 [image_design] starting...")
        result = await _run_sync(
            lambda: image_agent.design(
                state["research_payload"],
                state["image_context"],
                state["context"],
            )
        )
        logger.debug("[image_design] raw response snippet: %.300s", result.raw_response)
        if isinstance(result.content, dict) and "error" in result.content:
            logger.warning("⚠️  [image_design] failed: %s", result.content.get("error"))
            image_plan: Dict[str, Any] = {}
        else:
            image_plan = _safe_parse(result.raw_response)
        logger.info("📸 [image_design] plan generated: %s", bool(image_plan.get("prompt")))
        logger.info("✅ [image_design] done")
        return {"image_plan": image_plan}

    async def node_image_generate(state: ContentState) -> dict:
        logger.info("🔵 [image_generate] starting...")
        image_plan = state.get("image_plan", {})
        prompt_text = _augment_image_prompt(image_plan)
        size = _aspect_ratio_to_size(image_plan)
        logger.info("📐 [image_generate] size: %s", size)
        image_file_path: Optional[str] = None
        try:
            gen = ImageGenerator()
            gen.openai_api_key = settings.openai_api_key
            img_path = await _run_sync(lambda: gen.generate(prompt_text, size=size))
            # Sanity-check: reject suspiciously small files (e.g. 1x1 pixel fallbacks)
            if img_path.stat().st_size < 1024:
                raise RuntimeError(
                    f"Generated image is only {img_path.stat().st_size} bytes — treating as failed generation"
                )
            image_file_path = str(img_path)
            logger.info("✅ [image_generate] done")
        except Exception as exc:
            logger.warning("⚠️  [image_generate] failed: %s", exc)
        return {"image_file_path": image_file_path}

    async def node_edit(state: ContentState) -> dict:
        # On first pass use blog_draft; on revisions use the previous edited_content
        source = state.get("edited_content") or state.get("blog_draft", "")
        feedback = state.get("feedback_text") or None
        is_revision = bool(feedback)

        # ── Pre-edit: run cliché screener and readability scorer ──────────────
        cliche_report = cliche_screener.screen(source)
        readability_report = readability_scorer.score(source)

        cliche_feedback = cliche_report.feedback_for_editor()
        readability_feedback = readability_report.feedback_for_editor()

        if cliche_report.flag_count > 0:
            logger.info("🔍 [edit] cliché screener: %d flag(s) found", cliche_report.flag_count)
        if readability_report.issues:
            logger.info("📖 [edit] readability: %d issue(s) — %s", len(readability_report.issues), readability_report.summary().splitlines()[0])

        # Merge pre-screen feedback into any existing evaluation feedback
        pre_screen_block = "\n\n".join(
            part for part in [cliche_feedback, readability_feedback] if part
        )
        combined_feedback = "\n\n".join(
            part for part in [feedback or "", pre_screen_block] if part
        ) or None

        if is_revision:
            logger.info("🔵 [edit/revision] starting — applying evaluation feedback with bold rewrites...")
            result = await _run_sync(lambda: revision_editor_agent.refine(source, feedback=combined_feedback))
        else:
            logger.info("🔵 [edit] starting — first-pass polish...")
            result = await _run_sync(lambda: editor_agent.refine(source, feedback=combined_feedback if pre_screen_block else None))

        edited_content = result.content.get("message", str(result.raw_response))
        logger.debug("[edit] edited word count: ~%d", len(edited_content.split()))
        logger.info("✅ [edit] done")
        return {"edited_content": edited_content, "feedback_text": ""}

    async def node_evaluate(state: ContentState) -> dict:
        logger.info("🔵 [evaluate] starting...")
        evaluation = await _run_sync(lambda: evaluation_agent.evaluate(state["edited_content"]))
        logger.debug("[evaluate] raw JSON: %.500s", str(evaluation.raw_json))
        revision_count = state.get("revision_count", 0) + 1
        passes = evaluation.average_score >= SCORE_THRESHOLD

        # Always log per-criterion scores for full visibility
        score_lines = "  ".join(
            f"{k.replace('_', ' ').title()}: {v:.1f}"
            for k, v in evaluation.scores.items()
        )
        logger.info("📊 [evaluate] scores — %s", score_lines)
        logger.info("📊 [evaluate] average: %.1f/10  (threshold: %s)", evaluation.average_score, SCORE_THRESHOLD)

        if evaluation.feedback.get("critical_issues"):
            logger.info("📊 [evaluate] critical issues:")
            for issue in evaluation.feedback["critical_issues"]:
                logger.warning("   ✗ %s", issue)

        if passes:
            logger.info("✅ [evaluate] approved after %d cycle(s)", revision_count)
        elif revision_count >= MAX_REVISIONS:
            logger.warning("⚠️  [evaluate] below threshold after %d revision(s). Proceeding to publish.", MAX_REVISIONS)
        else:
            logger.info("🔄 [evaluate] queuing revision %d...", revision_count + 1)

        feedback_text = "" if passes else evaluation_agent.format_feedback_for_editor(evaluation)
        return {
            "evaluation_scores": evaluation.scores,
            "evaluation_avg": evaluation.average_score,
            "evaluation_passed": passes,
            "feedback_text": feedback_text,
            "revision_count": revision_count,
        }

    async def node_prepare_metadata(state: ContentState) -> dict:
        logger.info("🔵 [prepare_metadata] starting...")
        result = await _run_sync(lambda: publisher.prepare_metadata(state["edited_content"]))
        metadata_payload = _safe_parse(result.raw_response)
        logger.debug("[prepare_metadata] payload keys: %s", list(metadata_payload.keys()))
        if not metadata_payload.get("content_ready", True):
            issues = metadata_payload.get("blocking_issues", [])
            logger.warning("⚠️  [prepare_metadata] content_ready=False. Blocking issues: %s", issues)
        logger.info("✅ [prepare_metadata] done")
        return {"metadata_payload": metadata_payload}

    async def node_publish(state: ContentState) -> dict:
        logger.info("🔵 [publish] starting...")
        result = await _run_sync(
            lambda: publisher.upload_to_webflow(
                state.get("metadata_payload", {}),
                state["edited_content"],
                image_file_path=state.get("image_file_path"),
            )
        )
        status = result.get("status", "unknown")
        logger.info("✅ [publish] done — status: %s", status)

        # Record this run in the persistent topic history
        research_payload = state.get("research_payload", {})
        try:
            webflow_item_id = (
                result.get("webflow_item_id")
                or result.get("webflow_item", {}).get("id")
                or result.get("item", {}).get("id")
                or result.get("id")
            )
            topic_store.append(
                topic=research_payload.get("topic", ""),
                slug=research_payload.get("suggested_title", ""),
                suggested_title=research_payload.get("suggested_title", ""),
                final_score=state.get("evaluation_avg", 0.0),
                revision_count=state.get("revision_count", 0),
                evaluation_passed=state.get("evaluation_passed", False),
                webflow_item_id=webflow_item_id,
            )
            logger.info("💾 [publish] run recorded to topic history")
        except Exception as exc:
            logger.warning("⚠️  [publish] topic history write failed: %s", exc)

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

    # ── Checkpointing ──────────────────────────────────────────────────────────
    # MemorySaver checkpoints every completed node within a single run so that
    # a failed node can be retried without re-executing earlier work.  It is
    # always available (no external dependency) and is fully compatible with
    # ainvoke.  Cross-process persistence (resuming after a hard crash between
    # separate Python invocations) requires AsyncSqliteSaver wired into the
    # Supervisor's async lifecycle — that is left as a future enhancement.
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()
    logger.info("[graph] MemorySaver checkpointer attached (in-process checkpointing enabled)")

    return builder.compile(checkpointer=checkpointer)
