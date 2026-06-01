"""
Career Compass Content Pipeline — LangGraph StateGraph

Graph topology
--------------
                              ┌─── node_write ─── node_edit ─┐
                              │                               ▼
START ── node_research ───────┤                         node_evaluate ──┐
                              │                                          │
                              └─── node_image_design                    │
                                         │                              ▼
                               node_image_generate ──► node_prepare_metadata ── node_publish → END

Parallel branches
-----------------
All nodes are ``async def``.  After node_research the CONTENT branch
(write → edit → evaluate) and the IMAGE branch (image_design → image_generate)
run as truly concurrent async tasks via ``graph.ainvoke()``.
node_prepare_metadata is a fan-in node: it waits for BOTH branches to complete
before executing — ensuring image_file_path is always set in shared state before
node_publish reads it.  node_image_generate always runs (it handles the no-prompt
case with a fast early return), keeping the fan-in unambiguous for LangGraph.

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
IMAGE_HISTORY_WINDOW: int = 8


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
    """Append guardrails while preserving diversity chosen by the image plan."""
    prompt_text = image_plan.get("prompt", "")
    aspect_ratio = image_plan.get("aspect_ratio")
    diversity_tags = _extract_diversity_tags(image_plan)

    constraints = [
        "Generate a full-bleed edge-to-edge hero illustration.",
        "Use bright, optimistic lighting with warm highlights and open midtones.",
        "Avoid gloomy low-key darkness and avoid default centered poster layouts.",
    ]
    if aspect_ratio:
        constraints.insert(0, f"Respect {aspect_ratio} aspect ratio.")
    if diversity_tags:
        constraints.append(
            "Preserve the requested diversity profile exactly: "
            + "; ".join(f"{k}={v}" for k, v in diversity_tags.items())
            + "."
        )
    return f"{prompt_text}\n\nAdditional constraints:\n- " + "\n- ".join(constraints)


def _build_diversity_target(history: list[Dict[str, Any]], topic_seed: str) -> Dict[str, str]:
    """Pick a deterministic diversity profile to rotate across runs."""
    profiles = [
        {
            "viewpoint": "bird's-eye",
            "composition_archetype": "distributed constellation",
            "complexity_level": "high-detail layered",
            "lighting_style": "warm daylight optimism",
            "shot_distance": "wide environmental",
            "camera_angle": "top-oblique",
            "focal_strategy": "multi-node narrative",
            "orientation_treatment": "left-to-right progression",
        },
        {
            "viewpoint": "eye-level",
            "composition_archetype": "branching crossroads",
            "complexity_level": "medium narrative",
            "lighting_style": "clear noon contrast",
            "shot_distance": "mid scene",
            "camera_angle": "straight-on",
            "focal_strategy": "foreground-to-background journey",
            "orientation_treatment": "center-offset momentum",
        },
        {
            "viewpoint": "low-angle",
            "composition_archetype": "diagonal ascent",
            "complexity_level": "minimal symbolic",
            "lighting_style": "golden-hour directional glow",
            "shot_distance": "close to medium",
            "camera_angle": "low tilt-up",
            "focal_strategy": "single dominant motif",
            "orientation_treatment": "right-to-left tension-to-release",
        },
        {
            "viewpoint": "isometric/sectional",
            "composition_archetype": "layered skyline",
            "complexity_level": "high-detail layered",
            "lighting_style": "crisp ambient light",
            "shot_distance": "wide systemic",
            "camera_angle": "sectional cutaway",
            "focal_strategy": "asymmetrical balance",
            "orientation_treatment": "tri-zone rhythm",
        },
    ]

    offset = (len(history) + sum(ord(ch) for ch in (topic_seed or ""))) % len(profiles)
    target = dict(profiles[offset])

    if history:
        last = history[0].get("image_diversity", {})
        if isinstance(last, dict):
            # Avoid immediate axis repeats by stepping one profile if too similar.
            shared_axes = sum(
                1
                for axis in ("viewpoint", "composition_archetype", "shot_distance", "camera_angle")
                if str(last.get(axis, "")).strip().lower() == str(target.get(axis, "")).strip().lower()
            )
            if shared_axes >= 2:
                target = dict(profiles[(offset + 1) % len(profiles)])

    return target


def _format_diversity_target(target: Dict[str, str]) -> str:
    if not target:
        return "(No diversity target available.)"
    return (
        "Required diversity profile for this run:\n"
        + "\n".join(f"- {key}: {value}" for key, value in target.items())
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


def _extract_diversity_tags(image_plan: Dict[str, Any]) -> Dict[str, str]:
    """Normalize diversity tags from the image plan payload."""
    raw = image_plan.get("diversity_tags", {}) if isinstance(image_plan, dict) else {}
    if not isinstance(raw, dict):
        return {}
    keys = (
        "viewpoint",
        "composition_archetype",
        "complexity_level",
        "lighting_style",
        "shot_distance",
        "camera_angle",
        "focal_strategy",
    )
    return {
        key: str(raw.get(key, "")).strip()
        for key in keys
        if str(raw.get(key, "")).strip()
    }


def _build_recent_diversity_summary(history: list[Dict[str, Any]]) -> str:
    """Build compact anti-repetition context for the image prompt."""
    if not history:
        return "(No recent image history available.)"
    lines = ["Most recent image profiles (newest first):"]
    for idx, item in enumerate(history[:IMAGE_HISTORY_WINDOW], 1):
        tags = item.get("image_diversity", {})
        if not isinstance(tags, dict):
            continue
        lines.append(
            f"{idx}. viewpoint={tags.get('viewpoint', '?')}; "
            f"composition={tags.get('composition_archetype', '?')}; "
            f"distance={tags.get('shot_distance', '?')}; "
            f"angle={tags.get('camera_angle', '?')}; "
            f"complexity={tags.get('complexity_level', '?')}; "
            f"lighting={tags.get('lighting_style', '?')}"
        )
    return "\n".join(lines)


def _build_diversity_avoid_patterns(history: list[Dict[str, Any]]) -> list[str]:
    """Build simple lexical avoid-pattern list from recent history tags."""
    patterns: list[str] = []
    for item in history[:IMAGE_HISTORY_WINDOW]:
        tags = item.get("image_diversity", {})
        if not isinstance(tags, dict):
            continue
        for key in (
            "composition_archetype",
            "viewpoint",
            "shot_distance",
            "camera_angle",
            "focal_strategy",
        ):
            value = str(tags.get(key, "")).strip().lower()
            if value:
                patterns.append(value)
    # Keep order stable while removing duplicates.
    return list(dict.fromkeys(patterns))


def _score_image_diversity(
    current_tags: Dict[str, str],
    history: list[Dict[str, Any]],
) -> tuple[float, list[str]]:
    """Return novelty score (0-1) and diagnostics against recent history."""
    if not current_tags or not history:
        return 1.0, ["No recent image history to compare; treating as fully novel."]

    diagnostics: list[str] = []
    penalty = 0.0
    recent = history[:IMAGE_HISTORY_WINDOW]
    recent_tags = [h.get("image_diversity", {}) for h in recent if isinstance(h.get("image_diversity", {}), dict)]

    # Penalize repeating any axis from the immediate previous generation.
    previous = recent_tags[0] if recent_tags else {}
    for axis, value in current_tags.items():
        if not value:
            continue
        if str(previous.get(axis, "")).strip().lower() == value.strip().lower():
            penalty += 0.10
            diagnostics.append(f"Immediate repeat on axis '{axis}': {value}")

    # Penalize frequent repetition over recent window.
    for axis, value in current_tags.items():
        if not value:
            continue
        repeats = sum(
            1
            for tags in recent_tags
            if str(tags.get(axis, "")).strip().lower() == value.strip().lower()
        )
        reuse_rate = repeats / max(len(recent_tags), 1)
        if reuse_rate >= 0.5:
            penalty += 0.08
            diagnostics.append(f"High reuse on axis '{axis}' ({repeats}/{len(recent_tags)} recent runs)")

    # Strong penalty for repeating the same composition-distance-angle combo in last 3.
    combo = (
        current_tags.get("composition_archetype", "").strip().lower(),
        current_tags.get("shot_distance", "").strip().lower(),
        current_tags.get("camera_angle", "").strip().lower(),
    )
    if all(combo):
        for tags in recent_tags[:3]:
            prev_combo = (
                str(tags.get("composition_archetype", "")).strip().lower(),
                str(tags.get("shot_distance", "")).strip().lower(),
                str(tags.get("camera_angle", "")).strip().lower(),
            )
            if combo == prev_combo:
                penalty += 0.35
                diagnostics.append("Critical repeat: composition+distance+angle matches recent output")
                break

    score = max(0.0, min(1.0, 1.0 - penalty))
    if not diagnostics:
        diagnostics.append("Good novelty against recent image history.")
    return score, diagnostics


def _has_substantive_closing(markdown: str) -> bool:
    """Heuristic check that the ending contains at least two real prose paragraphs."""
    if not markdown:
        return False
    tail = markdown[-2200:]
    blocks = [b.strip() for b in re.split(r"\n\s*\n", tail) if b.strip()]
    prose_blocks = [
        b for b in blocks
        if not b.startswith("#")
        and not b.startswith(">")
        and not b.startswith("-")
        and not re.match(r"^\d+\.\s", b)
    ]
    if len(prose_blocks) < 2:
        return False
    return len(prose_blocks[-1].split()) + len(prose_blocks[-2].split()) >= 80


def _check_required_ending(markdown: str) -> list[str]:
    """Return unmet hard requirements for final publish quality."""
    issues: list[str] = []
    if not re.search(r"\bcareer\s+compass\b", markdown or "", re.IGNORECASE):
        issues.append("Missing a natural 'Career Compass' mention near the end of the article.")
    if not _has_substantive_closing(markdown or ""):
        issues.append("Closing is too abrupt; needs 2-4 substantive concluding paragraphs.")
    return issues


def _append_fallback_closing(markdown: str) -> str:
    """Final safety net if LLM repair still misses required ending elements."""
    closing = (
        "\n\nBefore you publish your next move at work, make one choice on purpose: "
        "ask earlier, with sharper context, while options are still open. "
        "That is what separates reactive stress from professional judgment.\n\n"
        "If you want help building that habit, Career Compass is designed for exactly this moment: "
        "a personalized growth plan, weekly coaching nudges, and visible progress tracking so your "
        "career growth is deliberate instead of accidental."
    )
    base = (markdown or "").rstrip()
    if re.search(r"\bcareer\s+compass\b", base, re.IGNORECASE) and _has_substantive_closing(base):
        return base
    return base + closing


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

        recent_image_history = topic_store.recent_image_history(limit=IMAGE_HISTORY_WINDOW)
        diversity_target = _build_diversity_target(
            recent_image_history,
            research_payload.get("topic", "") or research_payload.get("suggested_title", ""),
        )
        avoid_patterns = _build_diversity_avoid_patterns(recent_image_history)
        image_context, image_sources = rag.as_image_context_with_sources(
            query=topic,
            avoid_patterns=avoid_patterns,
        )
        logger.info(
            "🧭 [research] image diversity context — recent history: %d  avoid patterns: %d  selected sources: %d",
            len(recent_image_history),
            len(avoid_patterns),
            len(image_sources),
        )

        logger.info("📋 [research] topic: %s", research_payload.get("suggested_title", topic))
        logger.info("✅ [research] done")
        return {
            "research_payload": research_payload,
            "text_context": rag.as_text_context(query=topic),
            "image_context": image_context,
            "recent_image_history": recent_image_history,
            "image_diversity_target": diversity_target,
            "image_retrieval_sources": image_sources,
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
                recent_history_context=_build_recent_diversity_summary(state.get("recent_image_history", [])),
                diversity_target_context=_format_diversity_target(state.get("image_diversity_target", {})),
            )
        )
        logger.debug("[image_design] raw response snippet: %.300s", result.raw_response)
        if isinstance(result.content, dict) and "error" in result.content:
            logger.warning("⚠️  [image_design] failed: %s", result.content.get("error"))
            image_plan: Dict[str, Any] = {}
        else:
            image_plan = _safe_parse(result.raw_response)
        diversity_tags = _extract_diversity_tags(image_plan)
        diversity_score, diversity_feedback = _score_image_diversity(
            diversity_tags,
            state.get("recent_image_history", []),
        )
        logger.info("🧪 [image_design] diversity score: %.2f", diversity_score)
        for message in diversity_feedback[:3]:
            logger.info("🧪 [image_design] %s", message)
        logger.info("📸 [image_design] plan generated: %s", bool(image_plan.get("prompt")))
        logger.info("✅ [image_design] done")
        return {
            "image_plan": image_plan,
            "image_diversity_score": diversity_score,
            "image_diversity_feedback": diversity_feedback,
        }

    async def node_image_generate(state: ContentState) -> dict:
        logger.info("🔵 [image_generate] starting...")
        image_plan = state.get("image_plan", {})
        if not image_plan.get("prompt"):
            logger.warning("⚠️  [image_generate] no prompt in image_plan — skipping generation")
            return {"image_file_path": None, "image_generation_error": "no prompt"}
        prompt_text = _augment_image_prompt(image_plan)
        size = _aspect_ratio_to_size(image_plan)
        logger.info("📐 [image_generate] size: %s", size)
        image_file_path: Optional[str] = None
        gen = ImageGenerator()
        gen.openai_api_key = settings.openai_api_key
        # Honour the model configured in Settings (env override still respected
        # inside _resolve_image_model, but Settings is the canonical source).
        if settings.openai_image_model:
            import os as _os
            _os.environ.setdefault("OPENAI_IMAGE_MODEL", settings.openai_image_model)

        attempted_sizes = [size]
        if size != "1024x1024":
            # Fallback pass for transient model/size issues.
            attempted_sizes.append("1024x1024")

        last_error: Optional[str] = None
        for attempt_idx, attempt_size in enumerate(attempted_sizes, 1):
            try:
                logger.info("🧪 [image_generate] attempt %d/%d with size %s", attempt_idx, len(attempted_sizes), attempt_size)
                img_path = await _run_sync(lambda: gen.generate(prompt_text, size=attempt_size))
                # Sanity-check: reject suspiciously small files (e.g. 1x1 pixel fallbacks)
                if img_path.stat().st_size < 1024:
                    raise RuntimeError(
                        f"Generated image is only {img_path.stat().st_size} bytes — treating as failed generation"
                    )
                image_file_path = str(img_path)
                break
            except Exception as exc:
                last_error = str(exc)
                logger.warning("⚠️  [image_generate] attempt %d failed: %s", attempt_idx, exc)

        if not image_file_path:
            return {"image_file_path": None, "image_generation_error": last_error or "image generation failed"}

        logger.info("✅ [image_generate] done")
        return {"image_file_path": image_file_path, "image_generation_error": None}

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
        content_for_publish = state.get("edited_content", "")
        ending_issues = _check_required_ending(content_for_publish)

        if ending_issues:
            logger.warning("⚠️  [prepare_metadata] ending guard triggered: %s", " | ".join(ending_issues))
            repair_feedback = (
                "HARD REQUIREMENTS BEFORE PUBLISH:\n"
                "- Add a natural Career Compass mention in the final section/conclusion.\n"
                "- End with 2-4 substantive closing paragraphs that synthesize the argument and leave a concrete mindset shift.\n"
                "- Preserve the article's core argument and tone. Avoid adding FAQ/template sections.\n\n"
                f"Current issues: {'; '.join(ending_issues)}"
            )
            repaired = await _run_sync(lambda: revision_editor_agent.refine(content_for_publish, feedback=repair_feedback))
            content_for_publish = repaired.content.get("message", str(repaired.raw_response))

            remaining_issues = _check_required_ending(content_for_publish)
            if remaining_issues:
                logger.warning("⚠️  [prepare_metadata] LLM repair incomplete; applying deterministic fallback closing.")
                content_for_publish = _append_fallback_closing(content_for_publish)

        result = await _run_sync(lambda: publisher.prepare_metadata(content_for_publish))
        metadata_payload = _safe_parse(result.raw_response)
        logger.debug("[prepare_metadata] payload keys: %s", list(metadata_payload.keys()))
        if not metadata_payload.get("content_ready", True):
            issues = metadata_payload.get("blocking_issues", [])
            logger.warning("⚠️  [prepare_metadata] content_ready=False. Blocking issues: %s", issues)
        logger.info("✅ [prepare_metadata] done")
        return {"metadata_payload": metadata_payload, "edited_content": content_for_publish}

    async def node_publish(state: ContentState) -> dict:
        logger.info("🔵 [publish] starting...")
        image_path = state.get("image_file_path")
        if image_path:
            logger.info("🖼️  [publish] image available: %s", image_path)
            logger.info("🧭 [publish] image diversity score: %.2f", state.get("image_diversity_score", 0.0))
        else:
            image_err = state.get("image_generation_error")
            if image_err:
                logger.warning("⚠️  [publish] publishing without image (generation failed: %s)", image_err)
            else:
                logger.warning("⚠️  [publish] publishing without image (no image_file_path in state)")
        result = await _run_sync(
            lambda: publisher.upload_to_webflow(
                state.get("metadata_payload", {}),
                state["edited_content"],
                image_file_path=image_path,
            )
        )
        result["image_attached"] = bool(image_path) and result.get("status") not in ("skipped", "error")
        status = result.get("status", "unknown")
        logger.info("✅ [publish] done — status: %s  image_attached: %s", status, result["image_attached"])
        publish_error = result.get("reason") or (result.get("webflow_publish") or {}).get("error")

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
                slug=state.get("metadata_payload", {}).get("slug") or research_payload.get("suggested_title", ""),
                suggested_title=research_payload.get("suggested_title", ""),
                final_score=state.get("evaluation_avg", 0.0),
                revision_count=state.get("revision_count", 0),
                evaluation_passed=state.get("evaluation_passed", False),
                webflow_item_id=webflow_item_id,
                image_diversity=_extract_diversity_tags(state.get("image_plan", {})),
                image_inspiration_sources=state.get("image_retrieval_sources", []),
                publish_status=str(status),
                image_attached=bool(result.get("image_attached", False)),
                publish_error=str(publish_error) if publish_error else None,
            )
            logger.info("💾 [publish] run recorded to topic history")
        except Exception as exc:
            logger.warning("⚠️  [publish] topic history write failed: %s", exc)

        if status == "error":
            raise RuntimeError(str(publish_error or "Webflow publish failed"))

        return {"publish_result": result}

    # ── Routing (conditional edges) ────────────────────────────────────────────

    def route_after_evaluate(state: ContentState) -> str:
        """Loop back to editing or proceed to publishing."""
        if state.get("evaluation_passed") or state.get("revision_count", 0) >= MAX_REVISIONS:
            return "node_prepare_metadata"
        return "node_edit"

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

    # Image branch: design → generate → END (branch terminates here).
    # node_image_generate sets image_file_path in shared LangGraph state and then
    # terminates its branch.  The content branch (node_evaluate → node_prepare_metadata)
    # runs independently; because image generation completes well before the content
    # loop finishes, image_file_path is always present in shared state when
    # node_prepare_metadata reads it.  Routing image_generate to END (rather than
    # to node_prepare_metadata) prevents node_prepare_metadata from firing twice —
    # once from the image branch and once from node_evaluate — which caused a
    # double-publish bug.
    builder.add_edge("node_image_design", "node_image_generate")
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
