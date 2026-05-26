from __future__ import annotations

from typing import Any, Dict, Optional

from typing_extensions import TypedDict

from .agents.base import AgentContext


class ContentState(TypedDict, total=False):
    """
    Shared state object threaded through every node in the content pipeline graph.

    Fields use last-write-wins semantics (LangGraph default). Each node returns
    only the keys it updates; all other keys are preserved from the previous state.
    """

    # ── Caller-supplied inputs ─────────────────────────────────────────────────
    existing_content: str   # Published Webflow posts + local brand examples
    context: AgentContext   # Brand voice, colors, audience, channels

    # ── Research outputs (node_research) ──────────────────────────────────────
    research_payload: Dict[str, Any]   # Full JSON from ResearchAgent
    text_context: str                  # RAG text snippets for WritingAgent
    image_context: str                 # RAG image examples for ImageAgent
    recent_image_history: list[Dict[str, Any]]  # Rolling window of prior image diversity tags
    image_retrieval_sources: list[str]  # Sources used to build image_context

    # ── Content creation loop ──────────────────────────────────────────────────
    blog_draft: str         # Raw draft from WritingAgent
    edited_content: str     # Polished draft from EditingAgent (updated each revision)
    revision_count: int     # Number of EvaluationAgent cycles completed

    # ── Evaluation fields (node_evaluate) ─────────────────────────────────────
    evaluation_scores: Dict[str, float]  # Per-criterion scores
    evaluation_avg: float                # Average score across all criteria
    evaluation_passed: bool              # True when avg >= SCORE_THRESHOLD
    feedback_text: str                   # Formatted critique sent back to EditingAgent

    # ── Image branch (runs in parallel with content loop) ─────────────────────
    # node_image_design and node_image_generate run concurrently with the
    # write/edit/evaluate loop. Both branches fan in to node_prepare_metadata,
    # which waits for both to complete before executing. This guarantees
    # image_file_path is always set in shared state before node_publish reads it.
    image_plan: Dict[str, Any]           # Prompt, alt_text, palette, aspect_ratio
    image_diversity_score: float         # 0-1 novelty score vs recent_image_history
    image_diversity_feedback: list[str]  # Diversity diagnostics for logging/QA
    image_file_path: Optional[str]       # Local path to generated image (or None)
    image_generation_error: Optional[str]  # Error message if generation failed, else None

    # ── Publishing (node_prepare_metadata, node_publish) ──────────────────────
    metadata_payload: Dict[str, Any]  # Title, slug, summary, tags, etc.
    publish_result: Dict[str, Any]    # Webflow API response
