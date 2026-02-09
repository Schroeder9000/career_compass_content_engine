from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional

from langchain_openai import ChatOpenAI

from .agents.base import AgentContext
from .agents.image_agent import ImageAgent
from .agents.publishing_agent import PublishingAgent
from .agents.research_agent import ResearchAgent
from .agents.writing_agent import WritingAgent
from .agents.editing_agent import EditingAgent
from .agents.evaluation_agent import EvaluationAgent
from .agents.image_generator import ImageGenerator
from .config import Settings, get_settings
from .rag import RAGPipeline


def build_llm(settings: Settings, task_type: str = "default") -> ChatOpenAI:
    """
    Build LLM optimized for specific task types.
    
    Task types:
    - research: Strategic analysis and planning (highest reasoning)
    - writing: Long-form content generation (high creativity and quality)
    - editing: Refinement and polish (balanced quality and consistency)
    - image: Visual concept design (creative but structured)
    - metadata: Structured data generation (precision and consistency)
    """
    # Use the most advanced models for highest quality output
    # o1 for research (deep reasoning), GPT-4 for content generation
    model_config = {
        "research": {
            "model": "o1",  # Highest reasoning for strategic analysis
            "temperature": 1,  # o1 models use temperature=1
        },
        "writing": {
            "model": "gpt-4o",  # Best for long-form creative content
            "temperature": 0.7,  # Higher creativity for engaging content
        },
        "editing": {
            "model": "gpt-4o",  # Best quality for refinement
            "temperature": 0.5,  # Balanced for consistency with creativity
        },
        "image": {
            "model": "gpt-4o",  # Best for creative visual concepts
            "temperature": 0.8,  # High creativity for unique visuals
        },
        "metadata": {
            "model": "gpt-4o",  # Best for structured output
            "temperature": 0.3,  # Lower for consistent formatting
        },
        "default": {
            "model": "gpt-4o",  # Default to best general model
            "temperature": 0.7,
        },
    }
    
    config = model_config.get(task_type, model_config["default"])
    return ChatOpenAI(
        model=config["model"],
        api_key=settings.openai_api_key,
        temperature=config["temperature"],
    )


@dataclass
class WorkflowResult:
    research: Dict[str, str]
    blog_post: str
    image_plan: Dict[str, str]
    publish_result: Dict[str, str]


class Supervisor:
    """Coordinates the Career Compass content workflow."""

    def __init__(self, settings: Optional[Settings] = None, rag_pipeline: Optional[RAGPipeline] = None) -> None:
        self.settings = settings or get_settings()
        # Build task-optimized LLMs for highest quality output
        self.llm_research = build_llm(self.settings, "research")
        self.llm_writing = build_llm(self.settings, "writing")
        self.llm_editing = build_llm(self.settings, "editing")
        self.llm_image = build_llm(self.settings, "image")
        self.llm_metadata = build_llm(self.settings, "metadata")
        
        self.rag = rag_pipeline or RAGPipeline.from_examples()
        self.research_agent = ResearchAgent(
            llm=self.llm_research,
            name="SEO Researcher",
            system_prompt="Strategic planner who identifies high-impact Career Compass topics.",
        )
        self.writer_agent = WritingAgent(
            llm=self.llm_writing,
            name="Content Writer",
            system_prompt="Principal writer who crafts concise, actionable Career Compass blogs.",
        )
        self.editor_agent = EditingAgent(
            llm=self.llm_editing,
            name="Senior Editor",
            system_prompt="Senior editor who polishes drafts to publish-ready quality.",
        )
        self.evaluation_agent = EvaluationAgent(
            llm=self.llm_editing,
            name="Chief Content Officer",
            system_prompt="Chief Content Officer who evaluates content quality against publishing standards.",
        )
        self.image_agent = ImageAgent(
            llm=self.llm_image,
            name="Visual Designer",
            system_prompt="Designer who produces on-brand imagery prompts for Career Compass.",
        )
        self.publisher = PublishingAgent(
            llm=self.llm_metadata,
            name="Publishing QA",
            system_prompt="Ensures metadata and publishing quality for Career Compass posts.",
            settings=self.settings,
        )

    def run_workflow(self, existing_content: str, context: AgentContext) -> WorkflowResult:
        research_result = self.research_agent.recommend(existing_content, context)
        research_payload = _safe_parse(research_result.raw_response)
        
        print(f"📋 Research payload keys: {list(research_payload.keys())}")
        
        # Get separate contexts for different agent types
        text_context = self.rag.as_text_context(query=research_payload.get("topic", "Career Compass"))
        image_context = self.rag.as_image_context(query=research_payload.get("topic", "Career Compass"))

        blog_result = self.writer_agent.draft(research_payload, text_context, context)
        blog_content = blog_result.content.get("message", str(blog_result.raw_response))

        # Run editing/refinement step with evaluation loop (max 2 iterations)
        edited_result = self.editor_agent.refine(blog_content)
        edited_content = edited_result.content.get("message", str(edited_result.raw_response))
        
        max_revisions = 2
        for revision_count in range(max_revisions):
            # Evaluate the edited content
            evaluation = self.evaluation_agent.evaluate(edited_content)
            
            # If content passes threshold or we've hit max revisions, proceed
            if evaluation.passes_threshold or revision_count == max_revisions - 1:
                if not evaluation.passes_threshold:
                    print(f"⚠️  Content below threshold after {max_revisions} revisions (score: {evaluation.average_score:.1f}/10). Proceeding to publish.")
                else:
                    print(f"✅ Content approved (score: {evaluation.average_score:.1f}/10) after {revision_count + 1} iteration(s)")
                break
            
            # Content needs revision - format feedback and re-edit
            print(f"🔄 Revision {revision_count + 1}: Content needs improvement (score: {evaluation.average_score:.1f}/10)")
            feedback_text = self.evaluation_agent.format_feedback_for_editor(evaluation)
            edited_result = self.editor_agent.refine(edited_content, feedback=feedback_text)
            edited_content = edited_result.content.get("message", str(edited_result.raw_response))

        image_result = self.image_agent.design(research_payload, image_context, context)
        
        # Check if image agent returned an error
        if isinstance(image_result.content, dict) and "error" in image_result.content:
            print(f"⚠️  Image generation failed: {image_result.content.get('error')}")
            print(f"    Raw response: {image_result.content.get('raw', 'N/A')[:200]}")
            image_plan = {}
        else:
            image_plan = _safe_parse(image_result.raw_response)
        
        print(f"📸 Image plan generated: {bool(image_plan.get('prompt'))}")

        # If an image prompt was produced, generate and upload the image locally before publishing
        image_file_path = None
        if image_plan and image_plan.get("prompt"):
            gen = ImageGenerator()
            # prefer the configured OpenAI key from settings rather than relying on env
            gen.openai_api_key = self.settings.openai_api_key
            # Determine size: prefer explicit `size`, else map `aspect_ratio` like '16:9' to a concrete pixel size.
            size = None
            if image_plan.get("size"):
                size = image_plan.get("size")
            elif image_plan.get("aspect_ratio"):
                ar = str(image_plan.get("aspect_ratio")).strip()
                if ar in ("16:9", "16x9", "16by9"):
                    # DALL-E 3 supports: 1024x1024, 1792x1024, or 1024x1792
                    size = "1792x1024"
                elif ar in ("4:3", "4x3", "4by3"):
                    # map to the closest supported landscape size
                    size = "1792x1024"
                elif ar in ("1:1", "1x1"):
                    size = "1024x1024"
                else:
                    # default fallback to a square-ish image if unknown
                    size = "1024x1024"
            else:
                size = "1024x1024"

            # Encourage the model to compose for the requested aspect ratio and fill the entire frame.
            # STRATEGY: Be very specific about distributing elements LEFT, CENTER, and RIGHT
            prompt_text = image_plan.get("prompt")
            if image_plan.get("aspect_ratio"):
                # Use explicit spatial positioning to force horizontal spread
                prompt_text = (
                    f"WIDE CINEMATIC PANORAMA in {image_plan.get('aspect_ratio')} format. "
                    f"COMPOSITION STRUCTURE: Divide the canvas into THREE ZONES - "
                    f"LEFT THIRD has visual elements, CENTER THIRD has visual elements, RIGHT THIRD has visual elements. "
                    f"Elements flow and connect across all three zones creating a continuous horizontal narrative. "
                    f"{prompt_text} "
                    f"CRITICAL SPATIAL REQUIREMENTS: Visual elements must extend from the FAR LEFT EDGE all the way to the FAR RIGHT EDGE. "
                    f"Use repeating patterns, flowing lines, or distributed subjects that span the FULL HORIZONTAL WIDTH. "
                    f"Think ULTRA-WIDE landscape photography or cinema aspect ratio - NEVER a centered square subject with empty sides."
                )
            else:
                prompt_text = f"FULL-FRAME COMPOSITION: {prompt_text} CRITICAL: Create a full-frame composition that fills the entire canvas edge-to-edge with no borders, padding, or empty space around the illustration."

            try:
                img_path = gen.generate(prompt_text, size=size)
                image_file_path = str(img_path)
            except Exception:
                image_file_path = None

        # Use the edited content for metadata generation and publishing
        metadata_result = self.publisher.prepare_metadata(edited_content)
        metadata_payload = _safe_parse(metadata_result.raw_response)
        publish_result = self.publisher.upload_to_webflow(metadata_payload, edited_content, image_file_path=image_file_path)

        return WorkflowResult(
            research=research_payload,
            blog_post=edited_content,
            image_plan=image_plan,
            publish_result=publish_result,
        )


def _safe_parse(raw: str) -> Dict[str, str]:
    if not raw:
        return {}
    # If the model wrapped JSON in code fences or added a language tag, extract the JSON block.
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        import re

        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}
