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
        
        # Get separate contexts for different agent types
        text_context = self.rag.as_text_context(query=research_payload.get("topic", "Career Compass"))
        image_context = self.rag.as_image_context(query=research_payload.get("topic", "Career Compass"))

        blog_result = self.writer_agent.draft(research_payload, text_context, context)
        blog_content = blog_result.content.get("message", str(blog_result.raw_response))

        # Run editing/refinement step
        edited_result = self.editor_agent.refine(blog_content)
        edited_content = edited_result.content.get("message", str(edited_result.raw_response))

        image_result = self.image_agent.design(research_payload, image_context, context)
        image_plan = _safe_parse(image_result.raw_response)

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
            prompt_text = image_plan.get("prompt")
            if image_plan.get("aspect_ratio"):
                prompt_text = f"{prompt_text} CRITICAL COMPOSITION REQUIREMENTS: This must be a WIDE HORIZONTAL landscape illustration that completely fills the entire {image_plan.get('aspect_ratio')} canvas from edge to edge. DO NOT create a square or portrait composition centered in empty space. DO NOT leave any blank margins, borders, or padding. The illustration must extend fully to the left edge, right edge, top edge, and bottom edge. Use the full WIDTH of the landscape format and spread the design elements horizontally across the entire frame."
            else:
                prompt_text = f"{prompt_text} CRITICAL: Create a full-frame composition that fills the entire canvas edge-to-edge with no borders, padding, or empty space around the illustration."

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
