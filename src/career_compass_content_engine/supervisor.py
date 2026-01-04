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
from .agents.image_generator import ImageGenerator
from .config import Settings, get_settings
from .rag import RAGPipeline


def build_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key, temperature=0.4)


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
        self.llm = build_llm(self.settings)
        self.rag = rag_pipeline or RAGPipeline.from_examples()
        self.research_agent = ResearchAgent(
            llm=self.llm,
            name="SEO Researcher",
            system_prompt="Strategic planner who identifies high-impact Career Compass topics.",
        )
        self.writer_agent = WritingAgent(
            llm=self.llm,
            name="Content Writer",
            system_prompt="Principal writer who crafts concise, actionable Career Compass blogs.",
        )
        self.image_agent = ImageAgent(
            llm=self.llm,
            name="Visual Designer",
            system_prompt="Designer who produces on-brand imagery prompts for Career Compass.",
        )
        self.publisher = PublishingAgent(
            llm=self.llm,
            name="Publishing QA",
            system_prompt="Ensures metadata and publishing quality for Career Compass posts.",
            settings=self.settings,
        )

    def run_workflow(self, existing_content: str, context: AgentContext) -> WorkflowResult:
        research_result = self.research_agent.recommend(existing_content, context)
        research_payload = _safe_parse(research_result.raw_response)
        rag_context = self.rag.as_context(query=research_payload.get("topic", "Career Compass"))

        blog_result = self.writer_agent.draft(research_payload, rag_context, context)
        blog_content = blog_result.content.get("message", str(blog_result.raw_response))

        image_result = self.image_agent.design(research_payload, rag_context, context)
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
                    # OpenAI Images supports a limited set of sizes; pick the widest landscape option available.
                    size = "1536x1024"
                elif ar in ("4:3", "4x3", "4by3"):
                    # map to the closest supported landscape size
                    size = "1536x1024"
                elif ar in ("1:1", "1x1"):
                    size = "1024x1024"
                else:
                    # default fallback to a square-ish image if unknown
                    size = "1024x1024"
            else:
                size = "1024x1024"

            # Encourage the model to compose for the requested aspect ratio in the prompt text.
            prompt_text = image_plan.get("prompt")
            if image_plan.get("aspect_ratio"):
                prompt_text = f"{prompt_text} -- Prefer aspect ratio {image_plan.get('aspect_ratio')} when composing the image."

            try:
                img_path = gen.generate(prompt_text, size=size)
                image_file_path = str(img_path)
            except Exception:
                image_file_path = None

        metadata_result = self.publisher.prepare_metadata(blog_content)
        metadata_payload = _safe_parse(metadata_result.raw_response)
        publish_result = self.publisher.upload_to_webflow(metadata_payload, blog_content, image_file_path=image_file_path)

        return WorkflowResult(
            research=research_payload,
            blog_post=blog_content,
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
