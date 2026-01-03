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

        metadata_result = self.publisher.prepare_metadata(blog_content)
        metadata_payload = _safe_parse(metadata_result.raw_response)
        publish_result = self.publisher.upload_to_webflow(
            metadata_payload,
            blog_content,
            image_url=image_plan.get("image_url"),
            image_file_path=image_plan.get("image_file_path"),
            thumb_image_url=image_plan.get("thumbnail_url"),
        )

        return WorkflowResult(
            research=research_payload,
            blog_post=blog_content,
            image_plan=image_plan,
            publish_result=publish_result,
        )


def _safe_parse(raw: str) -> Dict[str, str]:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
