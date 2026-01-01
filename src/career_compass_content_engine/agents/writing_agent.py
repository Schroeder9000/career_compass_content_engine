from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from langchain_core.language_models.chat_models import BaseChatModel

from .base import AgentContext, AgentResult, BaseAgent


WRITING_TEMPLATE = """
You are the lead writer for Career Compass.
Write a full blog post that matches the existing brand voice and tone.

Topic: {topic}
Summary: {summary}
Outline: {outline}

Brand voice and audience: {context}
Ground the writing in the provided examples of prior content:
{examples}

Return Markdown with clear headings, subheadings, and call-to-actions for Career Compass.
Keep paragraphs concise, friendly, and actionable.
"""


@dataclass
class WritingAgent(BaseAgent):
    """Agent that drafts the full blog post."""

    llm: BaseChatModel

    def draft(self, topic_payload: Dict[str, str], rag_context: str, context: AgentContext) -> AgentResult:
        prompt = WRITING_TEMPLATE.format(
            topic=topic_payload.get("topic", ""),
            summary=topic_payload.get("summary", topic_payload.get("rationale", "")),
            outline=topic_payload.get("outline", ""),
            context=f"Voice: {context.brand_voice}\nAudience: {context.target_audience}",
            examples=rag_context or "(No examples provided; default to Career Compass tone.)",
        )
        return self.run(prompt)
