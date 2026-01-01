from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_core.language_models.chat_models import BaseChatModel

from .base import AgentContext, AgentResult, BaseAgent


RESEARCH_TEMPLATE = """
You are the SEO and content strategy agent for Career Compass.
You receive the following context about previous posts and brand guidelines.

Existing content snippets:
{existing_content}

Brand voice and audience: {context}

Perform lightweight SEO research using intuition, propose the next blog topic, and outline.
Return a concise JSON block with keys: topic, rationale, target_keywords (list), summary, outline (list of sections), and suggested_title.
"""


@dataclass
class ResearchAgent(BaseAgent):
    """Agent that recommends the next blog topic and outline."""

    llm: BaseChatModel

    def recommend(self, existing_content: str, context: AgentContext) -> AgentResult:
        prompt = RESEARCH_TEMPLATE.format(
            existing_content=existing_content or "No prior content provided.",
            context=f"Voice: {context.brand_voice}\nAudience: {context.target_audience}",
        )
        return self.run(prompt)
