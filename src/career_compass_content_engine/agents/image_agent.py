from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from langchain_core.language_models.chat_models import BaseChatModel

from .base import AgentContext, AgentResult, BaseAgent


IMAGE_TEMPLATE = """
You are the visual designer for Career Compass.
Create a detailed image prompt for the blog hero illustration.

Topic: {topic}
Summary: {summary}
Brand style: {style}
Color palette: {colors}
Existing visual cues:
{examples}

Return a JSON object with keys: prompt, alt_text, suggested_palette (list of hex), aspect_ratio, and guidance on model settings.
"""


@dataclass
class ImageAgent(BaseAgent):
    """Agent that designs an image prompt consistent with the brand."""

    llm: BaseChatModel

    def design(self, topic_payload: Dict[str, str], rag_context: str, context: AgentContext) -> AgentResult:
        prompt = IMAGE_TEMPLATE.format(
            topic=topic_payload.get("topic", ""),
            summary=topic_payload.get("summary", topic_payload.get("rationale", "")),
            style=context.brand_voice,
            colors=context.brand_colors,
            examples=rag_context or "(No visual examples provided; follow brand colors.)",
        )
        return self.run(prompt)
