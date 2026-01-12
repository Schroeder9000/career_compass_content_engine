from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from langchain_core.language_models.chat_models import BaseChatModel

from .base import AgentResult, BaseAgent


EDIT_TEMPLATE = """
You are the senior editor for Career Compass.

Goal:
Turn the draft into something 90% publish-ready. Ensuring that the content is clear, on topic, and engaging. Don't accept any substandard writing or advice provided. Ensure that this is the highest level of thought leadership, with a delightful brand voice, comprised of equal parts wisdom and humor.

Editing rules:
- Keep Markdown headings and structure.
- Tighten the hook. 
- Remove clichés and filler.
- Improve on brand voice: be aspirational while also finding moments of comedy and highlighting the absurdities of Corporate America.
- Remove any “AI-sounding” lines or generic summaries.
- Only add external citations or stats if they are directly relevant and add value to the content. Don't add any new content that isn't already in the draft.
- Preserve the CTA section and keep it non-salesy.

Return ONLY the improved Markdown.

DRAFT:
{draft}
"""


@dataclass
class EditingAgent(BaseAgent):
    """Agent that edits and polishes a draft Markdown post."""

    llm: BaseChatModel

    def refine(self, draft_markdown: str) -> AgentResult:
        prompt = EDIT_TEMPLATE.format(draft=draft_markdown)
        return self.run(prompt)
