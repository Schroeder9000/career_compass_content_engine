from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from langchain_core.language_models.chat_models import BaseChatModel

from .base import AgentResult, BaseAgent


EDIT_TEMPLATE = """
You are the senior editor for Career Compass.

Goal:
Turn the draft into something 80–90% publish-ready.

Editing rules:
- Keep Markdown headings and structure.
- Tighten the hook. Remove clichés and filler.
- Increase specificity: add concrete examples, scripts, checklists where thin.
- Improve voice: more natural cadence, fewer repeated phrases.
- Remove any “AI-sounding” lines or generic summaries.
- Keep paragraphs short (1–4 sentences).
- Do NOT add external citations or stats.
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
