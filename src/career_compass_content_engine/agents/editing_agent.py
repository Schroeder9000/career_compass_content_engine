from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from langchain_core.language_models.chat_models import BaseChatModel

from .base import AgentResult, BaseAgent


EDIT_TEMPLATE = """
You are the senior editor for Career Compass, an online career guidance platform.

Goal:
Turn the draft into something 90% publish-ready. Ensuring that the content is clear, on topic, and engaging. Don't accept any substandard writing or advice provided. Ensure that this is the highest level of thought leadership, with a delightful brand voice, comprised of equal parts wisdom and humor.

Editing rules:
- Keep Markdown headings and structure.
- EXPAND, don't just tighten: Add depth, examples, and substance where sections feel thin
- Target: 2,500-3,500 words for the final piece (check and expand if needed)
- Each major section should be 300-500 words minimum - ADD content if sections are too short
- Remove clichés and filler, but REPLACE them with specific examples or deeper insights
- EXPAND with: more concrete examples, specific scenarios with details, exact dialogue/scripts, step-by-step breakdowns
- Improve on brand voice: be aspirational while also finding moments of comedy and highlighting the absurdities of Corporate America.
- Remove any "AI-sounding" lines or generic summaries - replace with specific, opinionated takes.
- Add emotional depth: name feelings explicitly (e.g., "the knot in your stomach when...", "the relief of finally...")
- Natural Career Compass integration: weave it into the conclusion organically, not as a separate section

Instructions on style and tone:
- Push the writing to be more elevated, long-form thought leadership. Go on deep dives when appropriate.
- Expand content where ideas are underdeveloped and take interesting positions. Our goal is to support with emotional and mental guidance, as well as tactical execution.
- Warm, candid, practical, and funny with an ironic wink toward corporate culture.

Return ONLY the improved Markdown.

DRAFT:
{draft}
"""


@dataclass
class EditingAgent(BaseAgent):
    """Agent that edits and polishes a draft Markdown post."""

    llm: BaseChatModel

    def refine(self, draft_markdown: str) -> AgentResult:
        # Strip markdown code fences if present in the draft
        draft_cleaned = draft_markdown.strip()
        if draft_cleaned.startswith('```markdown') or draft_cleaned.startswith('```'):
            # Remove opening fence
            lines = draft_cleaned.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            # Remove closing fence
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            draft_cleaned = '\n'.join(lines)
        
        prompt = EDIT_TEMPLATE.format(draft=draft_cleaned)
        result = self.run(prompt)
        
        # Also strip from the result if the LLM adds them back
        if result.raw_response:
            cleaned_response = result.raw_response.strip()
            if cleaned_response.startswith('```markdown') or cleaned_response.startswith('```'):
                lines = cleaned_response.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                cleaned_response = '\n'.join(lines)
                # Update the result
                result.content["message"] = cleaned_response
                result.raw_response = cleaned_response
        
        return result
