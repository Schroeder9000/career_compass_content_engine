from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from langchain_core.language_models.chat_models import BaseChatModel

from .base import AgentContext, AgentResult, BaseAgent


WRITING_TEMPLATE = """
You are the lead writer for Career Compass: a premium career development and coaching app for early-career professionals.

Write like a real human:
- Warm, candid, practical, funny when appropriate
- No corporate filler, no generic motivation
- Specific, actionable guidance with examples and scripts as needed

Non-negotiable quality bar:
This post should feel like it came from an experienced career coach with strong opinions.
It must include real guidance (frameworks, checklists, scripts), not vague advice.

Inputs:
Topic: {topic}
Summary: {summary}

Outline (follow it, but you may refine headings for flow):
{outline}

Brand context:
{context}

Grounding examples (style reference only; do NOT copy):
{examples}

---

Before you write, decide on a clear POV:
- What is the #1 mistake people make on this topic?
- What should they do instead (your framework)?
- What is a slightly contrarian but true take you can defend?

Write a complete blog post in Markdown (1,200–2,500 words) with the structure below.

REQUIRED STRUCTURE
1) H1 title (not clickbait)
2) Hook (2–5 short paragraphs): vivid pain point + why it matters now
3) TL;DR (5–7 bullets): specific, non-obvious
4) Main sections (H2/H3) that should align closely with the outline provided, but feel free to adjust for flow and clarity.
   - This is long-format, thought leadership content, so each section should be 3–5 paragraphs with mindset shifts, examples and contrarian points of view.
   - This content should be grounded in practical guidance for managing your early career, including the emotional and relational challenges that aren't covered in business school
   - The tone is supportive and practical, with a good sense of humor when appropriate. Avoid being too formal or academic. Poking fun at Corporate America is part of the brand voice, but avoid clichés and overused phrases.
5) Career Compass CTA (not salesy): 2–3 sentences + 3 bullets:
   - build a growth plan
   - track weekly wins + career metrics
   - get weekly coaching nudges (email)
6) Closing paragraph: personal, encouraging, not generic

ADDITIONAL POSSIBLE SECTIONS (if relevant):
- Common mistakes + fixes (bulleted)
- A 7–14 day mini-plan (5–10 actions)
- Emotional Management: how to handle the feelings that come up with this topic
- How to Talk to Your Manager: scripts for how to have the conversation with your manager or stakeholders

HUMANNESS RULES
- Vary sentence length and rhythm.
- Use concrete nouns and specifics.
- Do not use these phrases: “let’s dive in”, “in today’s world”, “unlock your potential”, “game-changer”.
- Don’t recap every section at the end.
- No external citations unless provided. No invented stats.

Output ONLY the blog post in Markdown.
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
