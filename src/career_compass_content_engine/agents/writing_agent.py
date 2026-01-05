from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from langchain_core.language_models.chat_models import BaseChatModel

from .base import AgentContext, AgentResult, BaseAgent


WRITING_TEMPLATE = """
You are the lead writer for Career Compass: a premium career development and coaching app for early-career professionals.

Write like a real human:
- Warm, candid, practical, lightly funny when appropriate
- 8th-grade reading level (simple words, short sentences)
- No corporate filler, no generic motivation
- Specific, actionable guidance with examples and scripts

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

Write a complete blog post in Markdown (1,200–1,800 words) with the structure below.

REQUIRED STRUCTURE
1) H1 title (not clickbait)
2) Hook (2–5 short paragraphs): vivid pain point + why it matters now
3) TL;DR (5–7 bullets): specific, non-obvious
4) Main sections (H2/H3) that include:
   - A simple framework with a memorable name (e.g., “The 3-Part 1:1 Win”)
   - At least 2 “say this / not that” examples
   - At least 1 ready-to-send message (email/Slack) template
   - At least 1 checklist (10–15 bullets)
   - At least 1 short “real-life moment” vignette (plausible, anonymous)
5) Common mistakes + fixes (bulleted)
6) A 7–14 day mini-plan (5–10 actions)
7) Career Compass CTA (not salesy): 2–3 sentences + 3 bullets:
   - build a growth plan
   - track weekly wins + career metrics
   - get weekly coaching nudges (email/push)
8) Closing paragraph: personal, encouraging, not generic

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
