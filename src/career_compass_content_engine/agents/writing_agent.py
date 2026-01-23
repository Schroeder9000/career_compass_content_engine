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
Suggested Title: {title}
Summary: {summary}

Outline (follow it, but you may refine headings for flow):
{outline}

IMPORTANT: Use the suggested title provided above as your H1, or make only minor refinements if absolutely necessary for clarity. Do NOT change the core structure or starting words of the title.

Brand context:
{context}

Grounding examples (style reference only; do NOT copy):
{examples}

---

Before you write, decide on a clear POV:
- What is the #1 mistake people make on this topic?
- What should they do instead (your framework)?
- What is a slightly contrarian but true take you can defend?

Write a complete blog post in Markdown (1,800–3,000 words minimum) with the structure below.

CRITICAL: DO NOT label structural elements like "Hook" or "CTA" as visible headings. These are instructions for you, not section titles.

REQUIRED STRUCTURE
1) H1 title (use the suggested title provided)

2) Opening hook (NO HEADING): 3-6 short, punchy paragraphs that:
   - Paint a vivid, relatable pain point or scenario
   - Explain why this matters NOW for early-career professionals
   - Set up your contrarian or nuanced take
   - Lead naturally into the substance

3) TL;DR section (use heading "## TL;DR"): 5-8 specific, non-obvious bullets that preview your key insights

4) Main content sections (H2/H3): Follow the outline provided, but adapt headings for natural flow
   - CRITICAL DEPTH REQUIREMENT: Each major H2 section MUST be 300-500 words minimum (5-8 paragraphs)
   - Each H3 subsection should be 150-250 words (3-4 paragraphs)
   - MANDATORY: Include 2-3 SPECIFIC, DETAILED examples per major section with names, situations, dialogue
   - MANDATORY: Provide exact scripts with actual phrasing (not just "say something like...")
   - MANDATORY: Build a framework or mental model with a memorable name
   - CONTRARIAN requirement: Challenge at least one piece of conventional wisdom per major section
   - EMOTIONAL depth: Explicitly name the feelings (anxiety, imposter syndrome, frustration) and validate them
   - TACTICAL depth: Give step-by-step instructions with actual timeframes and metrics
   - Use VARIED paragraph lengths: alternate between 1-2 sentence punchy paragraphs and 5-6 sentence analytical ones
   - SHOW YOUR REASONING: Explain WHY advice works, not just WHAT to do

5) Additional value-add sections (if relevant to topic - NO GENERIC HEADINGS):
   - Common mistakes section with specific scenarios
   - A multi-day or multi-week action plan
   - Scripts for difficult conversations
   - Decision frameworks or self-assessment tools
   - Emotional management strategies

6) Career Compass mention (NO HEADING - integrate naturally): 
   - In your final section or conclusion, naturally mention how Career Compass helps:
   - Build a personalized growth plan
   - Track weekly wins and career metrics  
   - Get weekly coaching nudges via email
   - Keep it to 2-3 sentences max, woven into the narrative

7) Closing (NO HEADING): 2-4 paragraphs that:
   - Synthesize your key insight without repeating the TL;DR
   - Provide encouragement that feels personal, not generic
   - Leave readers with a concrete next step or mindset shift

HUMANNESS & QUALITY RULES
- DEPTH OVER BREADTH: Fewer sections with more substance beats many shallow sections
- SPECIFIC OVER GENERIC: Use real examples, actual numbers, specific scenarios (e.g., "your Tuesday morning standup" not "team meetings")
- VARIED RHYTHM: Mix 1-sentence paragraphs with 4-5 sentence analytical paragraphs
- SHOW YOUR WORK: Explain the "why" behind advice, include the reasoning and tradeoffs
- CONTRARIAN WELCOME: Challenge conventional career advice when you have a defensible alternative view
- NO FLUFF PHRASES: Avoid "let's dive in", "in today's world", "unlock your potential", "game-changer", "at the end of the day"
- NO RECAP PARAGRAPHS: Don't summarize what you just said - keep moving forward
- NO INVENTED DATA: Do not cite statistics or studies unless provided in the context
- TACTICAL + EMOTIONAL: Balance concrete actions with emotional/relational insights
- Each major section should feel like a mini-essay with a clear argument or framework

CONTENT REQUIREMENTS FOR LONG-FORM:
- Include at least 3-4 specific examples or scenarios per main section
- Provide exact scripts or phrases for difficult conversations when relevant
- Build frameworks or mental models readers can apply (give them names if helpful)
- Address the emotional or social dynamics explicitly - the awkwardness, the fear, the office politics
- Challenge at least one piece of conventional wisdom with a nuanced take
- Give readers decision-making frameworks, not just "tips"

OUTPUT FORMAT:
- Output ONLY the blog post content in pure Markdown format
- DO NOT wrap your output in code blocks or ```markdown fences
- DO NOT include any meta-commentary, explanations, or notes
- Start directly with the H1 title
- End with the final closing paragraph
"""

@dataclass
class WritingAgent(BaseAgent):
    """Agent that drafts the full blog post."""

    llm: BaseChatModel

    def draft(self, topic_payload: Dict[str, str], rag_context: str, context: AgentContext) -> AgentResult:
        prompt = WRITING_TEMPLATE.format(
            topic=topic_payload.get("topic", ""),
            title=topic_payload.get("suggested_title", ""),
            summary=topic_payload.get("summary", topic_payload.get("rationale", "")),
            outline=topic_payload.get("outline", ""),
            context=f"Voice: {context.brand_voice}\nAudience: {context.target_audience}",
            examples=rag_context or "(No examples provided; default to Career Compass tone.)",
        )
        return self.run(prompt)
