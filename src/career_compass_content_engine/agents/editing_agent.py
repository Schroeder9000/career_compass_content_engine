from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from typing import Optional

from ..product_context import product_context
from ..editorial_policy import article_context
from .base import AgentResult, BaseAgent


EDIT_TEMPLATE = """
You are the senior editor for Career Compass, an online career guidance platform.

Goal:
Turn the draft into something 90% publish-ready. Ensuring that the content is clear, on topic, and engaging. Don't accept any substandard writing or advice provided. Ensure that this is the highest level of thought leadership, with a delightful brand voice, comprised of equal parts wisdom and humor.

Editing rules:
- Keep Markdown headings and structure.
- Length guidance: usually 1,000-2,200 words, but prioritize complete, high-quality coverage over hitting a rigid target.
- Preserve a coherent structure; add or merge H2 sections when needed for flow.
- Tighten and sharpen, don't pad: cut filler, replace vague sentences with specific ones, but don't add new sections or bulk up thin sections beyond what's needed for the argument.
- Remove clichés and filler, REPLACE with sharper, more specific language — not more words
- If a section is genuinely thin, add one strong sentence or concrete detail — not a whole new paragraph
- Improve on brand voice: be aspirational while also finding moments of comedy and highlighting the absurdities of Corporate America.
- Remove any "AI-sounding" lines or generic summaries - replace with specific, opinionated takes.
- Add emotional depth: name feelings explicitly (e.g., "the knot in your stomach when...", "the relief of finally...")
- Preserve the assigned reader need. Remove forced product connections and sales pivots; no brand mention is required.

Four specific failure modes to check and fix:
1. OPENING HOOK: If the article opens with "Picture this: It's [day/time]..." or any stamped-scene
   variation, rewrite the first 2-3 paragraphs entirely. Replace with a direct observation, a
   provocative claim, a sharp question, or a brief first-person anecdote. The goal is to hook
   with an idea, not a scene-setting device.
2. BRAND VOICE IN INTRO & CONCLUSION: These are the highest-risk zones for AI-sounding language.
   Read the opening two paragraphs and the closing two paragraphs closely. Hunt for and replace:
   - Hype phrases: "game-changer", "secret weapon", "unlock opportunities", "skyrocket", "catapult"
   - Vague stakes-setting: "cutthroat job market", "fast-paced world", "more competitive than ever"
   - Empty escalators: "the difference can be monumental", "opening doors that would otherwise remain closed"
   Replace each one with a specific, opinionated, concrete sentence that makes the same point
   without reaching for a cliché. The intro should sound like a smart friend talking, not a LinkedIn post.
3. LISTICLE FEEL: If the article reads like a numbered roundup (each H2 is just the next item
   in a list with no connecting argument), reframe it as thought-leadership. Add transitional
   reasoning between sections that shows how each point builds on the last. The article should
   feel like it's advancing a thesis, not ticking boxes.
4. LATE-SECTION THINNESS: Read the second half of the article carefully. If sections feel repetitive or padded, cut the weakest paragraphs — don't add more. Depth comes from specificity and argument, not length.

Instructions on style and tone:
- Push the writing to be more elevated, long-form thought leadership. Go on deep dives when appropriate.
- Expand content where ideas are underdeveloped and take interesting positions. Our goal is to support with emotional and mental guidance, as well as tactical execution.
- Warm, candid, practical, and funny with an ironic wink toward corporate culture.

Return ONLY the improved Markdown.

DRAFT:
{draft}
"""

REVISION_TEMPLATE = """
You are the senior editor for Career Compass, an online career guidance platform.

This draft has been evaluated and scored below our publishing threshold.
Your job is to substantially rewrite it to fix every issue called out below.
This is NOT a light polish — make bold, significant changes.

EVALUATION FEEDBACK:
{feedback}

CURRENT VERSION:
{draft}

---

Revision instructions (apply ALL of these):
- Attack every CRITICAL ISSUE head-on with concrete rewrites, not tweaks
- Sections scored below 8: restructure or sharpen them — don't necessarily expand; fix the argument or specificity
- Length guidance: usually 1,000-2,200 words, but prioritize completeness and quality over strict limits.
- Brand voice: replace any bland, generic, or AI-sounding passages with opinionated,
  funny, candid writing that has a strong point of view
- Emotional depth: explicitly name the feelings a reader would have at each stage
  (the dread of Sunday night, the electric relief of a good 1:1, etc.)
- Actionability: every section must contain something the reader can DO — but VARY how you deliver it.
  Do NOT close every section with the same bold "**Action for This Week:**" label; that pattern becomes
  wallpaper and readers stop reading it. Instead, rotate across these delivery styles:
    * Woven into the final paragraph as a natural sentence ("The move this week is to...")
    * A direct imperative that flows from the argument ("Pick one stakeholder and send that email today.")
    * A concrete question for the reader to sit with
    * Only occasionally as a bold callout — and vary the label ("Try This", "Your Move", etc.)
  The goal is that every section leaves the reader knowing what to do next, without every section
  sounding identical.
- Vary paragraph length and rhythm aggressively — no three paragraphs in a row
  with the same sentence count
- Kill every cliché and replace it with a specific, vivid, opinionated sentence
- You MAY restructure, reorder, or partially rewrite any section to fix the issues
- End with a strong, topic-specific insight and next step. Do not add a product mention to satisfy a supposed brand requirement.

Return ONLY the revised Markdown.
"""


@dataclass
class EditingAgent(BaseAgent):
    """Agent that edits and polishes a draft Markdown post."""

    llm: BaseChatModel

    def refine(self, draft_markdown: str, feedback: Optional[str] = None, *, editorial_brief: Optional[dict] = None) -> AgentResult:
        """
        Edit and refine a draft blog post.
        
        Args:
            draft_markdown: The draft content to edit
            feedback: Optional evaluation feedback for revision
            
        Returns:
            AgentResult with the edited content
        """
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
        
        # Choose template based on whether we have feedback
        if feedback:
            prompt = REVISION_TEMPLATE.format(draft=draft_cleaned, feedback=feedback)
        else:
            prompt = EDIT_TEMPLATE.format(draft=draft_cleaned)
        
        result = self.run(prompt + product_context() + article_context(editorial_brief))
        
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
