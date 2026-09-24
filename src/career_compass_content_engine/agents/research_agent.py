from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
import json

from ..product_context import product_context
from .base import AgentContext, AgentResult, BaseAgent


RESEARCH_TEMPLATE = """
You are the Research, SEO, and Answer Engine Optimization (AEO) strategist for Career Compass,
the company behind Promotion Case, a promotion and pay-conversation preparation product.

Your job is to recommend the *next* blog post topic that maximizes:
1) Traditional SEO (search intent, long-tail capture, internal linking)
2) AEO / LLM surfacing (content that is easy to quote, cite, summarize, and recommend)

Be rigorous, not fluffy. Avoid generic topics that have been covered 1,000 times.

---

### INPUTS

Brand voice and audience context:
{context}

Existing content (snippets or structured data like title, slug, summary, tags, publish date, headings):
{existing_content}

---

### REAL PERFORMANCE DATA
Use this to identify traffic patterns, ranking opportunities, and content gaps.
If a query has high impressions but low CTR, it signals a topic worth writing about.
If a page has high traffic but low engagement, it may need a companion piece or update.

{performance_context}

---

### LIVE SERP COMPETITIVE SNAPSHOT
Use this to understand what currently ranks for your target keywords.
DO NOT repeat the same angles as competitors. Choose a more specific, contrarian, or deeper take.

{competitor_context}

---

### WHAT TO DO (STRICT PROCESS)

1) Corpus audit:
    - Identify the main topic clusters already covered.
    - Identify gaps: questions not answered, missing foundational pages, missing “how-to” guides, missing templates/scripts/checklists, missing manager/career scenarios.
    - Identify risk of keyword/topic cannibalization with existing posts.

2) Opportunity selection (SEO + AEO):
    Choose ONE next topic that is:
    - High intent (solves a real early-career problem)
    - Narrow and specific (long-tail) but scalable into a series
    - “Answerable” (supports a crisp definition, steps, examples, checklist)
    - Citable (contains frameworks, numbered steps, tables/checklists, short definitions)
    - Naturally aligned with Career Compass (growth, clarity, confidence, practical action)    
    IMPORTANT: Title diversity & content format bias
    - DO NOT start titles with "Navigating" - this word is overused and should be avoided entirely
    - Vary your title structure. Avoid repetitive patterns like "Navigating..." or starting every title with "How to..."
    - AVOID pure list-format topics ("X Ways to...", "X Tips for...") as the primary format unless
      the topic genuinely requires enumeration. Prefer argument-driven, essay-style topics that
      build a central thesis. List formats can be used as a section structure inside the outline,
      but the overall article should feel like thought-leadership, not a roundup.
    - When structuring the outline, favor a narrative arc over a sequential checklist: the sections
      should advance a central argument rather than simply be the next item in a list.
    - Mix patterns: questions, contrarian takes, direct statements, frameworks, numbered lists
    - Examples of diverse patterns:
      * "The [Number] [Thing] That [Result]" (e.g., "The 5 Mistakes That Tank Your First 90 Days")
      * "[Action] Your Way to [Result]" (e.g., "Talk Your Way Into That Promotion")
      * "Why [Belief] Is Wrong About [Topic]" (e.g., "Why 'Fake It Till You Make It' Fails at Work")
      * "[Topic]: A [Framework/Guide] for [Audience]" (e.g., "Performance Reviews: A Survival Guide for New Managers")
      * Direct value prop (e.g., "Stop Overthinking Your Career Moves")
      * Question format (e.g., "Should You Really Take That Job Offer?")
      * Bold statement (e.g., "Your First 90 Days Don't Have to Be Perfect")
3) Make it AEO-first:
    - Provide 8–15 target questions (query-style) the post will answer.
    - Provide “quotable blocks” recommendations (e.g., TL;DR, definitions, step-by-step, do/don’t list).
    - Recommend 1–2 schema types to add (Article + FAQPage or HowTo where appropriate).

4) Create an outline designed for:
    - Thought leadership essay structure: 3–4 H2 sections maximum, each advancing a central argument
    - NO FAQ section, NO playbook/checklist section, NO "Common Mistakes" list in the outline
    - Skimmability (clear H2/H3)
    - Fast extraction for AI answers (short sections, crisp bullets, explicit definitions)
    - Internal linking to existing posts (name which posts to link to by slug/title if possible)

---

### CONSTRAINTS / QUALITY BAR

- No vague topics like “How to be successful at work” or “Productivity tips”.
- Prefer topics that match early-career moments: first 90 days, 1:1s, performance reviews, promotion path, managing up, work stress, communication, conflict, stakeholder alignment, learning plans.
- Avoid medical/legal/financial advice.
- Do NOT invent performance data or claim rankings. Use only what’s provided.

---

### OUTPUT FORMAT (STRICT JSON ONLY)

Return a single JSON object with keys:

- topic (short descriptor)
- suggested_title (SEO-friendly, ≤ 60 chars preferred)
- rationale (3–6 bullets, include why it’s winnable + AEO-friendly)
- target_keywords (5–10 strings; include long-tail variants)
- target_questions (8–15 strings; phrased as real queries)
- outline (array of section objects: {{heading, goal, bullets}} — EXACTLY 3–4 items, no more)
- internal_links (array of objects: {{target_post, anchor_text, reason}})
- suggested_schema (array of strings, e.g. ["FAQPage","HowTo"]) 
- cannibalization_risks (array of strings; empty if none)
- series_followups (array of 3 next topics if this performs well)
- content_assets (array: e.g. ["checklist","template","examples","scripts"]) 
- confidence (0.0–1.0)
- data_needed (array of what to fetch next if missing, e.g. ["full post titles/slugs","Search Console queries"]) 

Do not include any commentary outside the JSON.
"""


@dataclass
class ResearchAgent(BaseAgent):
    """Agent that recommends the next blog topic and outline."""

    llm: BaseChatModel

    def recommend(
        self,
        existing_content: str,
        context: AgentContext,
        performance_context: Optional[str] = None,
        competitor_context: Optional[str] = None,
    ) -> AgentResult:
        prompt = RESEARCH_TEMPLATE.format(
            existing_content=existing_content or "No prior content provided.",
            context=f"Voice: {context.brand_voice}\nAudience: {context.target_audience}",
            performance_context=performance_context or "No GSC/GA performance data provided.",
            competitor_context=competitor_context or "No SERP competitive data provided.",
        )
        prompt += product_context()
        # Invoke model and parse JSON output robustly
        try:
            response = self.llm.invoke(prompt)
            raw = None
            if isinstance(response, str):
                raw = response
            else:
                for attr in ("content", "text", "message", "data"):
                    val = getattr(response, attr, None) if not isinstance(response, dict) else response.get(attr)
                    if val:
                        raw = val
                        break
            if raw is None:
                raw = str(response)
        except Exception:
            result = self.run(prompt)
            raw = result.content.get("message")

        parsed = None
        try:
            parsed = raw if isinstance(raw, dict) else json.loads(raw)
            return AgentResult(content=parsed, raw_response=json.dumps(parsed))
        except Exception:
            # Try to extract a JSON object from the raw text as a fallback
            try:
                import re

                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    candidate = m.group(0)
                    parsed = json.loads(candidate)
                    return AgentResult(content=parsed, raw_response=json.dumps(parsed))
            except Exception:
                parsed = None

        # If still not parsed, ask the model to reformat the previous analysis into strict JSON
        try:
            reform_prompt = (
                "Reformat the following research analysis into a strict JSON object with keys: topic, suggested_title, rationale, "
                "target_keywords, target_questions, outline, internal_links, suggested_schema, cannibalization_risks, "
                "series_followups, content_assets, confidence, data_needed.\n\nAnalysis:\n" + str(raw)
            )
            reform = self.run(reform_prompt)
            reform_raw = reform.raw_response
            parsed = json.loads(reform_raw) if isinstance(reform_raw, str) else reform_raw
            return AgentResult(content=parsed, raw_response=json.dumps(parsed))
        except Exception as e:
            # As a last-resort fallback, synthesize a minimal research payload heuristically
            try:
                short_topic = (str(raw).splitlines()[0] or "Suggested Topic")[:60]
            except Exception:
                short_topic = "Suggested Topic"
            fallback = {
                "topic": short_topic,
                "suggested_title": short_topic,
                "rationale": ["Fallback: original model output was not structured JSON; use as draft idea."],
                "target_keywords": [],
                "target_questions": [],
                "outline": [],
                "internal_links": [],
                "suggested_schema": [],
                "cannibalization_risks": [],
                "series_followups": [],
                "content_assets": [],
                "confidence": 0.3,
                "data_needed": ["full post titles/slugs", "Search Console queries"],
            }
            return AgentResult(content=fallback, raw_response=json.dumps(fallback))
