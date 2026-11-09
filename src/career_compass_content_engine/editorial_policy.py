"""Audience-first editorial coverage, separate from the app's product facts."""
import json
import re
from collections import Counter


PILLARS = {
    "first_jobs": ("First jobs and transitions", "first 90 days new job early career"),
    "workplace_relationships": ("Communication and workplace relationships", "early career communication feedback workplace conflict"),
    "skills_and_learning": ("Skills and learning", "early career learning skills mentoring"),
    "work_habits": ("Work habits, boundaries and sustainable work", "early career workload boundaries time management"),
    "confidence_and_identity": ("Confidence, belonging and professional identity", "early career confidence belonging workplace mistakes"),
    "career_direction": ("Career direction and job search", "early career choosing a career job search interviews"),
    "workplace_change": ("Workplace change and practical AI literacy", "early career workplace change AI literacy"),
    "advancement_and_pay": ("Advancement and compensation", "early career promotion salary negotiation"),
}
ADVANCEMENT = "advancement_and_pay"
ADVANCEMENT_PATTERN = re.compile(
    r"\b(?:promotions?|promoted|salary|salaries|compensation)\b|"
    r"\bpay (?:raise|rise|negotiation)\b|\b(?:ask|asking|request|requesting|negotiate|negotiating) for (?:a |your )?raise\b",
    re.IGNORECASE,
)
PRODUCT_PATTERN = re.compile(r"\bpromotion\s+case\b|apps\.apple\.com/[^\s)]*6670568018", re.IGNORECASE)
PRODUCT_LINK = re.compile(r"apps\.apple\.com/[^\s)]*6670568018|careeraiapp\.com/(?:promotion-case|help/promotion-case)\b", re.IGNORECASE)

EDITORIAL_POLICY = """
EDITORIAL MISSION — audience needs come before product promotion:
Career Compass publishes useful, distinctive guidance for early-career professionals
across the full experience of starting and building a working life. Cover first jobs,
communication, relationships, learning, confidence, belonging, sustainable work,
career choices, job search, workplace change and AI literacy, as well as advancement.
Career Compass's editorial audience is broader than Promotion Case's product audience.
Use the assigned coverage pillar; do not turn every problem into a promotion or pay case.
Search performance and historical articles inform the angle, not the editorial agenda.
Product facts are an accuracy reference, never a checklist of copy to insert.
No article requires a brand mention, product link, sales CTA, price, or feature summary.
Broad career articles should end with useful, topic-specific guidance without a product pitch.
On a directly relevant advancement/pay article, a product connection is optional: at most
one short paragraph of one or two sentences, no more than 60 words. Omission is fully valid.
Never reward or penalize an article for including or omitting a product mention.
Assess topic fit and usefulness, and reject a forced sales pivot or product boilerplate.
Do not describe the Promotion Case app as a general-purpose coach for these broader topics.
Preserve source accuracy, clearly labeled hypothetical examples, and existing copy rules.
"""


def published_history(records: list[dict]) -> list[dict]:
    """Accept newest-first history, ignoring drafts, failures and duplicate publications."""
    result, seen = [], set()
    for record in records:
        status = record.get("publish_status")
        published = status == "success" or (
            status is None and record.get("webflow_item_id") and record.get("evaluation_passed") is True
        )
        if not published:
            continue
        key = record.get("webflow_item_id") or record.get("slug") or record.get("suggested_title")
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result


def record_pillar(record: dict) -> str:
    # Recognize older promotion-heavy runs that predate explicit editorial tracking.
    subject = " ".join(str(record.get(k, "")) for k in ("topic", "suggested_title", "slug"))
    if ADVANCEMENT_PATTERN.search(subject.replace("-", " ")):
        return ADVANCEMENT
    pillar = record.get("editorial_pillar")
    return pillar if pillar in PILLARS else "unclassified"


def brief_for(pillar: str) -> dict:
    label, seed = PILLARS[pillar]
    return {"pillar": pillar, "coverage": label, "search_seed": seed,
            "product_connection": "optional_brief" if pillar == ADVANCEMENT else "none"}


def choose_editorial_brief(records: list[dict]) -> dict:
    recent = published_history(records)[:10]
    pillars = [record_pillar(r) for r in recent]
    counts = Counter(pillars)
    candidates = list(PILLARS)
    # At most two advancement/pay posts in the next rolling ten; never consecutive.
    if pillars[:1] == [ADVANCEMENT] or pillars[:9].count(ADVANCEMENT) >= 2:
        candidates.remove(ADVANCEMENT)
    # Prefer under-covered pillars, then the least recently used, then stable order.
    chosen = min(candidates, key=lambda p: (counts[p],
        -(pillars.index(p) if p in pillars else len(pillars) + 1), list(PILLARS).index(p)))
    return {**brief_for(chosen), "recent_published_counts": dict(counts)}


def article_context(brief: dict | None = None) -> str:
    context = "\n\n" + EDITORIAL_POLICY
    if brief:
        context += "\nASSIGNED ARTICLE BRIEF (preserve through revisions):\n" + json.dumps(brief, ensure_ascii=False)
    return context


def research_plan_issues(payload: dict, brief: dict) -> list[str]:
    if not isinstance(payload, dict) or not payload.get("topic") or not payload.get("suggested_title"):
        return ["Research must provide a concrete topic and suggested title."]
    subject = f"{payload['topic']} {payload['suggested_title']}"
    if brief["pillar"] != ADVANCEMENT and ADVANCEMENT_PATTERN.search(subject):
        return ["Select a topic within the assigned broad career pillar, not another advancement/pay article."]
    return []


def article_editorial_issues(markdown: str, brief: dict | None = None) -> list[str]:
    """Backstop the evaluation and final metadata repair; never insert promotional copy."""
    issues = []
    heading = re.search(r"^#\s+(.+)$", markdown or "", re.MULTILINE)
    if brief and brief.get("pillar") != ADVANCEMENT:
        if heading and ADVANCEMENT_PATTERN.search(heading.group(1)):
            issues.append("The article title pivots away from its assigned broad career pillar into advancement/pay.")
        # An editorial mention of a promotion case can be educational; brand/app pitches cannot.
        if ("Promotion Case" in (markdown or "") or PRODUCT_LINK.search(markdown or "")
                or (re.search(r"\bcareer\s+compass\b", markdown or "", re.IGNORECASE)
                    and PRODUCT_PATTERN.search(markdown or ""))):
            issues.append("Remove the Promotion Case product pitch from this broad career article.")
    product_blocks = [b for b in re.split(r"\n\s*\n", markdown or "")
                      if not b.lstrip().startswith("#") and ("Promotion Case" in b or PRODUCT_LINK.search(b)
                      or ("career compass" in b.lower() and PRODUCT_PATTERN.search(b)))]
    if len(product_blocks) > 1 or any(len(b.split()) > 60 for b in product_blocks):
        issues.append("Reduce the optional product connection to at most one short paragraph (60 words maximum).")
    return issues
