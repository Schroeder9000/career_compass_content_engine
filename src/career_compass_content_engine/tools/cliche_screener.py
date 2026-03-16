"""Cliché and AI-tell pre-screener for draft content.

Scans Markdown text for banned phrases, overused filler words, and
AI-sounding language before the LLM evaluation cycle. Flagging these
upfront gives the editing agent targeted, specific feedback and can
short-circuit unnecessary revision loops.

Usage
-----
    from career_compass_content_engine.tools.cliche_screener import ClicheScreener

    screener = ClicheScreener()
    report = screener.screen(draft_markdown)
    print(report.summary())
    if report.flag_count > 0:
        print(report.feedback_for_editor())
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Banned phrase registry
# Each entry: (pattern_string, category, replacement_hint)
# Patterns are matched case-insensitively; word-boundary anchors keep them
# from matching substrings (e.g. "navigating" shouldn't block "navigation").
# ---------------------------------------------------------------------------
_BANNED: List[Tuple[str, str, str]] = [
    # Overused scene-setting opening hooks
    (r"\bpicture this\b",              "banned_hook",  "Open with a sharp observation, a direct claim, or a first-person moment instead"),
    (r"^it'?s (monday|tuesday|wednesday|thursday|friday|saturday|sunday) (morning|night|evening|afternoon)",
                                       "banned_hook",  "Time-stamped scene openers are overused — start with an idea, not a clock"),
    (r"^imagine\b",                    "banned_hook",  "Avoid opening a paragraph with 'Imagine...' — it's a formulaic scene-setter; make the point directly instead"),
    # AI filler openers
    (r"\blet'?s dive in\b",          "ai_opener",    "Start with the hook — skip the warm-up"),
    (r"\blet'?s explore\b",           "ai_opener",    "Just start making the point"),
    (r"\bin this article\b",          "ai_opener",    "The reader knows what article they're in"),
    (r"\bin this post\b",             "ai_opener",    "The reader knows what post they're in"),
    (r"\bwithout further ado\b",      "ai_opener",    "Cut it entirely"),
    (r"\btoday we'?re going to\b",    "ai_opener",    "Drop the preamble and make the point"),
    # Overused buzzwords
    (r"\bgame[- ]changer\b",          "buzzword",     "Name what actually changes and why"),
    (r"\bleverage\b",                 "buzzword",     "Use 'use' or be specific"),
    (r"\blevel up\b",                 "buzzword",     "Describe the concrete improvement"),
    (r"\bsynergy\b",                  "buzzword",     "Say what actually combines or aligns"),
    (r"\bunpack\b",                   "buzzword",     "Use 'examine', 'break down', or just do it"),
    (r"\bdeep[- ]dive\b",             "buzzword",     "Just go deep — don't announce it"),
    (r"\bpivot\b",                    "buzzword",     "Use 'shift', 'change direction', or be specific"),
    (r"\bseamless(ly)?\b",            "buzzword",     "Describe the specific friction that's removed"),
    (r"\brobust\b",                   "buzzword",     "Say what specifically makes it strong"),
    (r"\bscalable\b",                 "buzzword",     "Describe the actual growth dynamic"),
    (r"\bholistic(ally)?\b",          "buzzword",     "Name the specific dimensions you mean"),
    (r"\bthought leader(ship)?\b",    "buzzword",     "Show the insight instead of labeling it"),
    (r"\bproactive(ly)?\b",           "buzzword",     "Describe the specific early action"),
    # Cliché phrases
    (r"\bat the end of the day\b",    "cliche",       "State the actual conclusion directly"),
    (r"\bin today'?s fast[- ]paced\b","cliche",       "Drop the time-qualifier and make the point"),
    (r"\bnavigate\b",                 "cliche",       "Be specific: manage, handle, deal with, get through"),
    (r"\bnavigating\b",               "cliche",       "Be specific: managing, handling, working through"),
    (r"\bit'?s no secret\b",          "cliche",       "Drop it — if it's not a secret, just say it"),
    (r"\bin a world where\b",         "cliche",       "Cut the throat-clearing and start with the reality"),
    (r"\bwhen it comes to\b",         "cliche",       "Restructure: 'For X,' or 'On X,' or just lead with the noun"),
    (r"\btake it to the next level\b","cliche",       "Describe the specific next level"),
    (r"\bfake it (till|until|'?til) you make it\b", "cliche", "Engage directly with why this fails or works"),
    (r"\boutside (the|your) comfort zone\b",         "cliche", "Describe the specific uncomfortable thing"),
    (r"\bwork smarter, not harder\b", "cliche",       "Give the actual tactic"),
    (r"\bthe elephant in the room\b", "cliche",       "Name the thing directly"),
    # Vague motivational padding
    (r"\bit can feel overwhelming\b", "vague_emotion","Name the specific sensation: 'the knot in your chest', 'the low-grade dread'"),
    (r"\byou might feel (nervous|anxious|unsure|uncertain)\b", "vague_emotion",
     "Name the physical or situational feeling specifically"),
    (r"\bdon'?t be afraid to\b",      "vague_emotion","Be direct: tell them what to do"),
    (r"\bnever underestimate\b",       "vague_emotion","State the point directly"),
    (r"\bremember (that )?you'?re not alone\b", "vague_emotion", "Show them they're not alone with a specific scenario"),
    # Weak qualifiers
    (r"\bbasically\b",                "weak_qualifier","Cut it — or say what you mean specifically"),
    (r"\bsort of\b",                  "weak_qualifier","Commit to the claim"),
    (r"\bkind of\b",                  "weak_qualifier","Commit to the claim"),
    (r"\bvery unique\b",              "weak_qualifier","'Unique' is already superlative"),
    (r"\bvery (important|crucial|critical)\b", "weak_qualifier", "If it's that important, just say it; cut 'very'"),
]

# Build compiled patterns once at import time
_COMPILED: List[Tuple[re.Pattern, str, str, str]] = [
    (re.compile(pattern, re.IGNORECASE), pattern, category, hint)
    for pattern, category, hint in _BANNED
]


@dataclass
class ScreenerFlag:
    """A single hit from the screener."""
    phrase: str          # The matched text
    pattern: str         # The regex pattern that matched
    category: str        # e.g. "ai_opener", "buzzword"
    hint: str            # Replacement hint for the editor
    line_number: int     # 1-based line number
    line_text: str       # Full line content (truncated to 120 chars)


@dataclass
class ScreenerReport:
    """Aggregated result from ClicheScreener.screen()."""
    flags: List[ScreenerFlag] = field(default_factory=list)
    word_count: int = 0

    @property
    def flag_count(self) -> int:
        return len(self.flags)

    @property
    def flags_by_category(self) -> Dict[str, List[ScreenerFlag]]:
        out: Dict[str, List[ScreenerFlag]] = {}
        for f in self.flags:
            out.setdefault(f.category, []).append(f)
        return out

    def summary(self) -> str:
        if not self.flags:
            return f"✅ No clichés or AI-tells detected ({self.word_count} words)."
        cats = self.flags_by_category
        lines = [
            f"⚠️  Cliché screener: {self.flag_count} flag(s) across "
            f"{len(cats)} category(ies) in {self.word_count}-word draft.",
        ]
        for cat, items in cats.items():
            lines.append(f"  [{cat}] {len(items)} hit(s)")
        return "\n".join(lines)

    def feedback_for_editor(self) -> str:
        """Return a structured block ready to inject into the editing prompt."""
        if not self.flags:
            return ""

        lines = [
            "=== Cliché & AI-Tell Pre-Screen Report ===",
            f"Total flags: {self.flag_count}",
            "",
        ]
        for cat, items in self.flags_by_category.items():
            lines.append(f"[{cat.upper().replace('_', ' ')}]")
            for flag in items:
                snippet = flag.line_text.strip()[:100]
                lines.append(f"  Line {flag.line_number}: '{flag.phrase}'")
                lines.append(f"    Context: …{snippet}…")
                lines.append(f"    Fix: {flag.hint}")
            lines.append("")

        lines.append(
            "ACTION REQUIRED: Find and replace every flagged phrase above with "
            "specific, vivid, opinionated language. Do not just delete them — "
            "replace with substance."
        )
        return "\n".join(lines)


class ClicheScreener:
    """Scans Markdown draft content for clichés and AI-sounding language."""

    def screen(self, text: str) -> ScreenerReport:
        """Scan *text* and return a ScreenerReport with all flags."""
        lines = text.splitlines()
        word_count = len(text.split())
        flags: List[ScreenerFlag] = []

        for line_no, line in enumerate(lines, start=1):
            # Skip headings — the ResearchAgent intentionally bans certain heading words
            # but we don't want to double-flag structural elements here.
            stripped = line.strip()
            is_heading = stripped.startswith("#")

            for compiled, pattern, category, hint in _COMPILED:
                # For the "navigating" / "navigate" patterns, still flag even in headings
                # because those are explicitly forbidden in titles too.
                if is_heading and category not in ("cliche",):
                    continue
                for match in compiled.finditer(line):
                    flags.append(
                        ScreenerFlag(
                            phrase=match.group(0),
                            pattern=pattern,
                            category=category,
                            hint=hint,
                            line_number=line_no,
                            line_text=line[:120],
                        )
                    )

        return ScreenerReport(flags=flags, word_count=word_count)
