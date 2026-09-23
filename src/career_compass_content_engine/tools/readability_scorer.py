"""Readability scorer for draft content.

Computes readability metrics (Flesch-Kincaid, sentence/paragraph length
variance) without requiring any external API. Returns a structured report
and a prompt-ready feedback string for the editing agent.

The scorer uses the `textstat` library when available and falls back to
a pure-Python implementation for core metrics so it works without any
install.

Usage
-----
    from career_compass_content_engine.tools.readability_scorer import ReadabilityScorer

    scorer = ReadabilityScorer()
    report = scorer.score(draft_markdown)
    print(report.summary())
    print(report.feedback_for_editor())
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Targets (tuned for long-form career coaching content)
# ---------------------------------------------------------------------------
_FK_GRADE_TARGET = (8.0, 12.0)   # Flesch-Kincaid grade level sweet spot
_FLESCH_READING_TARGET = (50, 70) # Flesch Reading Ease (higher = easier)
_AVG_SENTENCE_WORDS_TARGET = (15, 22)
_MIN_PARAGRAPH_VARIANCE = 10.0   # std-dev of paragraph word counts; below = formulaic


def _count_syllables(word: str) -> int:
    """Pure-Python syllable counter (approximation)."""
    word = word.lower().strip(".,;:!?\"'()-")
    if not word:
        return 0
    # Remove silent trailing 'e'
    if word.endswith("e") and len(word) > 2:
        word = word[:-1]
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    return max(1, count)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences on terminal punctuation."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def _split_paragraphs(text: str) -> List[str]:
    """Split Markdown text into non-empty paragraphs."""
    paragraphs = re.split(r"\n{2,}", text.strip())
    cleaned = []
    for p in paragraphs:
        p = p.strip()
        # Skip headings, code fences, blank-ish lines
        if not p or p.startswith("#") or p.startswith("```"):
            continue
        cleaned.append(p)
    return cleaned


def _word_count(text: str) -> int:
    return len(text.split())


def _flesch_kincaid(total_words: int, total_sentences: int, total_syllables: int) -> Tuple[float, float]:
    """Return (Flesch Reading Ease, Flesch-Kincaid Grade Level)."""
    if total_sentences == 0 or total_words == 0:
        return 0.0, 0.0
    words_per_sentence = total_words / total_sentences
    sylls_per_word = total_syllables / total_words
    reading_ease = 206.835 - 1.015 * words_per_sentence - 84.6 * sylls_per_word
    grade_level = 0.39 * words_per_sentence + 11.8 * sylls_per_word - 15.59
    return round(reading_ease, 1), round(grade_level, 1)


def _std_dev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return round(variance ** 0.5, 1)


@dataclass
class ReadabilityReport:
    """Aggregated readability metrics for a piece of content."""

    # Core metrics
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    avg_sentence_words: float = 0.0
    paragraph_word_variance: float = 0.0   # std-dev of paragraph word counts

    # Flesch metrics (pure-Python fallback)
    flesch_reading_ease: float = 0.0
    fk_grade_level: float = 0.0

    # textstat library metrics (when available)
    textstat_fk_grade: Optional[float] = None
    textstat_reading_ease: Optional[float] = None
    textstat_smog: Optional[float] = None

    # Flags
    issues: List[str] = field(default_factory=list)

    def summary(self) -> str:
        fk = self.textstat_fk_grade if self.textstat_fk_grade is not None else self.fk_grade_level
        ease = self.textstat_reading_ease if self.textstat_reading_ease is not None else self.flesch_reading_ease
        lines = [
            "📖 Readability Report",
            f"   Words: {self.word_count}  |  Sentences: {self.sentence_count}  |  Paragraphs: {self.paragraph_count}",
            f"   Avg sentence length: {self.avg_sentence_words:.1f} words  (target: {_AVG_SENTENCE_WORDS_TARGET[0]}–{_AVG_SENTENCE_WORDS_TARGET[1]})",
            f"   Flesch Reading Ease: {ease:.1f}  (target: {_FLESCH_READING_TARGET[0]}–{_FLESCH_READING_TARGET[1]})",
            f"   FK Grade Level: {fk:.1f}  (target: {_FK_GRADE_TARGET[0]}–{_FK_GRADE_TARGET[1]})",
            f"   Paragraph length variance (σ): {self.paragraph_word_variance:.1f}  (target: ≥ {_MIN_PARAGRAPH_VARIANCE})",
        ]
        if self.issues:
            lines.append(f"   ⚠️  {len(self.issues)} issue(s) detected")
        else:
            lines.append("   ✅ All readability targets met")
        return "\n".join(lines)

    def feedback_for_editor(self) -> str:
        """Return a structured block ready to inject into the editing prompt."""
        if not self.issues:
            return ""

        fk = self.textstat_fk_grade if self.textstat_fk_grade is not None else self.fk_grade_level
        ease = self.textstat_reading_ease if self.textstat_reading_ease is not None else self.flesch_reading_ease

        lines = [
            "=== Readability Pre-Screen Report ===",
            f"Flesch Reading Ease: {ease:.1f} | FK Grade Level: {fk:.1f} | "
            f"Avg sentence: {self.avg_sentence_words:.1f} words | "
            f"Paragraph variance σ: {self.paragraph_word_variance:.1f}",
            "",
            "ISSUES FLAGGED:",
        ]
        for issue in self.issues:
            lines.append(f"  ✗ {issue}")

        lines += [
            "",
            "ACTION REQUIRED:",
            "  - Break long sentences into 2-3 shorter ones.",
            "  - Vary paragraph length deliberately: some 1-sentence punches, "
            "some 4-5 sentence deep dives.",
            "  - Aim for a mix: short declarative sentence. Then a longer one that "
            "builds a richer picture of the scenario before landing on the point.",
        ]
        return "\n".join(lines)


class ReadabilityScorer:
    """Scores a Markdown draft on readability and rhythm metrics."""

    def score(self, text: str) -> ReadabilityReport:
        """Analyse *text* and return a ReadabilityReport."""
        # Strip Markdown syntax for metric calculation
        plain = self._strip_markdown(text)

        words = plain.split()
        word_count = len(words)
        sentences = _split_sentences(plain)
        sentence_count = len(sentences)
        paragraphs = _split_paragraphs(text)  # use original for structure
        paragraph_count = len(paragraphs)

        avg_sentence_words = (word_count / sentence_count) if sentence_count else 0.0

        # Syllable counts for Flesch
        syllables = sum(_count_syllables(w) for w in words)
        flesch_ease, fk_grade = _flesch_kincaid(word_count, sentence_count, syllables)

        # Paragraph length variance
        para_lengths = [len(p.split()) for p in paragraphs]
        para_variance = _std_dev([float(x) for x in para_lengths])

        # Try textstat for better accuracy
        ts_fk: Optional[float] = None
        ts_ease: Optional[float] = None
        ts_smog: Optional[float] = None
        try:
            import textstat  # type: ignore
            ts_ease = round(float(textstat.flesch_reading_ease(plain)), 1)
            ts_fk = round(float(textstat.flesch_kincaid_grade(plain)), 1)
            ts_smog = round(float(textstat.smog_index(plain)), 1)
        except (ImportError, LookupError, OSError):
            # Optional dictionaries may be unavailable in a fresh/offline container.
            # Keep the local calculations instead of failing the publication run.
            ts_ease = ts_fk = ts_smog = None

        # Determine active reading score values
        active_ease = ts_ease if ts_ease is not None else flesch_ease
        active_fk = ts_fk if ts_fk is not None else fk_grade

        # Build issues list
        issues: List[str] = []

        if avg_sentence_words > _AVG_SENTENCE_WORDS_TARGET[1]:
            issues.append(
                f"Average sentence is {avg_sentence_words:.1f} words (target ≤ {_AVG_SENTENCE_WORDS_TARGET[1]}). "
                "Too many long sentences — vary with some short, punchy ones."
            )
        elif avg_sentence_words < _AVG_SENTENCE_WORDS_TARGET[0]:
            issues.append(
                f"Average sentence is {avg_sentence_words:.1f} words (target ≥ {_AVG_SENTENCE_WORDS_TARGET[0]}). "
                "Sentences are too choppy — add more connective depth in complex sections."
            )

        if active_fk > _FK_GRADE_TARGET[1]:
            issues.append(
                f"FK Grade Level {active_fk:.1f} is above {_FK_GRADE_TARGET[1]} — "
                "prose may be too dense for a career coaching audience. "
                "Simplify sentence structure and word choice in the most complex sections."
            )

        if active_ease < _FLESCH_READING_TARGET[0]:
            issues.append(
                f"Flesch Reading Ease {active_ease:.1f} is below {_FLESCH_READING_TARGET[0]} — "
                "draft reads as dense or academic. Break up complex sentences."
            )

        if para_variance < _MIN_PARAGRAPH_VARIANCE and paragraph_count >= 5:
            issues.append(
                f"Paragraph length variance (σ={para_variance:.1f}) is too low — "
                "paragraphs are all similar length, making the piece feel formulaic. "
                "Deliberately mix 1-sentence punches with longer analytical paragraphs."
            )

        return ReadabilityReport(
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            avg_sentence_words=round(avg_sentence_words, 1),
            paragraph_word_variance=para_variance,
            flesch_reading_ease=flesch_ease,
            fk_grade_level=fk_grade,
            textstat_fk_grade=ts_fk,
            textstat_reading_ease=ts_ease,
            textstat_smog=ts_smog,
            issues=issues,
        )

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove Markdown syntax, leaving only prose text."""
        # Remove fenced code blocks
        text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
        # Remove inline code
        text = re.sub(r"`[^`]+`", " ", text)
        # Remove images
        text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)
        # Remove links but keep link text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Remove headings markers
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Remove bold/italic
        text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
        text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
        # Remove horizontal rules
        text = re.sub(r"^[-*_]{3,}\s*$", " ", text, flags=re.MULTILINE)
        # Remove bullet/numbered list markers
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
