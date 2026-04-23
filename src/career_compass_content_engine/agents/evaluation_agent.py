from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List

from langchain_core.language_models.chat_models import BaseChatModel

from .base import BaseAgent


EVALUATION_TEMPLATE = """
You are the Chief Content Officer for Career Compass, responsible for final quality control before publishing.

Your job: Evaluate this edited blog post against our strict quality standards and provide actionable feedback.

SCORING CRITERIA (0-10 scale for each):

1. **Word Count & Depth** (0-10)
   - 800–1,200 words is the target range. Penalize articles that pad to fill space just as much as ones that feel incomplete.
   - 10 = Every section earns its place. The article is substantive without being bloated.
   - Deduct points if sections feel like filler, or if the article ends abruptly without a real conclusion.
   - Do NOT deduct points purely for being under 1,200 words if the content is genuinely complete.
   - Deduct points if content feels surface-level or lacks depth of analysis, nuance, or insight

2. **Specific Examples & Details** (0-10)
   - 10 = The article is vivid and concrete throughout. Where a named-character story is the most
     natural way to illustrate a point, it is fully developed (name, job title and company type,
     exact situation, dialogue, concrete outcome). Where a story isn't needed, vivid specificity
     carries the weight: real numbers, exact role names, concrete day/situation framing, word-for-word
     scripts. Not every article requires a character vignette — the standard is specificity, not format.
   - 8 = The article is consistently specific. Most sections use concrete details (named roles, exact
     phrases, real scenarios) rather than abstract generalities. A character vignette may or may not
     be present; what matters is that advice feels grounded, not theoretical.
   - 6 = A mix: some sections are vivid and concrete, but others fall back on abstract, generic advice
     ("you might find yourself in a meeting...", "many professionals struggle with...") with nothing
     to anchor the point in reality.
   - 3 = Mostly abstract advice throughout. Attempts at examples exist but none are grounded — no
     names, no specific scenarios, no dialogue, no concrete outcomes.
   - 0 = Purely theoretical. Nothing in the article could not have been written without any real-world
     context or experience.

3. **Brand Voice Quality** (0-10)
   - 10 = Warm, candid, funny, opinionated, contrarian where appropriate
   - 5 = Professional but generic, missing personality
   - 0 = Corporate speak, AI-sounding, clichéd
   - Deduct points for any "AI-sounding" phrases or generic filler ("let's dive in", "game-changer", "at the end of the day")

4. **Emotional Depth** (0-10)
   - 10 = The article clearly articulates specific emotions and provides helpful framing and actionable guidance (e.g. "the dread of negative feedback", "the electric relief when your manager finally sees your work"); validates the reader's experience in a way that feels specific and actionable
   - 8 = Named emotions with genuine resonance, but lacks deeper emotional intelligence or actionable validation
    - 5 = Some attempt at emotional resonance, but feelings are generic
   - 2 = A single passing mention of feelings with no real validation
   - 0 = Purely tactical, no emotional intelligence

5. **Actionability & Frameworks** (0-10)
   - 10 = Clear frameworks, scripts, step-by-step breakdowns, decision tools
   - 5 = Some tactical advice but vague
   - 0 = Only high-level guidance, no concrete actions

6. **Writing Quality** (0-10)
   - 10 = Varied rhythm, no fluff phrases, no AI tells, engaging prose
   - 5 = Competent but pedestrian, paragraphs of similar length and sentence count
   - 0 = Generic, repetitive, "AI-sounding" language

OVERALL QUALITY THRESHOLD: Average score must be ≥ 8.7 to publish

---

CONTENT TO EVALUATE:
{content}

---

YOUR OUTPUT MUST BE VALID JSON with this exact structure:
{{
  "scores": {{
    "word_count_depth": <0-10>,
    "specific_examples": <0-10>,
    "brand_voice": <0-10>,
    "emotional_depth": <0-10>,
    "actionability": <0-10>,
    "writing_quality": <0-10>
  }},
  "average_score": <calculated average>,
  "passes_threshold": <true if avg >= 8.7, false otherwise>,
  "feedback": {{
    "strengths": [
      "Specific strength 1 with line/section reference",
      "Specific strength 2 with line/section reference"
    ],
    "critical_issues": [
      "Issue 1: Specific problem with section name and what needs fixing",
      "Issue 2: Specific problem with section name and what needs fixing"
    ],
    "improvement_suggestions": [
      "Suggestion 1: Concrete action to improve a specific section",
      "Suggestion 2: Concrete action to improve a specific section"
    ]
  }},
  "recommended_action": "PUBLISH" or "REVISE"
}}

INSTRUCTIONS:
1. Count actual words (exclude headings and TL;DR bullets from section counts)
2. Identify specific sections by heading name in your feedback
3. Quote problematic phrases that sound AI-generated
4. Be tough but constructive - this must be publication-worthy
5. If recommending REVISE, prioritize 2-3 most critical issues
6. MANDATORY: For every criterion scored below 8.7, you MUST include at least one entry in
   critical_issues that states: the criterion name, the specific section where the problem occurs,
   the exact issue (quote the weak text if possible), and a concrete fix instruction.
   An empty critical_issues array is only acceptable when ALL criteria score 8.7 or above.

Return ONLY the JSON object, no other text.
"""


@dataclass
class EvaluationResult:
    """Structured evaluation result with scores and feedback."""
    scores: Dict[str, float]
    average_score: float
    passes_threshold: bool
    feedback: Dict[str, List[str]]
    recommended_action: str
    raw_json: Dict


@dataclass
class EvaluationAgent(BaseAgent):
    """Agent that evaluates content quality and provides structured feedback."""

    llm: BaseChatModel
    score_threshold: float = 8.7

    def evaluate(self, content: str) -> EvaluationResult:
        """
        Evaluate content against quality standards.
        
        Args:
            content: The blog post content to evaluate
            
        Returns:
            EvaluationResult with scores, feedback, and recommendation
        """
        prompt = EVALUATION_TEMPLATE.format(content=content)
        result = self.run(prompt)
        
        # Parse the JSON response
        evaluation_data = self._parse_evaluation(result.raw_response)
        
        return EvaluationResult(
            scores=evaluation_data.get("scores", {}),
            average_score=evaluation_data.get("average_score", 0.0),
            passes_threshold=evaluation_data.get("passes_threshold", False),
            feedback=evaluation_data.get("feedback", {}),
            recommended_action=evaluation_data.get("recommended_action", "REVISE"),
            raw_json=evaluation_data,
        )

    def _parse_evaluation(self, raw_response: str) -> Dict:
        """Parse JSON evaluation from LLM response, handling code fences."""
        if not raw_response:
            return self._default_evaluation()
        
        try:
            # Try direct JSON parse first
            return json.loads(raw_response.strip())
        except json.JSONDecodeError:
            # Try extracting JSON from code fences
            cleaned = raw_response.strip()
            
            # Remove markdown code fences
            if cleaned.startswith('```'):
                lines = cleaned.split('\n')
                # Remove opening fence (could be ```json or just ```)
                if lines[0].startswith('```'):
                    lines = lines[1:]
                # Remove closing fence
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                cleaned = '\n'.join(lines)
            
            # Try parsing cleaned version
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                # Try regex extraction as last resort
                match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(0))
                    except json.JSONDecodeError:
                        pass
        
        # Return default if all parsing fails
        return self._default_evaluation()

    def _default_evaluation(self) -> Dict:
        """Return a default evaluation when parsing fails."""
        return {
            "scores": {
                "word_count_depth": 5.0,
                "specific_examples": 5.0,
                "brand_voice": 5.0,
                "emotional_depth": 5.0,
                "actionability": 5.0,
                "writing_quality": 5.0,
            },
            "average_score": 5.0,
            "passes_threshold": False,
            "feedback": {
                "strengths": [],
                "critical_issues": ["Evaluation parsing failed - manual review required"],
                "improvement_suggestions": ["Re-run evaluation or review manually"],
            },
            "recommended_action": "REVISE",
        }

    def format_feedback_for_editor(self, evaluation: EvaluationResult) -> str:
        """
        Format evaluation feedback into a prompt for the editor to use.
        
        Args:
            evaluation: The evaluation result
            
        Returns:
            Formatted feedback string
        """
        feedback_parts = []
        
        feedback_parts.append("EVALUATION FEEDBACK")
        feedback_parts.append("=" * 50)
        feedback_parts.append(f"\nAverage Score: {evaluation.average_score:.1f}/10.0")
        feedback_parts.append(f"Status: {'NEEDS REVISION' if not evaluation.passes_threshold else 'PASSES'}\n")
        
        # Individual scores
        feedback_parts.append("INDIVIDUAL SCORES:")
        for criterion, score in evaluation.scores.items():
            criterion_name = criterion.replace('_', ' ').title()
            feedback_parts.append(f"  • {criterion_name}: {score}/10")
        
        # Strengths
        if evaluation.feedback.get("strengths"):
            feedback_parts.append("\nSTRENGTHS:")
            for strength in evaluation.feedback["strengths"]:
                feedback_parts.append(f"  ✓ {strength}")
        
        # Critical issues
        if evaluation.feedback.get("critical_issues"):
            feedback_parts.append("\nCRITICAL ISSUES TO FIX:")
            for issue in evaluation.feedback["critical_issues"]:
                feedback_parts.append(f"  ✗ {issue}")
        
        # Improvement suggestions
        if evaluation.feedback.get("improvement_suggestions"):
            feedback_parts.append("\nIMPROVEMENT SUGGESTIONS:")
            for suggestion in evaluation.feedback["improvement_suggestions"]:
                feedback_parts.append(f"  → {suggestion}")
        
        return "\n".join(feedback_parts)
