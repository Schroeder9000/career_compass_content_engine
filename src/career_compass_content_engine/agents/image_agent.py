from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from langchain_core.language_models.chat_models import BaseChatModel

from .base import AgentContext, AgentResult, BaseAgent
import json



IMAGE_TEMPLATE = """
You are a master illustrator and concept artist with 20+ years of experience
designing hero illustrations for premium mobile apps and editorial platforms.

You specialize in:
- Conceptual, metaphor-driven illustrations (not literal scenes)
- Clean, detailed, textured, modern composition that reads well on mobile
- Career, growth, and self-improvement themes for young professionals (ages 22–35)

Your task:
Create a single blog hero illustration for the Career Compass content library.

The image must:
- Communicate the core idea of the article clearly at a glance
- Feel aspirational, thoughtful, and artistic (not playful or cartoonish)
- Be abstract or metaphorical rather than literal
- Be instantly recognizable as "Career Compass" branded content

---

### ARTICLE CONTEXT
Title / Topic:
{topic}

Short summary of the article:
{summary}

Key idea to visually communicate:
{key_idea}

---

### BRAND & STYLE CONSTRAINTS
Primary visual style:
Concept art / modern editorial illustration

Art direction:
- CRITICAL: Design for a WIDE 16:9 landscape format — the illustration must be HORIZONTAL and fill the entire width of the frame
- DO NOT create a square or vertical composition centered in empty space
- DO NOT leave blank margins, borders, or padding around the illustration
- The artwork must extend completely to the left edge, right edge, top edge, and bottom edge of the 16:9 canvas
- Use the full WIDTH of the landscape format — spread design elements horizontally across the frame
- Abstract human figures or symbolic elements (not detailed faces or stick people)
- Visual metaphor preferred over literal workplace scenes
- Subtle sense of motion or progress
- Incorporate Art Deco influences (geometric shapes, stylized forms)
- Artistic, and creative (not corporate stock photo style). Take interesting viewpoints.
- Include detail and texture; avoid flat colors or overly simplistic styles.

Color Palette
- Primary — Compass Orange : A warm, burnished orange that signals momentum and progress. It feels earned, not flashy—more sunrise than spotlight—representing steady growth, direction, and forward motion.
- Secondary — Charcoal Black & Warm Stone Gray : Deep charcoal anchors the palette with seriousness and contrast, while layered warm grays add depth, texture, and restraint. Together, they create a grounded, architectural foundation that lets the orange lead without noise.
- Overall Treatment :High-contrast and intentional. Earth-toned, human, and timeless. No bright blues, purples, or neon hues—nothing that distracts, cheapens, or dates the experience. Every color serves clarity, focus, and long-term credibility.

Texture & finish:
- Linocut-style
- Avoid photorealism entirely

---

### AUDIENCE
Target audience:
Early-career professionals and young knowledge workers

Tone:
Confident, calm, focused, growth-oriented

---

### COMPOSITION REQUIREMENTS
Aspect ratio:
16:9 (blog hero image)

Framing:
- MANDATORY: Full-bleed composition that fills the entire 16:9 LANDSCAPE frame from edge to edge
- CRITICAL: The composition must be WIDE and HORIZONTAL, not square
- DO NOT create a square illustration in the center with empty space on the sides
- Design elements must extend completely to the left and right edges of the frame
- The artwork should leverage the full HORIZONTAL width of the 16:9 format
- No padding, margins, borders, or empty space around the illustration whatsoever

---

### EXAMPLES / VISUAL INSPIRATION
(Use only as inspiration, do NOT copy literally)
{examples}

---

### OUTPUT FORMAT
Return a **strict JSON object** with the following keys:

- prompt: A detailed, high-quality image-generation prompt
- alt_text: Accessible, descriptive alt text (≤ 140 characters)
- suggested_palette: List of 3–5 hex color codes aligned with the brand
- aspect_ratio: "16:9"
- guidance_on_model_settings: Brief notes (e.g. abstraction level, style strength)

Do not include any extra commentary outside the JSON.
"""


@dataclass
class ImageAgent(BaseAgent):
    """Agent that designs an image prompt consistent with the brand."""

    llm: BaseChatModel

    def design(self, topic_payload: Dict[str, str], rag_context: str, context: AgentContext) -> AgentResult:
        prompt = IMAGE_TEMPLATE.format(
            topic=topic_payload.get("topic", ""),
            summary=topic_payload.get("summary", topic_payload.get("rationale", "")),
            key_idea=topic_payload.get("key_idea", topic_payload.get("idea", "")),
            examples=rag_context or "(No visual examples provided; follow brand colors.)",
        )
        # Run the LLM and expect JSON output. Parse and validate keys.
        # Use llm.invoke directly to access raw model output objects when available.
        try:
            response = self.llm.invoke(prompt)
            # keep a reference to the raw result for consistent debugging/returns
            result = response
        except Exception:
            # Fallback to BaseAgent.run wrapper
            result = self.run(prompt)
            response = result.content.get("message")

        # extract raw text from a variety of response shapes
        raw = None
        if isinstance(response, str):
            raw = response
        else:
            # try common attributes
            for attr in ("content", "text", "message", "data"):
                val = getattr(response, attr, None) if not isinstance(response, dict) else response.get(attr)
                if val:
                    raw = val
                    break
        if raw is None:
            # final fallback to stringified response
            raw = str(response)
        parsed = None
        try:
            if isinstance(raw, dict):
                parsed = raw
            else:
                parsed = json.loads(raw)
        except Exception as e:
            # If parsing fails, attempt to extract a JSON block from the text
            try:
                import re

                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    parsed = json.loads(m.group(0))
            except Exception:
                parsed = None
            if parsed is None:
                return AgentResult(content={"error": "invalid_json", "raw": raw, "parse_error": str(e)}, raw_response=str(raw))

        # Minimal validation
        if not parsed.get("prompt") or not parsed.get("alt_text"):
            return AgentResult(content={"error": "missing_fields", "parsed": parsed}, raw_response=json.dumps(parsed))

        return AgentResult(content=parsed, raw_response=json.dumps(parsed))
