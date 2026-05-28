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
- Conceptual, metaphor-driven illustrations for professional growth
- Clean, detailed, textured compositions that read clearly on mobile
- Career and self-improvement themes for young professionals (ages 22-35)

Your task:
Create a single blog hero illustration for the Career Compass content library.

The image must:
- Communicate the article's core idea clearly at a glance
- Feel aspirational, optimistic, and growth-oriented
- Be metaphorical rather than a literal workplace scene
- Be recognizable as Career Compass branded content

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
Primary style direction:
Modern editorial concept illustration with texture and intentional craft.

Allowed aesthetic range (rotate to avoid repetition):
- Linocut-inspired editorial (brand anchor)
- Geometric symbolic collage
- Painterly vector-texture hybrid
- Stylized architectural concept art

Hard constraints:
- CRITICAL: The final composition must be full-bleed 16:9 landscape.
- No borders, padding, or letterboxing.
- Keep abstract human or symbolic forms; avoid detailed faces.
- Avoid photorealism and stock-photo feel.

Color system:
- Primary anchor: Compass Orange (#f45601)
- Structural anchors: Charcoal Black and Warm Stone Gray
- Use brighter, optimistic lighting with warm highlights and open midtones.
- Keep contrast intentional, but avoid heavy shadow dominance or gloomy low-key scenes.
- No neon hues, no electric purples, no distracting saturated blues.

Tone direction:
- Confident, calm, directional, hopeful.
- Visual mood should suggest momentum, agency, and possibility.

---

### DIVERSITY CONTROLS (MANDATORY)
For each run, intentionally choose a different combination than recent outputs.

Vary perspective:
- Bird's-eye, eye-level, low-angle, sectional/isometric, panoramic horizon view.

Vary composition archetype:
- Diagonal ascent path, branching crossroads, distributed constellation, layered skyline, spiral progression.

Vary viewpoint and focal strategy:
- Single dominant motif, multi-node narrative, foreground-to-background journey, asymmetrical balance.

Vary complexity level:
- Minimal symbolic, medium narrative, high-detail layered world.

Vary orientation treatment inside 16:9:
- Left-to-right progression, right-to-left tension-to-release, center-offset momentum, tri-zone rhythm, radial spread.

Anti-repetition rules:
- Do not reuse the same central metaphor in consecutive runs.
- Do not default to dark backgrounds.
- Do not default to centered compositions.

---

### COMPOSITION REQUIREMENTS
Aspect ratio:
16:9 (blog hero image)

Framing:
- Do not force the same grid each run.
- You may use zone-based, radial, diagonal, layered-depth, or asymmetric structures.
- Keep the scene edge-to-edge and readable at mobile thumbnail size.

---

### EXAMPLES / VISUAL INSPIRATION
(Use as directional references only. Do not copy literally.)
{examples}

### RECENT IMAGE HISTORY (DO NOT REPEAT)
{recent_history}

If recent history is provided, you must choose a clearly different composition/profile combination.

### REQUIRED DIVERSITY PROFILE FOR THIS RUN
{diversity_target}

Treat this profile as mandatory unless it conflicts with 16:9 full-bleed constraints.

---

### OUTPUT FORMAT
Return a strict JSON object with these keys:

- prompt: A detailed, high-quality image-generation prompt. It must:
    * Start by stating this is a full-bleed 16:9 hero illustration.
    * Explicitly encode the required diversity profile in concrete visual language.
    * Include bright, optimistic lighting guidance.
    * End with: "Full-bleed composition filling the entire 16:9 canvas."
- alt_text: Accessible, descriptive alt text (<= 140 characters)
- suggested_palette: List of 3-5 hex color codes aligned with brand anchors
- aspect_ratio: "16:9"
- guidance_on_model_settings: Brief notes (for example: abstraction level, style strength)
- diversity_tags: Object with keys viewpoint, composition_archetype, complexity_level, lighting_style, shot_distance, camera_angle, focal_strategy

Do not include any text outside the JSON object.
"""


@dataclass
class ImageAgent(BaseAgent):
    """Agent that designs an image prompt consistent with the brand."""

    llm: BaseChatModel

    def design(
        self,
        topic_payload: Dict[str, str],
        rag_context: str,
        context: AgentContext,
        recent_history_context: str = "",
        diversity_target_context: str = "",
    ) -> AgentResult:
        prompt = IMAGE_TEMPLATE.format(
            topic=topic_payload.get("topic", ""),
            summary=topic_payload.get("summary", topic_payload.get("rationale", "")),
            key_idea=topic_payload.get("key_idea", topic_payload.get("idea", "")),
            examples=rag_context or "(No visual examples provided; follow brand colors.)",
            recent_history=recent_history_context or "(No prior image history available.)",
            diversity_target=diversity_target_context or "(No enforced diversity target provided.)",
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
