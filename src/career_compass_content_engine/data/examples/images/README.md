This directory stores visual references and text sidecars for image RAG retrieval.

What gets embedded by the current pipeline:
- `.md`, `.txt`, and `.json` files only.

What does not get embedded directly:
- Binary image files (`.png`, `.webp`, `.jpg`).

Recommended pattern:
- Keep the image binary as a visual source of truth.
- Add a same-name JSON sidecar that captures prompt notes, viewpoint, composition,
  complexity, lighting, palette, and anti-patterns.
- Keep one collection-level guide file for global brand and diversity policy.

Current collection goals:
- Aspirational and growth-oriented visuals.
- Brighter, optimistic tone (avoid repeated dark outputs).
- Wide variation in perspective, layout archetype, orientation treatment,
  viewpoint, and complexity.
- Preserve 16:9 full-bleed hero format and Career Compass palette anchors.
