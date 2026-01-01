from __future__ import annotations

from pathlib import Path

from .agents.base import AgentContext
from .config import get_settings
from .rag import EXAMPLES_DIR, SUPPORTED_SUFFIXES, _read_text
from .supervisor import Supervisor


def load_existing_content() -> str:
    """Load textual examples to give the research agent brand context."""

    text_dir = EXAMPLES_DIR / "text"
    snippets = []
    for path in sorted(text_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            content = _read_text(path)
            if content:
                snippets.append(content)
    return "\n\n".join(snippets)


def main() -> None:  # pragma: no cover - CLI entry point
    settings = get_settings()
    supervisor = Supervisor(settings=settings)
    context = AgentContext(
        brand_voice="Supportive, practical, and optimistic career guidance with tactical steps.",
        brand_colors="Navy (#0B1D51), Coral (#FF6B6B), Sand (#F4F1DE)",
        target_audience="Young professionals pivoting careers and early-career job seekers",
        distribution_channels="Webflow blog, LinkedIn, and email newsletter",
    )
    existing_content = load_existing_content()
    result = supervisor.run_workflow(existing_content, context)
    print("Research recommendation:\n", result.research)
    print("\nGenerated blog post:\n", result.blog_post)
    print("\nImage plan:\n", result.image_plan)
    print("\nPublish result:\n", result.publish_result)


if __name__ == "__main__":
    main()
