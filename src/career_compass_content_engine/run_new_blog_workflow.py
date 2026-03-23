from __future__ import annotations

import asyncio

from .agents.base import AgentContext
from .agents.publishing_agent import WebflowCMS
from .config import get_settings

from .rag import EXAMPLES_DIR, SUPPORTED_SUFFIXES, _read_text

from .supervisor import Supervisor


def _fetch_webflow_content(settings) -> str:
    """Fetch published content from Webflow CMS."""
    if not settings.webflow_api_token or not settings.webflow_collection_id:
        return "[Webflow content unavailable: missing API token or collection ID]"
    
    try:
        cms = WebflowCMS(settings.webflow_api_token)
        result = cms.list_items(settings.webflow_collection_id, limit=100)
        
        items = result.get("items", [])
        if not items:
            return "[No published items found in Webflow collection]"
        
        # Format items for the research agent
        formatted_items = []
        for item in items:
            field_data = item.get("fieldData", {})
            
            # Extract key fields
            title = field_data.get("name", field_data.get("title", "Untitled"))
            slug = field_data.get("slug", "")
            summary = field_data.get("description", field_data.get("summary", ""))
            published_date = item.get("publishedOn", item.get("createdOn", ""))
            
            # Build structured representation
            item_text = f"Title: {title}\n"
            if slug:
                item_text += f"Slug: {slug}\n"
            if summary:
                item_text += f"Summary: {summary}\n"
            if published_date:
                item_text += f"Published: {published_date}\n"
            
            # Include article body preview if available
            article_body = field_data.get("post-body", field_data.get("article", field_data.get("content", "")))
            if article_body:
                # Strip HTML tags for cleaner preview
                import re
                clean_body = re.sub(r"<[^>]+>", "", article_body)
                preview = clean_body[:500].strip()
                if len(clean_body) > 500:
                    preview += "..."
                item_text += f"Preview: {preview}\n"
            
            formatted_items.append(item_text)
        
        return "\n\n---\n\n".join(formatted_items)
    
    except Exception as exc:
        return f"[Error fetching Webflow content: {exc}]"


def load_existing_content() -> str:
    """Load published content from Webflow and local examples to give the research agent full context."""
    settings = get_settings()
    
    # Fetch published Webflow content
    webflow_content = _fetch_webflow_content(settings)
    
    # Also load local text examples for brand voice reference
    text_dir = EXAMPLES_DIR / "text"
    local_snippets = []
    for path in sorted(text_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            content = _read_text(path)
            if content:
                local_snippets.append(content)
    
    local_content = "\n\n".join(local_snippets) if local_snippets else ""
    
    # Combine Webflow content and local examples
    sections = []
    if webflow_content:
        sections.append(f"=== PUBLISHED WEBFLOW CONTENT ===\n\n{webflow_content}")
    if local_content:
        sections.append(f"=== LOCAL BRAND EXAMPLES ===\n\n{local_content}")
    
    return "\n\n\n".join(sections) if sections else "No existing content available."


def main() -> None:  # pragma: no cover - CLI entry point
    settings = get_settings()
    supervisor = Supervisor(settings=settings)
    context = AgentContext(
        brand_voice="Supportive, practical, and optimistic career guidance with tactical steps.",
        brand_colors="Orange (#f45601), Dark Grey (#252422), Red (#dd2902), White (#fcfcfc), Light Grey (#3f3d39), Black (#110c10), Yellow (#ffc100)",
        target_audience="Young professionals seeking to grow their careers and navigate workplace challenges.",
        distribution_channels="Webflow blog, LinkedIn, and email newsletter",
    )
    existing_content = load_existing_content()
    result = asyncio.run(supervisor.run_workflow(existing_content, context))
    print("Research recommendation:\n", result.research)
    print("\nGenerated blog post:\n", result.blog_post)
    print("\nImage plan:\n", result.image_plan)
    print("\nPublish result:\n", result.publish_result)
    print(
        f"\nToken usage — total: {result.total_tokens} "
        f"(prompt: {result.prompt_tokens} / completion: {result.completion_tokens})  "
        f"est. cost: ${result.total_cost_usd:.4f}"
    )


if __name__ == "__main__":
    main()
