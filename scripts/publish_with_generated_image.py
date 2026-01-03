#!/usr/bin/env python3
"""Example runner: ImageAgent -> ImageGenerator -> PublishingAgent

Generates an image (via OpenAI or placeholder), uploads it to Webflow, and creates a draft Blog Post.
Run with: PYTHONPATH=src .venv/bin/python scripts/publish_with_generated_image.py
"""
from pathlib import Path
import json
from dotenv import load_dotenv
load_dotenv("../.env")

from career_compass_content_engine.config import Settings
from career_compass_content_engine.agents.image_agent import ImageAgent
from career_compass_content_engine.agents.image_generator import ImageGenerator
from career_compass_content_engine.agents.publishing_agent import PublishingAgent
from career_compass_content_engine.agents.base import AgentContext


class DummyLLM:
    def invoke(self, prompt: str):
        # Return a simple JSON response with prompt and alt_text
        class R:
            pass

        obj = {
            "prompt": "A clean, modern hero illustration of a person at a laptop surrounded by career icons, flat colors",
            "alt_text": "Person working at laptop with career icons",
            "suggested_palette": ["#0b5cff", "#ffffff", "#111827"],
            "aspect_ratio": "16:9",
            "guidance": {"style": "flat", "model": "stable"},
        }
        r = R()
        r.content = json.dumps(obj)
        return r


def main():
    settings = Settings.from_env()
    # Ensure collection id is populated; otherwise the script will error
    # If collection id isn't set, attempt to locate 'Blog Posts' collection on the site.
    # Discover and prefer the Blog Posts collection for this run.
    site = settings.webflow_site_id
    if not site:
        print("WEBFLOW_SITE_ID not set in env; aborting")
        return
    import requests
    headers = {"Authorization": f"Bearer {settings.webflow_api_token}", "accept-version": "1.0.0"}
    cols = requests.get(f'https://api.webflow.com/v2/sites/{site}/collections', headers=headers, timeout=10).json().get('collections', [])
    blog = next((c for c in cols if c.get('displayName','').lower()=='blog posts' or c.get('slug')=='blog'), None)
    if blog:
        settings.webflow_collection_id = blog['id']
        print('Using Blog Posts collection id:', settings.webflow_collection_id)
    else:
        print("Blog Posts collection not found; aborting")
        return

    image_agent = ImageAgent(llm=DummyLLM(), name="img", system_prompt="",)
    context = AgentContext(
        brand_voice="Helpful, concise",
        brand_colors="#0b5cff,#ffffff",
        target_audience="job seekers",
        distribution_channels="blog"
    )
    design = image_agent.design({"topic": "career guidance"}, rag_context="", context=context)
    # design.content should be a dict; adapt if returned differently
    if isinstance(design.content, dict) and design.content.get("error"):
        print("ImageAgent returned error:", design.content)
        return

    prompt = design.content.get("prompt")
    alt = design.content.get("alt_text")

    gen = ImageGenerator()
    out = gen.generate(prompt, out_name=None)
    print("Generated image saved to", out)

    pub_agent = PublishingAgent(llm=DummyLLM(), name="pub", system_prompt="", settings=settings)
    meta = {"title": "Generated Image Post", "summary": "Post created with generated image", "html": "<p>Generated content</p>", "image_alt": alt}
    res = pub_agent.upload_to_webflow(meta, blog_markdown="<p>body</p>", image_file_path=str(out))
    print("Publish result:", res)


if __name__ == "__main__":
    main()
