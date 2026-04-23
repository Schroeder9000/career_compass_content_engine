"""
Standalone diagnostic script: generate an image and publish a minimal blog post to Webflow.

Steps:
  1. Generate a test image with gpt-image-1
  2. Upload the image to Webflow Assets and capture hostedUrl
  3. Create a minimal draft CMS item with the image attached
  4. Inspect the created item and confirm image fields are populated

Run from the repo root:
    source .venv/bin/activate
    python scripts/test_image_to_webflow.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# ── make sure the src package is importable regardless of install state ──────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from career_compass_content_engine.config import Settings
from career_compass_content_engine.agents.image_generator import ImageGenerator
from career_compass_content_engine.agents.publishing_agent import WebflowCMS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_image_to_webflow")

DIVIDER = "─" * 60

# ── configurable test values ─────────────────────────────────────────────────
IMAGE_PROMPT = (
    "A clean, modern illustration of a young professional writing at a desk, "
    "warm light, soft pastel background, minimal style."
)
IMAGE_SIZE = "1536x1024"  # native 16:9 for gpt-image-1
TEST_TITLE = "Test Post — Image Upload Diagnostic"
TEST_SLUG = "test-post-image-upload-diagnostic"
TEST_BODY_HTML = "<p>This is a diagnostic test post. It can be safely deleted.</p>"
TEST_DESCRIPTION = "Diagnostic post for image upload testing."


def step(n: int, label: str) -> None:
    log.info("")
    log.info("%s", DIVIDER)
    log.info("STEP %d: %s", n, label)
    log.info("%s", DIVIDER)


def main() -> None:
    settings = Settings.from_env()

    # ── guard: check required env vars ───────────────────────────────────────
    missing = []
    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not settings.webflow_api_token:
        missing.append("WEBFLOW_API_TOKEN")
    if not settings.webflow_site_id:
        missing.append("WEBFLOW_SITE_ID")
    if not settings.webflow_collection_id:
        missing.append("WEBFLOW_COLLECTION_ID")
    if missing:
        log.error("Missing required env vars: %s", ", ".join(missing))
        sys.exit(1)

    log.info("Settings loaded OK — site_id=%s  collection_id=%s",
             settings.webflow_site_id, settings.webflow_collection_id)

    # ─────────────────────────────────────────────────────────────────────────
    step(1, "Generate image with gpt-image-1")
    # ─────────────────────────────────────────────────────────────────────────
    gen = ImageGenerator(out_dir=".cache/images")
    gen.openai_api_key = settings.openai_api_key
    cached_path = Path(".cache/images/test_diagnostic_image.jpg")
    if cached_path.exists():
        image_path = cached_path
        log.info("✅ Reusing cached image: %s  (%d bytes)", image_path, image_path.stat().st_size)
    else:
        try:
            image_path = gen.generate(IMAGE_PROMPT, out_name="test_diagnostic_image.jpg", size=IMAGE_SIZE)
            log.info("✅ Image generated: %s  (%d bytes)", image_path, image_path.stat().st_size)
        except Exception as exc:
            log.error("❌ Image generation FAILED: %s", exc)
            sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────────
    step(2, "Upload image to Webflow Assets")
    # ─────────────────────────────────────────────────────────────────────────
    cms = WebflowCMS(settings.webflow_api_token)
    try:
        asset_result = cms.upload_asset(settings.webflow_site_id, image_path)
        log.info("upload_asset() returned: %s", json.dumps(asset_result, indent=2))
    except Exception as exc:
        log.error("❌ upload_asset() FAILED: %s", exc)
        sys.exit(1)

    if not asset_result or not asset_result.get("url"):
        log.error("❌ upload_asset() returned no URL. Full result: %s", asset_result)
        sys.exit(1)

    hosted_url = asset_result["url"]
    log.info("✅ Hosted image URL: %s", hosted_url)

    # ─────────────────────────────────────────────────────────────────────────
    step(3, "Inspect collection fields")
    # ─────────────────────────────────────────────────────────────────────────
    try:
        col_fields = cms.get_collection_fields(settings.webflow_collection_id)
        log.info("Collection has %d fields total. Raw field details:", len(col_fields))
        for f in col_fields:
            log.info("  slug=%-45s  type=%s", f.get("slug"), f.get("type"))
        image_fields = [f for f in col_fields if (f.get("type") or "").lower() in ("image",)]
        log.info("Image-type fields: %s", [f.get("slug") for f in image_fields])
    except Exception as exc:
        log.warning("Could not inspect collection fields: %s", exc)
        col_fields = []
        image_fields = []

    # ─────────────────────────────────────────────────────────────────────────
    step(4, "Create minimal draft CMS item with image")
    # ─────────────────────────────────────────────────────────────────────────

    # Determine the correct field slugs for this collection from the live schema
    all_slugs = [f.get("slug") for f in col_fields]
    type_by_slug = {f.get("slug"): f.get("type") for f in col_fields}

    image_field_slugs = [f.get("slug") for f in col_fields
                         if (f.get("type") or "").lower() == "image"]
    main_image_slug = (
        "blog-post-featured-image-photo"
        if "blog-post-featured-image-photo" in image_field_slugs
        else (image_field_slugs[0] if image_field_slugs else "main-image")
    )
    thumb_image_slug = (
        "blog-post-thumbnail-image-photo"
        if "blog-post-thumbnail-image-photo" in image_field_slugs
        else (image_field_slugs[1] if len(image_field_slugs) > 1 else main_image_slug)
    )

    # Find the RichText / article field (handles Webflow type name variations)
    richtext_candidates = [
        f.get("slug") for f in col_fields
        if (f.get("type") or "").lower() in ("richtext", "rich-text", "rich_text", "richtextblock")
    ]
    # Also match by slug keyword as a fallback (handles unknown type names)
    if not richtext_candidates:
        richtext_candidates = [
            s for s in all_slugs
            if any(k in (s or "").lower() for k in ("rich", "body", "content", "article", "text"))
        ]
    article_field_slug = richtext_candidates[0] if richtext_candidates else "blog-post-richt-text"
    log.info("Using main_image_slug=%s  thumb_image_slug=%s  article_slug=%s",
             main_image_slug, thumb_image_slug, article_field_slug)

    # Build the image field value exactly as create_draft_blog_item does
    image_field_value = {"url": hosted_url, "alt": "Test diagnostic image"}

    # Build the minimal payload manually so we can inspect exactly what gets sent
    import requests as _requests
    import unicodedata, re as _re

    def slugify(text: str) -> str:
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        text = _re.sub(r"[^a-zA-Z0-9\s-]", "", text).strip().lower()
        text = _re.sub(r"[\s_-]+", "-", text)
        return text

    field_data: dict = {
        "name": TEST_TITLE,
        "slug": TEST_SLUG,
        article_field_slug: TEST_BODY_HTML,
        main_image_slug: image_field_value,
        thumb_image_slug: image_field_value,
        # required fixed-ID fields for this collection
        "blog-post-category": "6759b39c1464ba09dc7d785e",
        "blog-post-author": "6759b39c1464ba09dc7d783d",
    }

    # Add description field if we can identify it
    for f in col_fields:
        slug = f.get("slug", "")
        if slug in ("description", "summary") and slug not in field_data:
            field_data[slug] = TEST_DESCRIPTION

    log.info("Payload fieldData keys: %s", list(field_data.keys()))
    log.info("Image field value being sent: %s", json.dumps(image_field_value))

    payload = {
        "isArchived": False,
        "isDraft": True,
        "fieldData": field_data,
    }

    url = f"https://api.webflow.com/v2/collections/{settings.webflow_collection_id}/items"
    try:
        resp = cms._request_with_retries("POST", url, json=payload)
        log.info("Webflow response status: %d", resp.status_code)
        body = resp.json()
        if resp.status_code not in (200, 201, 202):
            log.error("❌ Create item FAILED: %s", json.dumps(body, indent=2))
            sys.exit(1)
    except Exception as exc:
        log.error("❌ Create item request FAILED: %s", exc)
        sys.exit(1)

    created_id = body.get("id")
    log.info("✅ Created item id: %s", created_id)

    # ─────────────────────────────────────────────────────────────────────────
    step(5, "Inspect created item — verify image fields")
    # ─────────────────────────────────────────────────────────────────────────
    returned_field_data = body.get("fieldData", {})
    main_img = returned_field_data.get(main_image_slug)
    thumb_img = returned_field_data.get(thumb_image_slug)

    log.info("Returned fieldData keys: %s", list(returned_field_data.keys()))
    log.info("%s field value: %s", main_image_slug, json.dumps(main_img))
    log.info("%s field value: %s", thumb_image_slug, json.dumps(thumb_img))

    if main_img and (main_img.get("url") or main_img.get("fileId")):
        log.info("✅ Main image is ATTACHED in Webflow item.")
    else:
        log.warning("⚠️  Main image field is EMPTY in Webflow item.")

    # ─────────────────────────────────────────────────────────────────────────
    log.info("")
    log.info("%s", DIVIDER)
    log.info("SUMMARY")
    log.info("%s", DIVIDER)
    log.info("Image file      : %s", image_path)
    log.info("Hosted URL      : %s", hosted_url)
    log.info("Webflow item id : %s", created_id)
    log.info("Image attached  : %s", bool(main_img and (main_img.get("url") or main_img.get("fileId"))))
    log.info("")
    log.info("You can view/delete the draft at:")
    log.info("  https://webflow.com/dashboard  → Blog Posts → %s", TEST_TITLE)


if __name__ == "__main__":
    main()
