from __future__ import annotations


import hashlib
import logging
import mimetypes
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


import requests
try:
    from markdown import markdown as md_to_html
except Exception:
    md_to_html = None
from langchain_core.language_models.chat_models import BaseChatModel

from ..config import Settings
from ..product_context import product_context
from ..publication_policy import metadata_blockers
from .base import AgentResult, BaseAgent

logger = logging.getLogger(__name__)


WEBFLOW_API_BASE = "https://api.webflow.com/v2"


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9\s-]", "", text).strip().lower()
    text = re.sub(r"[\s_-]+", "-", text)
    return text


class WebflowCMS:
    """Lightweight Webflow client for asset upload and CMS item creation."""

    def __init__(self, access_token: str) -> None:
        if not access_token:
            raise ValueError("Missing WEBFLOW_API_TOKEN")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "accept-version": "1.0.0",
            }
        )
        # basic retry/backoff config for network resilience
        self._max_retries = 3
        self._backoff_factor = 1

    def _request_with_retries(self, method: str, url: str, **kwargs):
        import time

        last_exc = None
        timeout = kwargs.pop("timeout", 60)
        # A timed-out create may have succeeded. Never blindly retry POST writes.
        attempts = self._max_retries if method.upper() in {"GET", "PATCH"} else 1
        for attempt in range(1, attempts + 1):
            try:
                resp = self.session.request(method, url, timeout=timeout, **kwargs)
                # treat 5xx and 429 as retryable
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise requests.HTTPError(f"Retryable HTTP error {resp.status_code}")
                return resp
            except (requests.RequestException, requests.HTTPError) as exc:
                last_exc = exc
                if attempt == attempts:
                    break
                sleep = self._backoff_factor * (2 ** (attempt - 1))
                time.sleep(sleep)
        raise RuntimeError(f"Request to {url} failed after {attempts} attempts: {last_exc}")

    def create_asset_metadata(self, site_id: str, file_path: Path) -> Dict[str, Any]:
        file_name = file_path.name
        file_hash = md5_file(file_path)
        url = f"{WEBFLOW_API_BASE}/sites/{site_id}/assets"
        payload = {"fileName": file_name, "fileHash": file_hash}
        try:
            resp = self._request_with_retries("POST", url, json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            raise RuntimeError(f"Failed to create asset metadata for {file_name}: {exc}")

    @staticmethod
    def _asset_hosted_url(metadata: Dict[str, Any]) -> Optional[str]:
        """Return the full CDN/S3 URL that Webflow can resolve for CMS image fields."""
        url = metadata.get("assetUrl") or metadata.get("hostedUrl")
        if url and url.startswith("https://s3.amazonaws.com/webflow-prod-assets/"):
            url = url.replace("https://s3.amazonaws.com/webflow-prod-assets/",
                              "https://cdn.prod.website-files.com/", 1)
        return url or None

    def upload_asset(self, site_id: str, file_path: Path) -> Optional[Dict[str, str]]:
        metadata = self.create_asset_metadata(site_id, file_path)
        upload_url = metadata.get("uploadUrl")
        upload_details = metadata.get("uploadDetails", {}) or {}
        hosted_url = self._asset_hosted_url(metadata)

        if not upload_url:
            # Asset with same hash already exists; use its hosted URL.
            if hosted_url:
                return {"url": hosted_url}
            return None

        # uploadDetails may be either a dict of form fields or contain a 'fields' key.
        if isinstance(upload_details, dict) and "fields" in upload_details:
            form_data = upload_details.get("fields", {})
        else:
            form_data = upload_details or {}
        # Slim Modal images may lack the OS MIME database entry for WebP.
        mime_type = "image/webp" if file_path.suffix.lower() == ".webp" else mimetypes.guess_type(file_path)[0]
        mime_type = mime_type or "application/octet-stream"

        try:
            with file_path.open("rb") as f:
                files = {"file": (file_path.name, f, mime_type)}
                # upload URL may be an S3 or similar; use requests.post directly but with retries
                resp = requests.post(upload_url, data=form_data, files=files, timeout=60)
                resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to upload asset {file_path.name} to {upload_url}: {exc}")

        if hosted_url:
            return {"url": hosted_url}
        return None

    @staticmethod
    def _build_image_field_value(image: Any, alt_text: str) -> Optional[Dict[str, str]]:
        """Build a CMS image field value from an upload_asset() result or URL string.

        Webflow CMS image fields work when given {"url": <full_hosted_url>, "alt": ...}.
        Webflow internally creates its own fileId and CDN copy from the URL.
        """
        if not image:
            return None

        url: Optional[str] = None
        if isinstance(image, dict):
            url = image.get("url") or image.get("hostedUrl")
        elif isinstance(image, str):
            url = image

        if url:
            return {"url": str(url), "alt": alt_text or ""}
        return None

    def get_collection_fields(self, collection_id: str):
        url = f"{WEBFLOW_API_BASE}/collections/{collection_id}"
        try:
            resp = self._request_with_retries("GET", url)
            resp.raise_for_status()
            return resp.json().get("fields", [])
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch collection {collection_id} fields: {exc}")

    def list_items(self, collection_id: str, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """Retrieve published items from a Webflow collection.
        
        Args:
            collection_id: The collection ID to fetch items from
            limit: Maximum number of items to retrieve per request (max 100)
            offset: Number of items to skip for pagination
            
        Returns:
            Dict containing 'items' list and pagination info
        """
        url = f"{WEBFLOW_API_BASE}/collections/{collection_id}/items"
        params = {"limit": min(limit, 100), "offset": offset}
        try:
            resp = self._request_with_retries("GET", url, params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            raise RuntimeError(f"Failed to list items from collection {collection_id}: {exc}")

    def find_item_by_slug(self, collection_id: str, slug: str) -> Optional[Dict[str, Any]]:
        """Return the first CMS item whose slug field matches *slug*, or None."""
        offset = 0
        page_size = 100
        while True:
            try:
                result = self.list_items(collection_id, limit=page_size, offset=offset)
            except Exception as exc:
                logger.warning("[WebflowCMS] find_item_by_slug list_items failed: %s", exc)
                raise RuntimeError("Cannot check for an existing CMS slug") from exc
            items = result.get("items", [])
            for item in items:
                if item.get("fieldData", {}).get("slug") == slug:
                    return item
            total = result.get("pagination", {}).get("total", 0)
            offset += len(items)
            if not items or offset >= total:
                break
        return None

    def update_item(self, collection_id: str, item_id: str, field_data: Dict[str, Any]) -> Dict[str, Any]:
        """PATCH an existing CMS item with the supplied field data."""
        url = f"{WEBFLOW_API_BASE}/collections/{collection_id}/items/{item_id}"
        try:
            resp = self._request_with_retries("PATCH", url, json={"fieldData": field_data})
            if resp.status_code not in (200, 201, 202):
                raise RuntimeError(f"Update item failed: {resp.status_code} {resp.text}")
            return resp.json()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to update CMS item {item_id} in collection {collection_id}: {exc}"
            )

    def create_draft_blog_item(
        self,
        collection_id: str,
        *,
        title: str,
        description: str,
        article_html: str,
        author_item_id: Optional[str],
        main_image_url: Optional[Any] = None,
        thumb_image_url: Optional[Any] = None,
        main_image_alt: str = "",
        thumb_image_alt: str = "",
        field_slug_title: str = "name",
        field_slug_slug: str = "slug",
        field_slug_description: str = "description",
        field_slug_article: str = "post-body",
        field_slug_author: str = "author",
        field_slug_main_image: str = "main-image",
        field_slug_thumb_image: str = "thumbnail-image",
        extra_field_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Use the single-item create endpoint for predictable behavior when creating one draft.
        url = f"{WEBFLOW_API_BASE}/collections/{collection_id}/items"
        payload: Dict[str, Any] = {
            "isArchived": False,
            "isDraft": True,
            "fieldData": {
                field_slug_title: title,
                field_slug_slug: slugify(title),
                field_slug_description: description,
                field_slug_article: article_html,
            },
        }

        main_image_value = self._build_image_field_value(main_image_url, main_image_alt)
        if main_image_value:
            payload["fieldData"][field_slug_main_image] = main_image_value

        thumb_image_value = self._build_image_field_value(thumb_image_url, thumb_image_alt)
        if thumb_image_value:
            payload["fieldData"][field_slug_thumb_image] = thumb_image_value

        if author_item_id:
            payload["fieldData"][field_slug_author] = author_item_id

        # merge any extra field data (categories, short summary, tags, etc.)
        if extra_field_data:
            payload["fieldData"].update(extra_field_data)

        try:
            resp = self._request_with_retries("POST", url, json=payload)
            if resp.status_code not in (200, 201, 202):
                raise RuntimeError(f"Create item failed: {resp.status_code} {resp.text}")
            return resp.json()
        except Exception as exc:
            raise RuntimeError(f"Failed to create CMS item in collection {collection_id}: {exc}")

    def publish_items(self, collection_id: str, item_ids: list[str]) -> Dict[str, Any]:
        if not item_ids:
            raise ValueError("item_ids must contain at least one item id")
        url = f"{WEBFLOW_API_BASE}/collections/{collection_id}/items/publish"
        payload = {"itemIds": item_ids}
        try:
            resp = self._request_with_retries("POST", url, json=payload)
            if resp.status_code not in (200, 201, 202):
                raise RuntimeError(f"Publish items failed: {resp.status_code} {resp.text}")
            body = resp.json()
            published_ids = set(body.get("publishedItemIds") or [])
            errors = body.get("errors") or []
            missing = [item_id for item_id in item_ids if item_id not in published_ids]
            if errors or missing:
                raise RuntimeError(
                    f"Publish API returned errors={errors} missing_published_ids={missing}"
                )
            return body
        except Exception as exc:
            raise RuntimeError(f"Failed to publish CMS item(s) in collection {collection_id}: {exc}")

PUBLISH_TEMPLATE = """
You are the senior Publishing & SEO QA agent for Career Compass,
the company behind Promotion Case, a focused promotion and pay-conversation preparation product.

You have final authority on whether a blog post is ready to be published.

Your responsibilities:
- Ensure the content is clear, valuable, and aligned with Career Compass brand voice
- Generate clean, SEO-ready publishing metadata
- Detect common failure modes (thin content, fluff, redundancy, weak framing)
- Protect long-term search and brand equity

---

### INPUT CONTENT
Full blog article content:
{article}

---

### BRAND VOICE & QUALITY BAR
Career Compass content must be:
- Practical, thoughtful, and grounded (no hustle porn, no generic motivation)
- Written for early-career professionals (ages ~22–35)
- Action-oriented with real takeaways
- Clear and skimmable (short paragraphs, strong structure)
- Credible and calm — not salesy or preachy

---

### SEO & CMS CONSTRAINTS
Title:
- ≤ 60 characters
- Clear value, not clickbait

Slug:
- Lowercase
- Hyphenated
- Concise, keyword-aligned

Summary (SEO meta description):
- Target length: 140–160 characters (aim for ~155).
- Hard max: 200 characters (if necessary, truncate to 200).
- Format: 1–2 plain-language sentences.
- Put the primary keyword or topic phrase near the start.
- Ensure the summary is compelling and engaging, encouraging readers to click and read the article.
- Include a clear, specific benefit (what the reader will be able to do or learn).
- If a draft summary is provided, rewrite it to meet these constraints.

Excerpt:
- 1 short paragraph that invites reading without repeating the title
- Written in a human, conversational tone

Tags:
- 3–6 tags max
- Use consistent taxonomy (career-growth, managers, productivity, etc.)
- Avoid duplicates, plurals, or near-synonyms

---

### QUALITY GATE
Set `content_ready` to TRUE only if ALL are true:
- The article delivers clear, non-obvious value
- The opening frames a real problem
- The structure is coherent and scannable
- There is at least one concrete, actionable takeaway
- The content matches Career Compass tone and positioning

If any of these fail, set `content_ready` to FALSE.

---

### OUTPUT FORMAT
Return a **strict JSON object** with the following keys:

- title
- slug
- summary
- excerpt
- tags (array of strings)
- content_ready (boolean)
- blocking_issues (array of strings, empty if ready)

Do not include commentary outside the JSON.
"""


@dataclass
class PublishingAgent(BaseAgent):
    """Agent that prepares and uploads content to Webflow."""

    llm: BaseChatModel
    settings: Settings

    def prepare_metadata(self, blog_markdown: str) -> AgentResult:
        prompt = f"{PUBLISH_TEMPLATE}{product_context()}\n\nContent to publish:\n{blog_markdown}"
        return self.run(prompt)

    def upload_to_webflow(
        self,
        metadata: Dict[str, Any],
        blog_markdown: str,
        image_url: Optional[Any] = None,
        image_file_path: Optional[str] = None,
        thumb_image_url: Optional[Any] = None,
    ) -> Dict[str, Any]:
        issues = metadata_blockers(metadata)
        if issues:
            return {"status": "blocked", "reason": "; ".join(issues)}
        mode = self.settings.publish_mode
        if mode not in {"live", "draft"}:
            return {"status": "error", "reason": "PUBLISH_MODE must be live or draft"}
        if not all((self.settings.webflow_api_token, self.settings.webflow_collection_id,
                    self.settings.webflow_site_id)):
            return {"status": "skipped", "reason": "Missing Webflow publishing configuration"}

        cms = WebflowCMS(self.settings.webflow_api_token)
        collection_id = self.settings.webflow_collection_id
        try:
            fields = {f["slug"]: f for f in cms.get_collection_fields(collection_id)}
        except Exception as exc:
            return {"status": "error", "reason": f"Cannot verify CMS schema: {exc}"}
        # This is the existing Aptonic collection contract. Schema drift must be
        # reviewed instead of silently writing to the first field of a given type.
        contract = {
            "name": "PlainText", "slug": "PlainText", "summary": "PlainText",
            "blog-post-richt-text": "RichText",
            "blog-post-featured-image-photo": "Image",
            "blog-post-thumbnail-image-photo": "Image",
            "blog-post-author": "Reference", "blog-post-category": "Reference",
        }
        mismatches = [key for key, kind in contract.items()
                      if fields.get(key, {}).get("type") != kind]
        if "publication-date" in fields and fields["publication-date"].get("type") != "DateTime":
            mismatches.append("publication-date")
        if mismatches:
            return {"status": "error", "reason": "CMS schema mismatch: " + ", ".join(mismatches)}

        title = metadata["title"].strip()
        slug = slugify(metadata.get("slug") or title)
        if not slug or not blog_markdown.strip() or md_to_html is None:
            return {"status": "blocked", "reason": "Article, slug, and Markdown renderer are required"}
        try:
            existing = cms.find_item_by_slug(collection_id, slug)
        except Exception as exc:
            return {"status": "error", "reason": str(exc)}
        if existing:
            return {"status": "blocked", "reason": "CMS slug already exists; review before retrying",
                    "webflow_item_id": existing.get("id"), "slug": slug}

        hosted_image = image_url
        responsive = None
        if not hosted_image and image_file_path:
            try:
                from ..image_optimization import upload_responsive_images
                for key in ('featured-image-srcset', 'thumbnail-image-srcset'):
                    if fields.get(key, {}).get('type') != 'PlainText':
                        raise ValueError(f'Responsive image CMS field is missing: {key}')
                for prefix in ('featured', 'thumbnail'):
                    for dimension in ('width', 'height'):
                        key = f'{prefix}-image-{dimension}'
                        if fields.get(key, {}).get('type') != 'Number':
                            raise ValueError(f'Responsive image CMS field is missing: {key}')
                responsive = upload_responsive_images(Path(image_file_path),
                    lambda path: cms.upload_asset(self.settings.webflow_site_id, path))
                hosted_image = responsive['main']
            except Exception as exc:
                return {"status": "error", "reason": f"Asset upload failed: {exc}"}
        thumbnail = thumb_image_url or (responsive['thumbnail'] if responsive else hosted_image)
        for key, value in (("blog-post-featured-image-photo", hosted_image),
                           ("blog-post-thumbnail-image-photo", thumbnail)):
            if fields[key].get("isRequired") and not value:
                return {"status": "error", "reason": f"Required CMS image is missing: {key}"}

        article_html = md_to_html(blog_markdown, extensions=["extra", "sane_lists"])
        # The Webflow template owns the page title. Do not duplicate it in the body.
        article_html = re.sub(r"^\s*<h1[^>]*>.*?</h1>\s*", "", article_html,
                              count=1, flags=re.DOTALL | re.IGNORECASE)
        article_html = re.sub(r"<(\/?)h1(?=[ >])", r"<\1h2", article_html, flags=re.IGNORECASE)
        if not article_html.strip():
            return {"status": "blocked", "reason": "Article body is empty after removing its title"}
        summary = metadata["summary"].strip()
        maximum = int((fields["summary"].get("validations") or {}).get("maxLength") or 200)
        if len(summary) > maximum:
            summary = summary[:maximum].rsplit(" ", 1)[0].rstrip()
        author = self.settings.webflow_author_id or "6759b39c1464ba09dc7d783d"
        category = self.settings.webflow_category_id or "6759b39c1464ba09dc7d785e"
        image_alt = metadata.get("image_alt") or title
        extra_fields = {"slug": slug, "blog-post-category": category}
        if responsive:
            extra_fields['featured-image-srcset'] = responsive['srcset']
            extra_fields['featured-image-width'] = responsive['width']
            extra_fields['featured-image-height'] = responsive['height']
            if not thumb_image_url:
                extra_fields['thumbnail-image-srcset'] = responsive['srcset']
                extra_fields['thumbnail-image-width'] = responsive['thumbnail_width']
                extra_fields['thumbnail-image-height'] = responsive['thumbnail_height']
        # Keep public dates stable when Webflow later republishes an older item.
        # Manual drafts receive their date from the editor when ready to publish.
        if mode == "live" and "publication-date" in fields:
            extra_fields["publication-date"] = datetime.now(timezone.utc).isoformat()
        try:
            item = cms.create_draft_blog_item(
                collection_id, title=title, description=summary, article_html=article_html,
                author_item_id=author, main_image_url=hosted_image, thumb_image_url=thumbnail,
                main_image_alt=image_alt, thumb_image_alt=metadata.get("thumbnail_alt") or image_alt,
                field_slug_description="summary", field_slug_article="blog-post-richt-text",
                field_slug_author="blog-post-author", field_slug_main_image="blog-post-featured-image-photo",
                field_slug_thumb_image="blog-post-thumbnail-image-photo",
                extra_field_data=extra_fields,
            )
        except Exception as exc:
            return {"status": "error", "reason": f"CMS create outcome requires review before retry: {exc}",
                    "slug": slug}
        item_id = item.get("id")
        if not item_id:
            return {"status": "error", "reason": "CMS create returned no item ID", "slug": slug}
        actual_slug = item.get("fieldData", {}).get("slug") or slug
        result = {
            "status": "draft", "webflow_item_id": item_id, "webflow_item": item,
            "slug": actual_slug, "url": f"https://careeraiapp.com/blog/{actual_slug}",
            "metadata": {"title": title, "slug": actual_slug, "summary": summary},
            "image_attached": bool(hosted_image), "publish_mode": mode,
        }
        if mode == "draft":
            return result
        try:
            result["webflow_publish"] = cms.publish_items(collection_id, [item_id])
        except Exception as exc:
            return {**result, "status": "error", "reason": f"Item publication requires review: {exc}"}
        # Publication is deliberately item-scoped. There is no site publish client.
        result["status"] = "success"
        try:
            refreshed = cms._request_with_retries(
                "GET", f"{WEBFLOW_API_BASE}/collections/{collection_id}/items/{item_id}"
            )
            refreshed.raise_for_status()
            result["webflow_item"] = refreshed.json()
        except Exception:
            logger.warning("CMS publication acknowledged; item refresh unavailable")
        return result
