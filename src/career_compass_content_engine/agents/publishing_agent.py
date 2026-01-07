from __future__ import annotations


import hashlib
import json
import mimetypes
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


import requests
try:
    from markdown import markdown as md_to_html
except Exception:
    md_to_html = None
from langchain_core.language_models.chat_models import BaseChatModel

from ..config import Settings
from .base import AgentResult, BaseAgent


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
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self.session.request(method, url, timeout=kwargs.pop("timeout", 60), **kwargs)
                # treat 5xx and 429 as retryable
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise requests.HTTPError(f"Retryable HTTP error {resp.status_code}")
                return resp
            except (requests.RequestException, requests.HTTPError) as exc:
                last_exc = exc
                if attempt == self._max_retries:
                    break
                sleep = self._backoff_factor * (2 ** (attempt - 1))
                time.sleep(sleep)
        raise RuntimeError(f"Request to {url} failed after {self._max_retries} attempts: {last_exc}")

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

    def upload_asset(self, site_id: str, file_path: Path) -> Optional[str]:
        metadata = self.create_asset_metadata(site_id, file_path)
        upload_url = metadata.get("uploadUrl")
        upload_details = metadata.get("uploadDetails", {}) or {}
        hosted_url = metadata.get("hostedUrl")

        if not upload_url:
            return hosted_url

        # uploadDetails may be either a dict of form fields or contain a 'fields' key.
        if isinstance(upload_details, dict) and "fields" in upload_details:
            form_data = upload_details.get("fields", {})
        else:
            form_data = upload_details or {}
        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or "application/octet-stream"

        try:
            with file_path.open("rb") as f:
                files = {"file": (file_path.name, f, mime_type)}
                # upload URL may be an S3 or similar; use requests.post directly but with retries
                resp = requests.post(upload_url, data=form_data, files=files, timeout=60)
                resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to upload asset {file_path.name} to {upload_url}: {exc}")

        return hosted_url or metadata.get("fileUrl")

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

    def create_draft_blog_item(
        self,
        collection_id: str,
        *,
        title: str,
        description: str,
        article_html: str,
        author_item_id: Optional[str],
        main_image_url: str,
        thumb_image_url: Optional[str] = None,
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
                # Always include both `url` and `alt` keys so Webflow receives
                # a consistent image object. Also include `alt_text` for
                # compatibility with any templates expecting that key.
                field_slug_main_image: {
                    "url": main_image_url,
                    "alt": main_image_alt,
                    "alt_text": main_image_alt,
                },
            },
        }

        if thumb_image_url:
            payload["fieldData"][field_slug_thumb_image] = {
                "url": thumb_image_url,
                "alt": thumb_image_alt,
                "alt_text": thumb_image_alt,
            }

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



PUBLISH_TEMPLATE = """
You are the senior Publishing & SEO QA agent for Career Compass,
a premium career development and coaching app for early-career professionals.

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
        prompt = f"{PUBLISH_TEMPLATE}\n\nContent to publish:\n{blog_markdown}"
        return self.run(prompt)

    def upload_to_webflow(
        self,
        metadata: Dict[str, str],
        blog_markdown: str,
        image_url: Optional[str] = None,
        image_file_path: Optional[str] = None,
        thumb_image_url: Optional[str] = None,
    ) -> Dict[str, str]:
        if not self.settings.webflow_api_token or not self.settings.webflow_collection_id or not self.settings.webflow_site_id:
            return {
                "status": "skipped",
                "reason": "Missing WEBFLOW_API_TOKEN, WEBFLOW_SITE_ID, or WEBFLOW_COLLECTION_ID; configure to enable publishing.",
            }

        cms = WebflowCMS(self.settings.webflow_api_token)

        hosted_image_url = image_url
        if not hosted_image_url and image_file_path:
            path = Path(image_file_path)
            if path.exists():
                try:
                    hosted_image_url = cms.upload_asset(self.settings.webflow_site_id, path)
                except Exception as exc:  # pragma: no cover - network call
                    return {"status": "error", "reason": f"Asset upload failed: {exc}"}

        if not hosted_image_url:
            return {
                "status": "skipped",
                "reason": "No image URL or file path provided; skipping Webflow CMS creation.",
            }

        # Derive title: prefer metadata, then suggested_title, then H1 from the markdown, else fallback
        title = metadata.get("title") or metadata.get("suggested_title") or None
        description = metadata.get("summary") or metadata.get("excerpt") or ""
        # If the metadata provided pre-rendered HTML, prefer it. Otherwise convert Markdown to HTML.
        if metadata.get("html"):
            article_html = metadata.get("html")
        else:
            if md_to_html:
                try:
                    article_html = md_to_html(blog_markdown, extensions=["extra", "sane_lists"])
                except Exception:
                    article_html = f"<pre>{blog_markdown}</pre>"
            else:
                # Minimal fallback: wrap paragraphs
                paragraphs = [p.strip() for p in (blog_markdown or "").split("\n\n") if p.strip()]
                article_html = "".join("<p>{}</p>".format(p.replace("\n", " ")) for p in paragraphs)

        if not title:
            # try to pull the H1 from the markdown as a sensible default
            m = re.search(r"^#\s+(.+)$", blog_markdown or "", flags=re.MULTILINE)
            if m:
                title = m.group(1).strip()
            else:
                title = "Career Compass Blog"
        author_id = self.settings.webflow_author_id

        # Discover collection field slugs so we send the correct field names.
        try:
            col_fields = cms.get_collection_fields(self.settings.webflow_collection_id)
        except Exception:
            col_fields = []

        # helpers to pick slugs
        def first_slug_by_type(t):
            for f in col_fields:
                if f.get("type") == t:
                    return f.get("slug")
            return None

        def first_plaintext_slug(exclude=()):
            for f in col_fields:
                if f.get("type") == "PlainText" and f.get("slug") not in exclude:
                    return f.get("slug")
            return None

        # determine slugs with sensible defaults
        field_slug_title = "name"
        field_slug_slug = "slug"
        field_slug_article = "post-body"
        field_slug_main_image = "main-image"
        field_slug_thumb_image = "thumbnail-image"
        field_slug_description = "description"
        field_slug_author = "author"

        # prefer actual slugs present in the collection
        if any(f.get("slug") == "name" for f in col_fields):
            field_slug_title = "name"
        else:
            field_slug_title = first_plaintext_slug() or field_slug_title

        if any(f.get("slug") == "slug" for f in col_fields):
            field_slug_slug = "slug"
        else:
            field_slug_slug = first_plaintext_slug(exclude=(field_slug_title,)) or field_slug_slug

        field_slug_article = first_slug_by_type("RichText") or field_slug_article
        image_slugs = [f.get("slug") for f in col_fields if f.get("type") == "Image"]
        field_slug_main_image = image_slugs[0] if image_slugs else field_slug_main_image
        field_slug_thumb_image = image_slugs[1] if len(image_slugs) > 1 else (image_slugs[0] if image_slugs else field_slug_thumb_image)
        field_slug_description = first_plaintext_slug(exclude=(field_slug_title, field_slug_slug)) or field_slug_description
        # find reference field for author if present
        for f in col_fields:
            if f.get("type") == "Reference":
                field_slug_author = f.get("slug")
                break

        # ensure alt text fallbacks are sane strings
        main_image_alt = metadata.get("image_alt") or metadata.get("alt") or title or ""
        thumb_image_alt = metadata.get("thumbnail_alt") or metadata.get("thumb_alt") or main_image_alt
        # Prepare extra field data: category assignment and short SEO summary (<=60 chars)
        extra_fields: Dict[str, Any] = {}

        # derive a short engaging summary (1-2 sentences). Do not enforce a strict character limit.
        raw_summary = metadata.get("search_summary") or metadata.get("summary") or metadata.get("excerpt") or ""
        short_summary = ""
        if raw_summary:
            # Try to keep it to the first two sentences for a concise meta description.
            import re

            sentences = [s.strip() for s in re.split(r'(?<=[\.!?])\s+', raw_summary) if s.strip()]
            if not sentences:
                short_summary = raw_summary
            else:
                short_summary = " ".join(sentences[:2])
                # final safety: strip trailing whitespace
                short_summary = short_summary.strip()

        # Discover likely category and summary/meta slugs from collection fields
        category_field_slug = None
        category_field_type = None
        summary_field_slug = None
        for f in col_fields:
            slug = (f.get("slug") or "")
            name = (f.get("name") or "")
            lslug = slug.lower()
            lname = name.lower()
            if not category_field_slug and ("category" in lslug or "category" in lname or "categories" in lslug):
                category_field_slug = slug
                category_field_type = f.get("type")
            if not summary_field_slug and any(k in lslug for k in ("summary", "meta", "description", "excerpt")):
                summary_field_slug = slug

        if not summary_field_slug:
            summary_field_slug = field_slug_description

        # If a category id is configured in settings or metadata, attach it to the extra fields
        category_id = metadata.get("category_id") or getattr(self.settings, "webflow_category_id", None)
        if category_id and category_field_slug:
            if category_field_type and "Multi" in (category_field_type or ""):
                extra_fields[category_field_slug] = [category_id]
            elif category_field_type == "Reference":
                extra_fields[category_field_slug] = category_id
            else:
                extra_fields[category_field_slug] = metadata.get("category") or "Resources"

        # Add short summary and ensure it meets SEO guidelines (target 140–160 chars,
        # hard max 200). If it doesn't meet the target, ask the LLM to rewrite it.
        if short_summary:
            def _rewrite_summary_to_target(text: str) -> str:
                base_prompt = (
                    "Rewrite the following meta description to meet SEO guidelines.\n\n"
                    "Constraints:\n"
                    "- Target length: 140–160 characters (aim for ~155).\n"
                    "- Hard max: 200 characters (truncate if necessary).\n"
                    "- Output must be 2-3 sentences.\n"
                    "- Put the primary keyword or topic phrase near the start.\n"
                    "- Include a clear, specific benefit (what the reader will be able to do or learn).\n"
                    "- Do NOT include surrounding quotation marks or extra commentary; return raw text only.\n\n"
                )

                def call_once(prompt_text: str) -> str:
                    try:
                        res = self.run(prompt_text)
                        new = None
                        if res and res.content:
                            new = res.content.get("message")
                        if isinstance(new, dict):
                            new = new.get("summary") or next(iter(new.values()), None)
                        if not new:
                            return text
                        new = str(new).strip()
                        # strip surrounding quotes if model included them
                        if len(new) >= 2 and ((new[0] == new[-1] == '"') or (new[0] == new[-1] == "'")):
                            new = new[1:-1].strip()
                        if len(new) > 200:
                            new = new[:197].rstrip() + "..."
                        return new
                    except Exception:
                        return text

                inst = base_prompt + "Original summary:\n\"\"\"" + text + "\"\"\""
                # First attempt
                out = call_once(inst)
                # If too short, try a stronger expand instruction once
                if len(out) < 140:
                    expand_prompt = (
                        base_prompt
                        + "If the rewritten summary is shorter than 140 characters, expand it so it falls between 140 and 160 characters, keeping the same constraints.\n\n"
                        + "Original summary:\n\"\"\""
                        + text
                        + "\"\"\""
                    )
                    out = call_once(expand_prompt)
                return out

            rewritten = short_summary
            if not (140 <= len(short_summary) <= 160):
                rewritten = _rewrite_summary_to_target(short_summary)
            if len(rewritten) > 200:
                rewritten = rewritten[:197].rstrip() + "..."

            extra_fields[summary_field_slug] = rewritten
            # Also ensure a dedicated `summary` field receives the short summary
            extra_fields["summary"] = rewritten

        # Add required Webflow taxonomy/reference fields for our collection
        # These are set to fixed IDs for category and author as requested.
        # Allow metadata to override if explicitly provided.
        extra_fields.setdefault("blog-post-category", metadata.get("blog-post-category") or "6759b39c1464ba09dc7d785e")
        extra_fields.setdefault("blog-post-author", metadata.get("blog-post-author") or "6759b39c1464ba09dc7d783d")

        try:
            response = cms.create_draft_blog_item(
                self.settings.webflow_collection_id,
                title=title,
                description=description,
                article_html=article_html,
                author_item_id=author_id,
                main_image_url=hosted_image_url,
                thumb_image_url=thumb_image_url or hosted_image_url,
                main_image_alt=main_image_alt,
                thumb_image_alt=thumb_image_alt,
                extra_field_data=extra_fields,
                field_slug_title=field_slug_title,
                field_slug_slug=field_slug_slug,
                field_slug_description=field_slug_description,
                field_slug_article=field_slug_article,
                field_slug_author=field_slug_author,
                field_slug_main_image=field_slug_main_image,
                field_slug_thumb_image=field_slug_thumb_image,
            )
        except Exception as exc:  # pragma: no cover - network call
            return {"status": "error", "reason": f"Webflow CMS create failed: {exc}"}

        return {"status": "success", "webflow_item": response}

