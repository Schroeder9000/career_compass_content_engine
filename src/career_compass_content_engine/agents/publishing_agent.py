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
            }
        )

    def create_asset_metadata(self, site_id: str, file_path: Path) -> Dict[str, Any]:
        file_name = file_path.name
        file_hash = md5_file(file_path)
        url = f"{WEBFLOW_API_BASE}/sites/{site_id}/assets"
        payload = {"fileName": file_name, "fileHash": file_hash}
        resp = self.session.post(url, data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        return resp.json()

    def upload_asset(self, site_id: str, file_path: Path) -> Optional[str]:
        metadata = self.create_asset_metadata(site_id, file_path)
        upload_url = metadata.get("uploadUrl")
        upload_details = metadata.get("uploadDetails", {}) or {}
        hosted_url = metadata.get("hostedUrl")

        if not upload_url:
            return hosted_url

        form_data = upload_details.get("fields", {})
        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or "application/octet-stream"

        with file_path.open("rb") as f:
            files = {"file": (file_path.name, f, mime_type)}
            resp = requests.post(upload_url, data=form_data, files=files, timeout=60)
            resp.raise_for_status()

        return hosted_url or metadata.get("fileUrl")

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
    ) -> Dict[str, Any]:
        url = f"{WEBFLOW_API_BASE}/collections/{collection_id}/items/bulk"
        payload: Dict[str, Any] = {
            "isArchived": False,
            "isDraft": True,
            "fieldData": {
                field_slug_title: title,
                field_slug_slug: slugify(title),
                field_slug_description: description,
                field_slug_article: article_html,
                field_slug_main_image: {"url": main_image_url, **({"alt": main_image_alt} if main_image_alt else {})},
            },
        }

        if thumb_image_url:
            payload["fieldData"][field_slug_thumb_image] = {
                "url": thumb_image_url,
                **({"alt": thumb_image_alt} if thumb_image_alt else {}),
            }

        if author_item_id:
            payload["fieldData"][field_slug_author] = author_item_id

        resp = self.session.post(url, data=json.dumps(payload), timeout=60)
        if resp.status_code not in (200, 201, 202):
            raise RuntimeError(f"Create item failed: {resp.status_code} {resp.text}")
        return resp.json()


PUBLISH_TEMPLATE = """
You are the publishing QA agent for Career Compass.
Review the blog content and summarize publishing metadata.
Return JSON with keys: title, slug, summary, tags (list), excerpt, and content_ready (boolean).
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

        title = metadata.get("title") or metadata.get("suggested_title") or "Career Compass Blog"
        description = metadata.get("summary") or metadata.get("excerpt") or ""
        article_html = metadata.get("html") or blog_markdown
        author_id = self.settings.webflow_author_id

        try:
            response = cms.create_draft_blog_item(
                self.settings.webflow_collection_id,
                title=title,
                description=description,
                article_html=article_html,
                author_item_id=author_id,
                main_image_url=hosted_image_url,
                thumb_image_url=thumb_image_url or hosted_image_url,
                main_image_alt=metadata.get("image_alt", ""),
                thumb_image_alt=metadata.get("thumbnail_alt", ""),
            )
        except Exception as exc:  # pragma: no cover - network call
            return {"status": "error", "reason": f"Webflow CMS create failed: {exc}"}

        return {"status": "success", "webflow_item": response}
