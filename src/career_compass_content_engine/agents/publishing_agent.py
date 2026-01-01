from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional

import requests
from langchain_core.language_models.chat_models import BaseChatModel

from ..config import Settings
from .base import AgentResult, BaseAgent


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

    def upload_to_webflow(self, metadata: Dict[str, str], blog_markdown: str, image_url: Optional[str] = None) -> Dict[str, str]:
        if not self.settings.webflow_api_token or not self.settings.webflow_collection_id:
            return {
                "status": "skipped",
                "reason": "Missing WEBFLOW_API_TOKEN or WEBFLOW_COLLECTION_ID; configure to enable publishing.",
            }

        headers = {
            "Authorization": f"Bearer {self.settings.webflow_api_token}",
            "accept-version": "1.0.0",
            "Content-Type": "application/json",
        }
        payload = {
            "fields": {
                "name": metadata.get("title", "Career Compass Blog"),
                "slug": metadata.get("slug", "career-compass-blog"),
                "_archived": False,
                "_draft": False,
                "post-summary": metadata.get("summary", ""),
                "post-body": blog_markdown,
            }
        }
        if image_url:
            payload["fields"]["main-image"] = image_url

        response = requests.post(
            f"{self.settings.base_url}/collections/{self.settings.webflow_collection_id}/items",
            headers=headers,
            data=json.dumps(payload),
            timeout=30,
        )
        response.raise_for_status()
        return {"status": "success", "webflow_item_id": response.json().get("_id")}
