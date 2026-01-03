from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    """Runtime configuration for the content engine."""

    openai_api_key: str
    webflow_api_token: Optional[str] = None
    webflow_site_id: Optional[str] = None
    webflow_collection_id: Optional[str] = None
    webflow_author_id: Optional[str] = None
    modal_secret_name: Optional[str] = None
    base_url: str = "https://api.webflow.com/v2"


    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            webflow_api_token=os.environ.get("WEBFLOW_API_TOKEN"),
            webflow_site_id=os.environ.get("WEBFLOW_SITE_ID"),
            webflow_collection_id=os.environ.get("WEBFLOW_COLLECTION_ID"),
            webflow_author_id=os.environ.get("WEBFLOW_AUTHOR_ID"),

            modal_secret_name=os.environ.get("MODAL_SECRET_NAME"),
        )

    def validate(self) -> None:
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for the content engine to run")


def get_settings() -> Settings:
    settings = Settings.from_env()
    settings.validate()
    return settings
