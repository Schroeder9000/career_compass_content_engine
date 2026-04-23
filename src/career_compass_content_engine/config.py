from __future__ import annotations

import logging
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
    webflow_category_id: Optional[str] = None
    modal_secret_name: Optional[str] = None
    base_url: str = "https://api.webflow.com/v2"
    # Google Search Console — path to a service account JSON key file
    gsc_service_account_json: Optional[str] = None
    # The verified site URL registered in GSC (e.g. "https://yoursite.com/")
    gsc_site_url: Optional[str] = None
    # Google Analytics 4 — same service account key file can be reused
    ga4_property_id: Optional[str] = None
    # Serper.dev API key for real-time SERP data
    serper_api_key: Optional[str] = None
    # OpenAI image generation model (gpt-image-2, gpt-image-1, dall-e-3, dall-e-2)
    openai_image_model: str = "gpt-image-2"
    # LangSmith tracing (optional — set LANGCHAIN_API_KEY to enable)
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "career-compass-content-engine"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_tracing: bool = False
    # Logging level (DEBUG | INFO | WARNING | ERROR)
    log_level: str = "INFO"
    # LangGraph checkpointing — SQLite DB path (set to empty string to disable)
    checkpointer_path: str = ".cache/checkpoints.db"


    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            webflow_api_token=os.environ.get("WEBFLOW_API_TOKEN"),
            webflow_site_id=os.environ.get("WEBFLOW_SITE_ID"),
            webflow_collection_id=os.environ.get("WEBFLOW_COLLECTION_ID"),
            webflow_author_id=os.environ.get("WEBFLOW_AUTHOR_ID"),
            webflow_category_id=os.environ.get("WEBFLOW_CATEGORY_ID"),
            modal_secret_name=os.environ.get("MODAL_SECRET_NAME"),
            gsc_service_account_json=os.environ.get("GSC_SERVICE_ACCOUNT_JSON"),
            gsc_site_url=os.environ.get("GSC_SITE_URL"),
            ga4_property_id=os.environ.get("GA4_PROPERTY_ID"),
            serper_api_key=os.environ.get("SERPER_API_KEY"),
            openai_image_model=(os.environ.get("OPENAI_IMAGE_MODEL") or "gpt-image-2").strip(),
            langsmith_api_key=os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY"),
            langsmith_project=(
                os.environ.get("LANGSMITH_PROJECT")
                or os.environ.get("LANGCHAIN_PROJECT")
                or "career-compass-content-engine"
            ),
            langsmith_endpoint=os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
            langsmith_tracing=(os.environ.get("LANGSMITH_TRACING") or "").strip().lower() in {"1", "true", "yes", "on"},
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            checkpointer_path=os.environ.get("CHECKPOINTER_PATH", ".cache/checkpoints.db"),
        )

    def validate(self) -> None:
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for the content engine to run")

    def configure_tracing(self) -> None:
        """Activate LangSmith tracing when a ``LANGCHAIN_API_KEY`` is present.

        LangChain reads ``LANGCHAIN_TRACING_V2`` and ``LANGCHAIN_API_KEY``
        directly from the environment.  Setting them here (after ``load_dotenv``)
        ensures they are present even when the caller passes the key through
        ``Settings`` rather than a ``.env`` file.
        """
        if not self.langsmith_api_key:
            return
        if not self.langsmith_tracing:
            return
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_ENDPOINT"] = self.langsmith_endpoint
        os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = self.langsmith_project
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project


def get_settings() -> Settings:
    settings = Settings.from_env()
    settings.validate()
    return settings


def configure_logging(level: Optional[str] = None) -> None:
    """Configure structured logging for the content engine.

    Safe to call multiple times — ``basicConfig`` is a no-op once any handler
    is already attached, so calling this from ``Supervisor.__init__`` will
    never override a caller's existing logging configuration.

    Args:
        level: Override the log level (``DEBUG`` | ``INFO`` | ``WARNING`` |
               ``ERROR``).  Defaults to the ``LOG_LEVEL`` env var or ``INFO``.
    """
    effective_level = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, effective_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
