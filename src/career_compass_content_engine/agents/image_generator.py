from __future__ import annotations

import logging
import os
from pathlib import Path
import requests
import base64
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import openai
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False


class ImageGenerator:
    """Wrapper for generating or fetching an image from a prompt.

    Behavior:
    - If `openai` package is available and `OPENAI_API_KEY` is set, attempt to use OpenAI Images API.
    - Otherwise fall back to downloading a placeholder image.
    - Optionally optimize (resize/compress) with Pillow when available.
    """

    def __init__(self, out_dir: str = ".cache/images", max_width: int = 2400, jpeg_quality: int = 90) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_width = max_width
        self.jpeg_quality = jpeg_quality
        # optional API key can be provided to prefer OpenAI images
        self.openai_api_key: Optional[str] = None

    def _download_url(self, url: str, out_path: Path) -> Path:
        import time

        # Try a few times to download the URL with a small backoff.
        last_exc = None
        for attempt in range(1, 4):
            try:
                logger.debug("[ImageGenerator] downloading url (attempt %d): %s", attempt, url)
                resp = requests.get(url, stream=True, timeout=30)
                resp.raise_for_status()
                with out_path.open("wb") as f:
                    for chunk in resp.iter_content(1024 * 32):
                        if chunk:
                            f.write(chunk)
                logger.info("[ImageGenerator] download succeeded: %s (%d bytes)", out_path, out_path.stat().st_size)
                return out_path
            except Exception as exc:
                last_exc = exc
                logger.warning("[ImageGenerator] download attempt %d failed: %s", attempt, exc)
                time.sleep(0.5 * attempt)

        # All retries failed — raise so the caller can handle the failure cleanly
        # rather than silently uploading a broken placeholder to Webflow.
        raise RuntimeError(f"Failed to download image from {url} after 3 attempts: {last_exc}")

    def _optimize_image(self, path: Path) -> Path:
        if not _PIL_AVAILABLE:
            return path
        try:
            with Image.open(path) as im:
                w, h = im.size
                if w > self.max_width:
                    new_h = int(self.max_width * h / w)
                    im = im.resize((self.max_width, new_h), Image.LANCZOS)
                # Convert PNG to optimized PNG or save as JPEG if originally large
                if im.mode in ("RGBA", "LA"):
                    # keep as PNG for transparency
                    im.save(path, optimize=True)
                else:
                    im = im.convert("RGB")
                    im.save(path, format="JPEG", quality=self.jpeg_quality, optimize=True)
        except Exception:
            # If optimization fails, return original
            return path
        return path

    def generate(self, prompt: str, out_name: Optional[str] = None, size: str = "1792x1024") -> Path:
        out_name = out_name or f"img_{abs(hash(prompt)) % (10**8)}.png"
        out_path = self.out_dir / out_name

        # Try OpenAI images via direct HTTP call (prefer base64 response to avoid external downloads)
        api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                logger.info("[ImageGenerator] requesting OpenAI Images API (b64) — size: %s", size)
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": "dall-e-3",
                    "prompt": prompt,
                    "n": 1,
                    "size": size,
                    "response_format": "b64_json",
                }
                resp = requests.post("https://api.openai.com/v1/images/generations", json=payload, headers=headers, timeout=60)
                if resp.status_code != 200:
                    logger.warning("[ImageGenerator] OpenAI image API returned %d: %s", resp.status_code, resp.text)
                    raise RuntimeError("OpenAI image API failed")
                j = resp.json()
                data = j.get("data") or []
                if data and data[0].get("b64_json"):
                    b64 = data[0]["b64_json"]
                    logger.info("[ImageGenerator] received base64 image from OpenAI")
                    raw = base64.b64decode(b64)
                    with out_path.open("wb") as f:
                        f.write(raw)
                    return self._optimize_image(out_path)
                # If API returned a URL instead, try to download it
                if data and data[0].get("url"):
                    url = data[0]["url"]
                    logger.info("[ImageGenerator] received image url from OpenAI: %s", url)
                    path = self._download_url(url, out_path)
                    return self._optimize_image(path)
            except Exception as exc:
                logger.warning("[ImageGenerator] OpenAI image generation failed: %r", exc)
                raise RuntimeError(f"Image generation failed: {exc}") from exc

        raise RuntimeError("Image generation failed: no API key configured and no fallback available.")
