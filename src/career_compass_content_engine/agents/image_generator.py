from __future__ import annotations

import os
from pathlib import Path
import requests
import base64
from typing import Optional

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

    def __init__(self, out_dir: str = ".cache/images", max_width: int = 1200, jpeg_quality: int = 85) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_width = max_width
        self.jpeg_quality = jpeg_quality

    def _download_url(self, url: str, out_path: Path) -> Path:
        try:
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            with out_path.open("wb") as f:
                for chunk in resp.iter_content(1024 * 32):
                    if chunk:
                        f.write(chunk)
            return out_path
        except Exception:
            # Network failed; write a tiny 1x1 PNG fallback so flow can continue offline.
            png_1x1 = base64.b64decode(
                b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
            )
            with out_path.open("wb") as f:
                f.write(png_1x1)
            return out_path

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

    def generate(self, prompt: str, out_name: Optional[str] = None, size: str = "1024x1024") -> Path:
        out_name = out_name or f"img_{abs(hash(prompt)) % (10**8)}.png"
        out_path = self.out_dir / out_name

        # Try OpenAI images if available
        if _OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                openai.api_key = os.getenv("OPENAI_API_KEY")
                # Newer openai libs support Images API via openai.Image.create
                resp = openai.Image.create(prompt=prompt, n=1, size=size)
                data = resp.get("data", [])
                if data and data[0].get("url"):
                    url = data[0]["url"]
                    path = self._download_url(url, out_path)
                    return self._optimize_image(path)
                # some SDKs return b64 data
                if data and data[0].get("b64_json"):
                    b64 = data[0]["b64_json"]
                    raw = base64.b64decode(b64)
                    with out_path.open("wb") as f:
                        f.write(raw)
                    return self._optimize_image(out_path)
            except Exception:
                # Fall back to placeholder on any failure
                pass

        # Fallback: download placeholder image
        placeholder = f"https://via.placeholder.com/800x600.png?text={requests.utils.requote_uri(prompt[:40])}"
        path = self._download_url(placeholder, out_path)
        return self._optimize_image(path)
