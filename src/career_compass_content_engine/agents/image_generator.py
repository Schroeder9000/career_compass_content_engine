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
        # optional API key can be provided to prefer OpenAI images
        self.openai_api_key: Optional[str] = None

    def _download_url(self, url: str, out_path: Path) -> Path:
        import time

        # Try a few times to download the URL with a small backoff.
        last_exc = None
        for attempt in range(1, 4):
            try:
                print(f"[ImageGenerator] downloading url (attempt {attempt}): {url}")
                resp = requests.get(url, stream=True, timeout=30)
                resp.raise_for_status()
                with out_path.open("wb") as f:
                    for chunk in resp.iter_content(1024 * 32):
                        if chunk:
                            f.write(chunk)
                print(f"[ImageGenerator] download succeeded: {out_path} ({out_path.stat().st_size} bytes)")
                return out_path
            except Exception as exc:
                last_exc = exc
                print(f"[ImageGenerator] download attempt {attempt} failed: {exc}")
                time.sleep(0.5 * attempt)

        # All retries failed — attempt to generate a simple fallback image locally (requires Pillow).
        print(f"[ImageGenerator] all download attempts failed for {url}; creating programmatic fallback image at {out_path}")
        if _PIL_AVAILABLE:
            try:
                from PIL import Image, ImageDraw, ImageFont

                W, H = 1200, 800
                bg_color = (240, 240, 240)
                text_color = (40, 40, 40)
                img = Image.new("RGB", (W, H), color=bg_color)
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None
                # Draw a simple centered message with truncated prompt
                msg = "Image generation failed — placeholder"
                prompt_snip = (url.split("?text=")[-1] if "?text=" in url else "")
                lines = [msg]
                if prompt_snip:
                    import urllib.parse

                    text = urllib.parse.unquote_plus(prompt_snip)[:200]
                    lines.append(text)
                y = 40
                for line in lines:
                    if font:
                        draw.text((40, y), line, fill=text_color, font=font)
                    else:
                        draw.text((40, y), line, fill=text_color)
                    y += 24
                img.save(out_path, format="PNG")
                print(f"[ImageGenerator] wrote fallback PNG: {out_path} ({out_path.stat().st_size} bytes)")
                return out_path
            except Exception as exc:
                print(f"[ImageGenerator] failed to create programmatic fallback image: {exc}")

        # If Pillow isn't available or generation failed, write a tiny 1x1 PNG fallback so flow can continue.
        print(f"[ImageGenerator] writing 1x1 PNG fallback to {out_path}")
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

        # Try OpenAI images via direct HTTP call (prefer base64 response to avoid external downloads)
        api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                print("[ImageGenerator] requesting OpenAI Images API (b64)")
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": "gpt-image-1",
                    "prompt": prompt,
                    "n": 1,
                    "size": size,
                }
                resp = requests.post("https://api.openai.com/v1/images/generations", json=payload, headers=headers, timeout=60)
                if resp.status_code != 200:
                    print(f"[ImageGenerator] OpenAI image API returned {resp.status_code}: {resp.text}")
                    raise RuntimeError("OpenAI image API failed")
                j = resp.json()
                data = j.get("data") or []
                if data and data[0].get("b64_json"):
                    b64 = data[0]["b64_json"]
                    print("[ImageGenerator] received base64 image from OpenAI")
                    raw = base64.b64decode(b64)
                    with out_path.open("wb") as f:
                        f.write(raw)
                    return self._optimize_image(out_path)
                # If API returned a URL instead, try to download it
                if data and data[0].get("url"):
                    url = data[0]["url"]
                    print(f"[ImageGenerator] received image url from OpenAI: {url}")
                    path = self._download_url(url, out_path)
                    return self._optimize_image(path)
            except Exception as exc:
                print("[ImageGenerator] OpenAI image generation failed or returned no usable data; falling back to placeholder")
                try:
                    print(f"[ImageGenerator] OpenAI exception: {repr(exc)}")
                except Exception:
                    pass
                # Fall through to placeholder logic below
                pass

        # Fallback: download placeholder image
        placeholder = f"https://via.placeholder.com/1200x800.png?text={requests.utils.requote_uri(prompt[:40])}"
        print(f"[ImageGenerator] using placeholder image: {placeholder}")
        path = self._download_url(placeholder, out_path)
        return self._optimize_image(path)
