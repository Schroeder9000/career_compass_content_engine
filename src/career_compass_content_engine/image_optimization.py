"""Deterministic web images shared by scheduled publishing and maintenance."""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps, features

MAX_WIDTH = 1600
MAX_BYTES = 1_000_000
VARIANT_WIDTHS = (500, 800, 1080)


@dataclass(frozen=True)
class WebImage:
    path: Path
    width: int
    height: int


def optimize_image(source: Path, *, max_width: int = MAX_WIDTH,
                   max_bytes: int = MAX_BYTES) -> WebImage:
    """Keep composition/alpha, use high quality WebP, and fail closed on errors."""
    if not features.check("webp"):
        raise RuntimeError("Pillow WebP support is required for image publishing")
    source = Path(source)
    with Image.open(source) as raw:
        if getattr(raw, "is_animated", False):
            raise ValueError("Animated images require a separate optimization review")
        raw.load()
        im = ImageOps.exif_transpose(raw)
        icc = im.info.get("icc_profile")
        has_alpha = "A" in im.getbands() or "transparency" in im.info
        im = im.convert("RGBA" if has_alpha else "RGB")
        if im.width > max_width:
            im = im.resize((max_width, round(im.height * max_width / im.width)), Image.Resampling.LANCZOS)
        if max_bytes < 1:
            raise ValueError("Image byte budget must be positive")
        output = source.with_suffix(".webp")
        # Keep an already optimized WebP byte-for-byte when it meets the contract.
        if (raw.format == "WEBP" and raw.getexif().get(274, 1) == 1
                and im.size == raw.size and source.stat().st_size <= max_bytes):
            return WebImage(source, im.width, im.height)
        for quality in (90, 88, 86):
            buffer = BytesIO()
            options = {"icc_profile": icc} if icc else {}
            im.save(buffer, "WEBP", quality=quality, method=6, **options)
            data = buffer.getvalue()
            if len(data) <= max_bytes:
                temp = output.with_suffix(".webp.tmp")
                temp.write_bytes(data)
                temp.replace(output)
                return WebImage(output, im.width, im.height)
    raise ValueError(f"Image exceeds {max_bytes} bytes at the minimum approved quality")


def responsive_images(source: Path) -> list[WebImage]:
    """Create each size from the same master, never from another thumbnail."""
    main = optimize_image(source)
    result = []
    with Image.open(main.path) as im:
        im.load()
        for width in VARIANT_WIDTHS:
            if width >= main.width:
                continue
            variant = main.path.with_name(f"{main.path.stem}-{width}w.webp")
            scaled = im.resize((width, round(main.height * width / main.width)), Image.Resampling.LANCZOS)
            options = {"icc_profile": im.info["icc_profile"]} if im.info.get("icc_profile") else {}
            scaled.save(variant, "WEBP", quality=90, method=6, **options)
            if variant.stat().st_size > MAX_BYTES:
                raise ValueError("Responsive image exceeds the publishing byte budget")
            result.append(WebImage(variant, width, scaled.height))
    return result + [main]


def upload_responsive_images(source: Path, upload) -> dict:
    """Upload through the existing Webflow client and expose a CMS srcset."""
    variants = []
    for image in responsive_images(source):
        hosted = upload(image.path)
        url = (hosted or {}).get("url")
        if not url or not url.startswith("https://"):
            raise ValueError("Responsive image upload did not return a hosted HTTPS URL")
        variants.append({"url": url, "width": image.width, "height": image.height})
    thumbnail = next((v for v in variants if v["width"] >= 800), variants[-1])
    return {"main": {"url": variants[-1]["url"]}, "thumbnail": {"url": thumbnail["url"]},
            "srcset": ", ".join(f"{v['url']} {v['width']}w" for v in variants),
            "width": variants[-1]["width"], "height": variants[-1]["height"],
            "thumbnail_width": thumbnail["width"], "thumbnail_height": thumbnail["height"]}
