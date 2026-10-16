"""Small read-only NASA GIBS image viewer client (not the FIRMS API)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


GIBS_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
GIBS_LAYERS = {
    "Suomi-NPP VIIRS true color": "VIIRS_SNPP_CorrectedReflectance_TrueColor",
    "NOAA-20 VIIRS true color": "VIIRS_NOAA20_CorrectedReflectance_TrueColor",
    "NOAA-21 VIIRS true color": "VIIRS_NOAA21_CorrectedReflectance_TrueColor",
}
MAX_IMAGE_BYTES = 12 * 1024 * 1024


def build_gibs_image_url(layer: str, image_date: date, *, width: int = 1200,
                         height: int = 800, bbox: tuple[float, float, float, float] =
                         (60.0, 5.0, 100.0, 38.0)) -> str:
    """Build a bounded, date-specific WMS GetMap URL for an allowlisted layer."""
    if layer not in GIBS_LAYERS.values():
        raise ValueError("Unsupported NASA GIBS imagery layer.")
    if image_date > datetime.now(timezone.utc).date():
        raise ValueError("Image date cannot be in the future.")
    west, south, east, north = map(float, bbox)
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Invalid geographic bounding box.")
    if not (128 <= width <= 2048 and 128 <= height <= 2048):
        raise ValueError("Image dimensions must be between 128 and 2048 pixels.")
    params = {
        "SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap",
        "LAYERS": layer, "STYLES": "", "FORMAT": "image/jpeg",
        "SRS": "EPSG:4326", "BBOX": f"{west},{south},{east},{north}",
        "WIDTH": str(width), "HEIGHT": str(height), "TIME": image_date.isoformat(),
    }
    return f"{GIBS_WMS}?{urlencode(params)}"


def fetch_gibs_image(url: str, *, timeout: float = 20.0) -> tuple[bytes, str]:
    """Fetch only image bytes from NASA's fixed GIBS host with size/content checks."""
    if not url.startswith(GIBS_WMS + "?"):
        raise ValueError("Only the configured NASA GIBS image endpoint is allowed.")
    request = Request(url, headers={"User-Agent": "FireWatchAI-satellite-image-viewer/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get_content_type()
            if content_type not in {"image/jpeg", "image/png"}:
                raise ValueError(f"NASA GIBS returned {content_type}, not an image.")
            image = response.read(MAX_IMAGE_BYTES + 1)
            if len(image) > MAX_IMAGE_BYTES:
                raise ValueError("NASA GIBS image exceeded the 12 MiB safety limit.")
            if not image:
                raise ValueError("NASA GIBS returned an empty image.")
            return image, content_type
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"NASA GIBS image request failed: {exc}") from exc
