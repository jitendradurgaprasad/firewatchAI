"""Bounded optional NASA FIRMS Area API client; secrets stay in environment."""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone


API_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
PRODUCTS = ("VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "VIIRS_SNPP_NRT", "MODIS_NRT")
MAX_RESPONSE_BYTES = 50 * 1024 * 1024


class FirmsAPIError(RuntimeError):
    """Safe-to-display acquisition error with no credential material."""


@dataclass(frozen=True)
class FirmsDownload:
    content: bytes
    source: str
    product: str
    retrieved_at: str
    area: str
    day_range: int
    filename: str


def _validated_area(area: str) -> str:
    try:
        parts = [float(value.strip()) for value in area.split(",")]
    except (AttributeError, ValueError) as exc:
        raise ValueError("Area must be west,south,east,north numeric coordinates.") from exc
    if len(parts) != 4:
        raise ValueError("Area must contain west,south,east,north.")
    west, south, east, north = parts
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Area bounds must be ordered and inside longitude/latitude limits.")
    return ",".join(f"{value:g}" for value in parts)


def fetch_recent_area(*, area: str, product: str = "VIIRS_NOAA20_NRT", day_range: int = 1,
                      map_key: str | None = None, timeout_seconds: int = 30,
                      max_response_bytes: int = MAX_RESPONSE_BYTES) -> FirmsDownload:
    """Download a recent FIRMS CSV for a bounded area (1–5 days, ≤50 MiB)."""
    key = (map_key if map_key is not None else os.environ.get("FIRMS_MAP_KEY", "")).strip()
    if not key:
        raise FirmsAPIError("FIRMS_MAP_KEY is not configured. Add the key locally or continue with the bundled sample/upload.")
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", key):
        raise FirmsAPIError("FIRMS_MAP_KEY has an unexpected format; verify it in your local environment.")
    if product not in PRODUCTS:
        raise ValueError("Unsupported FIRMS product selection.")
    if not isinstance(day_range, int) or not 1 <= day_range <= 5:
        raise ValueError("day_range must be an integer from 1 to 5.")
    if not 1 <= timeout_seconds <= 120 or not 1 <= max_response_bytes <= MAX_RESPONSE_BYTES:
        raise ValueError("Request timeout or response size is outside the allowed limit.")
    area_value = _validated_area(area)
    url = f"{API_BASE}/{key}/{product}/{area_value}/{day_range}"
    request = urllib.request.Request(url, headers={"User-Agent": "FireWatchAI/1.0 (local research dashboard)"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > max_response_bytes:
                raise FirmsAPIError("FIRMS response exceeds the configured 50 MiB download limit. Narrow the area or days.")
            content = response.read(max_response_bytes + 1)
    except FirmsAPIError:
        raise
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            detail = "FIRMS rejected the MAP_KEY. Check the key and its transaction quota."
        elif exc.code == 429:
            detail = "FIRMS rate or transaction limit was reached. Wait before retrying."
        else:
            detail = f"FIRMS request failed with HTTP status {exc.code}."
        raise FirmsAPIError(detail) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise FirmsAPIError("Could not reach NASA FIRMS. Check network access and try again later.") from None
    if len(content) > max_response_bytes:
        raise FirmsAPIError("FIRMS response exceeds the configured 50 MiB download limit. Narrow the area or days.")
    header = content[:4096].decode("utf-8-sig", errors="replace").lower()
    if not content.strip() or "latitude" not in header or "longitude" not in header:
        raise FirmsAPIError("FIRMS returned an empty response or an error page instead of a CSV. Verify product, area, and key.")
    return FirmsDownload(content=content, source="NASA FIRMS Area API (recent observations)",
                         product=product, retrieved_at=datetime.now(timezone.utc).isoformat(),
                         area=area_value, day_range=day_range,
                         filename=f"firms_{product.lower()}_{day_range}d.csv")
