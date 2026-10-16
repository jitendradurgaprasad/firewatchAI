"""Strict RGB image loading for the supported PNG/JPEG inference path."""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

SUPPORTED_FORMATS = {"JPEG", "PNG"}
MAX_IMAGE_PIXELS = 40_000_000
MIN_IMAGE_SIDE = 32


class ImageInputError(ValueError):
    """Raised when an image is invalid or unsupported."""


@dataclass(frozen=True)
class LoadedImage:
    image: Image.Image
    width: int
    height: int
    format: str


def load_rgb_image(path: str | Path) -> LoadedImage:
    """Load a validated 3-channel JPEG/PNG without silently dropping bands."""
    file_path = Path(path)
    if file_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ImageInputError("Unsupported image extension; use JPEG or PNG.")
    try:
        with Image.open(file_path) as source:
            image_format = source.format
            if image_format not in SUPPORTED_FORMATS:
                raise ImageInputError("Unsupported image encoding; use JPEG or PNG.")
            width, height = source.size
            if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
                raise ImageInputError(f"Image dimensions must be at least {MIN_IMAGE_SIDE}x{MIN_IMAGE_SIDE}.")
            if width * height > MAX_IMAGE_PIXELS:
                raise ImageInputError(f"Image exceeds the {MAX_IMAGE_PIXELS:,}-pixel safety limit.")
            if source.mode != "RGB":
                raise ImageInputError(f"Expected a 3-channel RGB image; found mode {source.mode!r}.")
            source.load()
            image = source.copy()
    except ImageInputError:
        raise
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ImageInputError(f"Could not read image: {exc}") from exc
    return LoadedImage(image=image, width=width, height=height, format=image_format)
