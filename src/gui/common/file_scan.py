"""Directory scanning for image and contour files."""
from __future__ import annotations

import os

from core.models import IMAGE_EXTENSIONS


def scan_directory(folder: str) -> tuple[dict[str, str], dict[str, str]]:
    """Scan *folder* for images and contour JSONs.

    Returns (image_files, json_files) keyed by lowercase name.
    """
    images: dict[str, str] = {}
    jsons: dict[str, str] = {}
    if not os.path.isdir(folder):
        return images, jsons
    try:
        for name in os.listdir(folder):
            name = name.lower()
            if name.endswith(IMAGE_EXTENSIONS):
                images[name.rsplit('.', 1)[0]] = name
            elif name.endswith(".json"):
                jsons[name.rsplit('.', 1)[0]] = name
    except:
        pass
    return images, jsons