"""Directory scanning for image and contour files."""
from __future__ import annotations

import os

from core.models import IMAGE_EXTENSIONS


def scan_directory(folder: str) -> tuple[dict[str, str], dict[str, str]]:
    """Scan *folder* for images and contour JSONs.

    Returns ``(image_files, json_files)``.  Both dicts are keyed by the file
    name **without extension** (in lower case), e.g. ``"1_1"`` maps to the
    actual on-disk name ``"1_1.jpg"`` and ``"1_1_contour"`` maps to
    ``"1_1_contour.JSON"``.  Values keep the original case so callers can
    build real paths on case-sensitive filesystems; lookups therefore use the
    bare base name they expect, e.g. ``json_files.get("1_1_contour")``.
    """
    images: dict[str, str] = {}
    jsons: dict[str, str] = {}
    if not os.path.isdir(folder):
        return images, jsons
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return images, jsons
    for name in names:
        lowered = name.lower()
        if lowered.endswith(IMAGE_EXTENSIONS):
            images[os.path.splitext(lowered)[0]] = name
        elif lowered.endswith(".json"):
            jsons[os.path.splitext(lowered)[0]] = name
    return images, jsons
