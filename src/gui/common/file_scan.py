"""Directory scanning for image and contour files."""
from __future__ import annotations

import os

from core.models import IMAGE_EXTENSIONS


def scan_directory(folder: str) -> tuple[dict[str, str], dict[str, str]]:
    """Scan *folder* for images and contour JSONs.

    Returns (image_files, json_files).  Both dicts are keyed by the
    *lowercase full file name* (extension included), e.g. ``"1_1.jpg"`` and
    ``"1_1_contour.json"``, and map to the **actual on-disk name** so that the
    original case is preserved when building paths (important on case-sensitive
    filesystems).  Lookups therefore use exactly the file name they expect,
    e.g. ``json_files.get("1_1_contour.json")``.
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
            images[lowered] = name
        elif lowered.endswith(".json"):
            jsons[lowered] = name
    return images, jsons
