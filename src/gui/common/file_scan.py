from __future__ import annotations

import os

from core.models import IMAGE_EXTENSIONS


def scan_directory(folder: str) -> tuple[dict[str, str], dict[str, str]]:
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
