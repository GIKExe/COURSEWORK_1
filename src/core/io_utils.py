"""Image I/O with Unicode path support and filename sanitization."""
from __future__ import annotations

import os
import re

import cv2
import numpy as np


def imread_unicode(path: str) -> np.ndarray | None:
    """Read an image from *path*, supporting Cyrillic characters."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(
    path: str,
    img: np.ndarray,
    params: list[int] | None = None,
) -> bool:
    """Write *img* to *path*, supporting Cyrillic characters."""
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".png"
        path += ext
    if params is None:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 95] if ext.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if ok:
        buf.tofile(path)
    return ok


def sanitize_filename(name: str) -> str:
    """Make *name* safe for use as a file prefix on Windows/Linux."""
    name = os.path.basename(str(name))
    name = re.sub(r'[/:*?"<>|\s]+', "", name)
    name = name.strip(".")
    return name or "image"


def fmt_float(value: float | None, digits: int = 3) -> str:
    """Format a float for CSV; returns empty string for None/NaN/Inf."""
    if value is None:
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""
    return f"{v:.{digits}f}"