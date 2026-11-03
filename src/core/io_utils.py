"""Image I/O with Unicode path support and filename sanitization."""
from __future__ import annotations

import math
import os
import re
from typing import Any

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


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    dst = np.empty_like(img)
    cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)
    return dst.astype(np.uint8)


def to_gray(image: np.ndarray, *, normalize: bool = False) -> np.ndarray:
    if image is None:
        raise ValueError("Изображение не передано")
    if image.ndim == 3:
        ch = image.shape[2]
        if ch == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif ch == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = image[:, :, 0].copy()
    else:
        gray = image.copy()
    if normalize:
        return to_uint8(gray)
    return gray


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


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def разобрать_точку(point: Any) -> tuple[float, float] | None:
    try:
        if isinstance(point, dict):
            x, y = float(point["x"]), float(point["y"])
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = float(point[0]), float(point[1])
        else:
            return None
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if math.isfinite(x) and math.isfinite(y):
        return x, y
    return None


def создать_каталог_файла(path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)