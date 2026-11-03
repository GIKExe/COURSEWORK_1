"""Contour JSON I/O and point manipulation."""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from core.io_utils import разобрать_точку, создать_каталог_файла
from core.models import ContourPayload


def contour_to_points(
    contour: np.ndarray | None,
    offset_x: int = 0,
    offset_y: int = 0,
) -> list[list[int]]:
    """Convert an OpenCV contour to [[x, y], ...], applying optional offset."""
    if contour is None:
        return []
    try:
        pts = contour.reshape(-1, 2)
    except Exception:
        return []
    return [[int(p[0]) - offset_x, int(p[1]) - offset_y] for p in pts]


def save_contour_json(
    path: str,
    file_index: str,
    seed_index: int,
    contour: np.ndarray | None,
    crop_origin: tuple[int, int],
    crop_shape: tuple[int, ...],
    bbox: tuple[int, int, int, int] | None = None,
    padding: int = 0,
    source_name: str | None = None,
) -> None:
    """Save seed boundary points as JSON in crop-image coordinates."""
    crop_h, crop_w = crop_shape[:2]
    ox, oy = crop_origin
    payload: ContourPayload = {
        "file_index": file_index,
        "index": int(seed_index),
        "points_space": "crop",
        "image_width": int(crop_w),
        "image_height": int(crop_h),
        "crop_origin_in_source": [int(ox), int(oy)],
        "padding": int(padding),
        "points": contour_to_points(contour, offset_x=ox, offset_y=oy),
    }
    if bbox is not None:
        bx, by, bw, bh = bbox
        payload["bbox_in_source"] = [int(bx), int(by), int(bw), int(bh)]
    if source_name:
        payload["source_name"] = source_name

    создать_каталог_файла(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=None)


def load_contour(path: str) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    """Load contour points from JSON. Returns (Nx2 float32 array, meta dict)."""
    if not os.path.isfile(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            data = {"points": data}
        pts = np.array(data.get("points", []), dtype=np.float32).reshape(-1, 2)
    except Exception:
        return None, None
    if len(pts) < 3:
        return None, None
    return pts, data


def parse_raw_points(raw: list[Any]) -> list[tuple[float, float]]:
    """Parse heterogeneous point formats into [(x, y), ...]."""
    result: list[tuple[float, float]] = []
    if not isinstance(raw, (list, tuple)):
        return result
    for point in raw:
        pt = разобрать_точку(point)
        if pt is not None:
            result.append(pt)
    return result