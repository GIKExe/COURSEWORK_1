from __future__ import annotations

import cv2
import numpy as np


def contour_mask(
    img_shape: tuple[int, ...],
    pts: np.ndarray,
    pad: int = 0,
    bbox: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int, np.ndarray] | None:
    img_h, img_w = int(img_shape[0]), int(img_shape[1])
    if bbox is None:
        x, y, w, h = cv2.boundingRect(np.asarray(pts).reshape(-1, 1, 2))
    else:
        x, y, w, h = (int(v) for v in bbox)
    pad = int(pad)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img_w, x + w + pad), min(img_h, y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    shifted = (
        np.asarray(pts).reshape(-1, 2) - np.array([x0, y0], dtype=np.int32)
    ).reshape(-1, 1, 2)
    cv2.drawContours(mask, [shifted], -1, 255, thickness=cv2.FILLED)
    return x0, y0, x1, y1, mask
