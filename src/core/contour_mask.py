"""Построение маски контура поверх обрезанной области изображения.

Общий код, ранее продублированный в четырёх местах:
``core.seed_geometry.compute_median_rgb``, ``core.seed_geometry.rgb_averaged``,
``core.seed_geometry.groove_score`` и ``core.segmentation.object_contrast``.
"""
from __future__ import annotations

import cv2
import numpy as np


def contour_mask(
    img_shape: tuple[int, ...],
    pts: np.ndarray,
    pad: int = 0,
    bbox: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int, np.ndarray] | None:
    """Построить маску залитого контура над областью изображения.

    Параметры:
        img_shape — форма исходного изображения (H, W, ...): ограничивает
            вычисляемую область краями кадра;
        pts — точки контура в формате (N, 2) или (N, 1, 2);
        pad — отступ вокруг ограничивающего прямоугольника контура, px;
        bbox — готовый прямоугольник (x, y, w, h); если не задан (None),
            вычисляется из pts через ``cv2.boundingRect``.

    Возвращает кортеж ``(x0, y0, x1, y1, mask)``: область в координатах
    изображения и uint8-маску размера ``(y1 - y0, x1 - x0)`` с залитым
    (``cv2.FILLED``) контуром, сдвинутым в начало области.

    Возвращает ``None``, если область получилась пустой
    (``x1 <= x0`` или ``y1 <= y0``) — вызывающий код должен обработать это
    так же, как и раньше (выход с «пустым» результатом).
    """
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
