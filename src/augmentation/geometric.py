"""Geometric transforms: rotation, flip, background synthesis."""
from __future__ import annotations

import math

import cv2
import numpy as np


def estimate_background_stats(img: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate median background colour and noise std, excluding the seed area."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [np.round(pts).astype(np.int32)], -1, 255, thickness=cv2.FILLED)
    d_k = max(3, min(w, h) // 12)
    if d_k % 2 == 0:
        d_k += 1
    dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_k, d_k)))
    bg_pixels = img[dilated == 0]
    if len(bg_pixels) >= 50:
        bg_median = np.median(bg_pixels, axis=0).astype(np.float32)
        bg_std = np.std(bg_pixels, axis=0).astype(np.float32)
    else:
        border_mask = np.ones((h, w), dtype=bool)
        border_mask[2:-2, 2:-2] = False
        border_pixels = img[border_mask]
        if len(border_pixels) >= 10:
            bg_median = np.median(border_pixels, axis=0).astype(np.float32)
            bg_std = np.std(border_pixels, axis=0).astype(np.float32)
        else:
            bg_median = np.array([33.0, 34.0, 32.0], dtype=np.float32)
            bg_std = np.array([8.0, 8.0, 8.0], dtype=np.float32)
    bg_std = np.maximum(bg_std, 3.5)
    return bg_median, bg_std


def rotate_any_angle(
    img: np.ndarray,
    pts: np.ndarray,
    angle_deg: float,
    flip: bool,
    bg_median: np.ndarray,
    bg_std: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate image + contour by arbitrary angle, fill voids with synthetic background."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rad = math.radians(angle_deg)
    cos_a, sin_a = abs(math.cos(rad)), abs(math.sin(rad))
    new_w, new_h = int(math.ceil(h * sin_a + w * cos_a)), int(math.ceil(h * cos_a + w * sin_a))
    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy

    pts_hom = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
    p_rot = (M @ pts_hom.T).T.astype(np.float32)

    bg_noise = rng.normal(bg_median, bg_std, (new_h, new_w, 3))
    bg_canvas = np.clip(bg_noise, 0, 255).astype(np.float32)

    src_mask = np.ones((h, w), dtype=np.float32)
    warped_mask = cv2.warpAffine(src_mask, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    warped_img = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    alpha = np.clip(warped_mask[:, :, None], 0.0, 1.0)
    out = np.clip(warped_img * alpha + bg_canvas * (1.0 - alpha), 0, 255).astype(np.uint8)

    A = M[:2, :2].copy()
    if flip:
        out = out[:, ::-1]
        p_rot[:, 0] = (new_w - 1) - p_rot[:, 0]
        A = np.diag([-1.0, 1.0]) @ A
    return np.ascontiguousarray(out), p_rot, A


def transform_angle(angle_deg: float, A: np.ndarray) -> float:
    """Compute seed long-axis angle after linear transform A (result in 0..180)."""
    a = math.radians(angle_deg)
    d = A @ np.array([math.cos(a), math.sin(a)])
    return math.degrees(math.atan2(d[1], d[0])) % 180.0