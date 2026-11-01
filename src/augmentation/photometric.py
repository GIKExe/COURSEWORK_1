"""Photometric augmentations: brightness, contrast, gamma, noise, blur, CLAHE."""
from __future__ import annotations

import math

import cv2
import numpy as np

ALPHA_RANGE = (0.85, 1.15)
BETA_RANGE = (-8, 8)
GAMMA_RANGE = (0.85, 1.15)
WB_RANGE = 0.04
HUE_SHIFT_RANGE = (-4, 4)
SAT_RANGE = (0.90, 1.10)
GRADIENT_MAX = 0.06
GAUSS_NOISE_MAX = 2.5
ISO_NOISE_PROB = 0.3
ISO_NOISE_SCALE = (0.005, 0.02)
BLUR_PROB = 0.25
BLUR_KSIZE_RANGE = (3, 5)
DEFOCUS_PROB = 0.15
DEFOCUS_RADIUS_RANGE = (1, 2)
MOTION_BLUR_PROB = 0.15
MOTION_BLUR_KSIZE = (3, 7)
MOTION_BLUR_ANGLE_RANGE = (0, 180)
CLAHE_PROB = 0.20
CLAHE_CLIP_RANGE = (1.0, 3.0)
CLAHE_GRID = (4, 4)
HISTEQ_PROB = 0.10
JPEG_QUALITY = (85, 95)


def _make_disk_kernel(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = (x * x + y * y <= radius * radius).astype(np.float32)
    total = mask.sum()
    if total > 0:
        mask /= total
    return mask


def _make_motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((length // 2, length // 2), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (length, length), flags=cv2.INTER_LINEAR)
    total = kernel.sum()
    if total > 0:
        kernel /= total
    return kernel


def _apply_hue_sat(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(*HUE_SHIFT_RANGE)) % 180.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(*SAT_RANGE), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _apply_clahe(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=rng.uniform(*CLAHE_CLIP_RANGE), tileGridSize=CLAHE_GRID)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _apply_histeq(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a random combination of photometric transforms."""
    x = img.astype(np.float32)
    h, w = x.shape[:2]

    gain = 1.0 + rng.uniform(-WB_RANGE, WB_RANGE, 3).astype(np.float32)
    x = x * rng.uniform(*ALPHA_RANGE) * gain + rng.uniform(*BETA_RANGE)

    g = rng.uniform(*GAMMA_RANGE)
    x = 255.0 * (np.clip(x, 0, 255) / 255.0) ** g

    x = _apply_hue_sat(np.clip(x, 0, 255).astype(np.uint8), rng).astype(np.float32)

    angle = rng.uniform(0, 2 * math.pi)
    strength = rng.uniform(0, GRADIENT_MAX)
    xs = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    ys = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    x = x * (1.0 + strength * (xs * math.cos(angle) + ys * math.sin(angle)))[:, :, None]

    sigma = rng.uniform(0, GAUSS_NOISE_MAX)
    if sigma > 0.1:
        x = x + rng.normal(0, sigma, x.shape).astype(np.float32)
    if rng.random() < ISO_NOISE_PROB:
        iso_scale = rng.uniform(*ISO_NOISE_SCALE)
        x_pos = np.clip(x, 0.1, 255)
        x = x + rng.normal(0, 1, x.shape).astype(np.float32) * iso_scale * np.sqrt(x_pos)

    result = np.clip(x, 0, 255).astype(np.uint8)

    blur_roll = rng.random()
    if blur_roll < BLUR_PROB:
        ksize = int(rng.integers(BLUR_KSIZE_RANGE[0], BLUR_KSIZE_RANGE[1] + 1))
        if ksize % 2 == 0:
            ksize += 1
        result = cv2.GaussianBlur(result, (ksize, ksize), 0)
    elif blur_roll < BLUR_PROB + DEFOCUS_PROB:
        radius = int(rng.integers(DEFOCUS_RADIUS_RANGE[0], DEFOCUS_RADIUS_RANGE[1] + 1))
        result = cv2.filter2D(result, -1, _make_disk_kernel(radius))
    elif blur_roll < BLUR_PROB + DEFOCUS_PROB + MOTION_BLUR_PROB:
        ksize = int(rng.integers(MOTION_BLUR_KSIZE[0], MOTION_BLUR_KSIZE[1] + 1))
        if ksize % 2 == 0:
            ksize += 1
        result = cv2.filter2D(result, -1, _make_motion_kernel(ksize, rng.uniform(*MOTION_BLUR_ANGLE_RANGE)))

    hist_roll = rng.random()
    if hist_roll < CLAHE_PROB:
        result = _apply_clahe(result, rng)
    elif hist_roll < CLAHE_PROB + HISTEQ_PROB:
        result = _apply_histeq(result)
    return result