"""Millimetre ruler drawing on PIL images."""
from __future__ import annotations

import math
from typing import Any

from PIL import Image, ImageDraw, ImageFont

_STEPS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100]
_RED = (255, 0, 0)


def _get_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont | None:
    for name in ("arial.ttf", "DejaVuSans.ttf", "Arial.ttf", "verdana.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        try:
            return ImageFont.load_default()
        except Exception:
            return None


def _choose_step(bar_mm: float, px_per_mm: float) -> float | None:
    for step in _STEPS:
        if step <= 0:
            continue
        if step * px_per_mm >= 10 and bar_mm / step <= 20:
            return step
    return None


def draw_mm_ruler(
    img: Image.Image,
    mm_per_px: float | None,
    display_scale: float,
    length_mm: float | None = None,
    width_mm: float | None = None,
) -> None:
    """Draw a vertical mm ruler on the left side of *img* (mutates in place)."""
    w, h = img.size
    if w < 140 or h < 100:
        return
    if not mm_per_px or mm_per_px <= 0 or display_scale <= 0:
        return
    px_per_mm = display_scale / mm_per_px
    if px_per_mm <= 0 or not math.isfinite(px_per_mm):
        return

    target_mm = length_mm if length_mm and length_mm > 0 else width_mm if width_mm and width_mm > 0 else None
    if not target_mm or target_mm <= 0:
        target_mm = (h * 0.5) / px_per_mm

    max_px, min_px = h * 0.82, 18.0
    bar_px = target_mm * px_per_mm
    if bar_px > max_px:
        bar_px, target_mm = max_px, max_px / px_per_mm
    if bar_px < min_px:
        bar_px, target_mm = min(min_px, max_px), min(min_px, max_px) / px_per_mm
    if bar_px < 5:
        return

    step_mm = _choose_step(target_mm, px_per_mm)
    draw = ImageDraw.Draw(img)
    font = _get_font()

    x_line, tick_len = 24, 14
    y0 = max(16, int((h - bar_px) / 2))
    y1 = y0 + int(bar_px)
    if y1 > h - 16:
        y1 = h - 16
        y0 = max(16, y1 - int(bar_px))

    text_y = max(4, y0 - 22)
    if text_y < 4:
        text_y = min(h - 20, y1 + 6)
    draw.text((8, text_y), f"{target_mm:.4g} мм", fill=_RED, font=font)
    draw.line([(x_line, y0), (x_line, y1)], fill=_RED, width=3)
    draw.line([(x_line, y0), (x_line + tick_len + 4, y0)], fill=_RED, width=3)
    draw.line([(x_line, y1), (x_line + tick_len + 4, y1)], fill=_RED, width=3)

    if step_mm:
        count = int(math.floor(target_mm / step_mm + 1e-6))
        for i in range(1, count + 1):
            mm = i * step_mm
            if mm > target_mm + 1e-9:
                break
            yy = y0 + int(mm * px_per_mm)
            if yy > y1:
                break
            draw.line([(x_line, yy), (x_line + tick_len, yy)], fill=_RED, width=2)
            if px_per_mm * step_mm >= 28 and count <= 12:
                draw.text((x_line + tick_len + 4, yy - 8), f"{mm:.4g}", fill=_RED, font=font)