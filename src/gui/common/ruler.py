from __future__ import annotations

import math
import tkinter as tk

_STEPS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100]


def _choose_step(bar_mm: float, px_per_mm: float) -> float | None:
    for step in _STEPS:
        if step <= 0:
            continue
        if step * px_per_mm >= 10 and bar_mm / step <= 20:
            return step
    return None


def draw_canvas_ruler(
    canvas: tk.Canvas,
    mm_per_px: float | None,
    display_scale: float,
    length_mm: float | None = None,
    width_mm: float | None = None,
    canvas_h: float | None = None,
    tag: str = "overlay",
    color: str = "#FF3333",
) -> None:
    """Draw a vector mm ruler directly onto a tk.Canvas item layer."""
    if not mm_per_px or mm_per_px <= 0 or display_scale <= 0:
        return
    px_per_mm = display_scale / mm_per_px
    if px_per_mm <= 0 or not math.isfinite(px_per_mm):
        return

    h = float(canvas_h) if canvas_h and canvas_h > 0 else float(canvas.winfo_height())
    if h < 80:
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

    x_line, tick_len = 24.0, 14.0
    y0 = max(16.0, (h - bar_px) / 2.0)
    y1 = y0 + bar_px
    if y1 > h - 16.0:
        y1 = h - 16.0
        y0 = max(16.0, y1 - bar_px)

    text_y = max(4.0, y0 - 14.0)
    canvas.create_text(
        8, text_y, anchor="w", text=f"{target_mm:.4g} мм",
        fill=color, font=("Segoe UI", 9, "bold"), tags=(tag,)
    )
    canvas.create_line(x_line, y0, x_line, y1, fill=color, width=3, tags=(tag,))
    canvas.create_line(x_line, y0, x_line + tick_len + 4, y0, fill=color, width=3, tags=(tag,))
    canvas.create_line(x_line, y1, x_line + tick_len + 4, y1, fill=color, width=3, tags=(tag,))

    if step_mm:
        count = int(math.floor(target_mm / step_mm + 1e-6))
        for i in range(1, count + 1):
            mm = i * step_mm
            if mm > target_mm + 1e-9:
                break
            yy = y0 + mm * px_per_mm
            if yy > y1:
                break
            canvas.create_line(x_line, yy, x_line + tick_len, yy, fill=color, width=2, tags=(tag,))
            if px_per_mm * step_mm >= 28 and count <= 12:
                canvas.create_text(
                    x_line + tick_len + 4, yy, anchor="w", text=f"{mm:.4g}",
                    fill=color, font=("Segoe UI", 8), tags=(tag,)
                )
