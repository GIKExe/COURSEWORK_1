"""Seed measurement statistics and status counting."""
from __future__ import annotations

from typing import Any

import numpy as np

from core.models import SeedItem, SizeStats, StatusCounts


def compute_size_stats(items: list[SeedItem], mm_per_px: float | None = None) -> SizeStats | None:
    if not items:
        return None
    k = float(mm_per_px) if mm_per_px else 1.0
    lengths = np.array([it["length_px"] for it in items], dtype=np.float64) * k
    widths = np.array([it["width_px"] for it in items], dtype=np.float64) * k
    return {
        "count": len(items),
        "unit": "мм" if mm_per_px else "px",
        "length": (float(lengths.min()), float(lengths.mean()), float(lengths.max())),
        "width": (float(widths.min()), float(widths.mean()), float(widths.max())),
    }


def format_size_stats(stats: SizeStats | None) -> str:
    if stats is None:
        return "Нет данных"
    unit = stats["unit"]
    dec = 2 if unit == "мм" else 1

    def line(title: str, values: tuple[float, float, float]) -> str:
        mn, avg, mx = values
        return f"{title}, {unit}: мин {mn:.{dec}f} / ср {avg:.{dec}f} / макс {mx:.{dec}f}"

    return f"Семян: {stats['count']}\n{line('Длина', stats['length'])}\n{line('Ширина', stats['width'])}"


def count_seed_statuses(items: list[dict[str, Any]]) -> StatusCounts:
    items = list(items or [])
    return {
        "total": len(items),
        "focused": sum(1 for it in items if it.get("status") == "focused"),
        "blurry": sum(1 for it in items if it.get("status") == "blurry"),
        "unknown": sum(1 for it in items if it.get("status") == "unknown"),
    }


def format_seed_status_message(counts: StatusCounts) -> str:
    return (
        f"Найдено: {counts['total']}; "
        f"фокус: {counts['focused']}; "
        f"размыто: {counts['blurry']}; "
        f"неизвестно: {counts['unknown']}."
    )