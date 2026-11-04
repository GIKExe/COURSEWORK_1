from __future__ import annotations

import math
from typing import Any

from .io_utils import parse_float

type ParamDef = tuple[str, str, float, float, float, Any, str]
type CheckboxDef = tuple[str, str, bool]

_PARAM_DEFS: tuple[ParamDef, ...] = (
    ("blur_kernel", "Размытие, ядро", 1, 31, 2, 21, "odd"),
    ("close_kernel", "Close, морфология", 1, 31, 1, 2, "int"),
    ("open_kernel", "Open, морфология", 1, 31, 1, 5, "int"),
    ("min_area", "Мин. площадь, px", 10, 100000, 10, 3000, "int"),
    ("max_area", "Макс. площадь, px", 100, 300000, 100, 40000, "int"),
    ("min_ratio", "Мин. вытянутость", 1.0, 10.0, 0.1, 1.0, "float"),
    ("max_ratio", "Макс. вытянутость", 1.0, 15.0, 0.1, 4.5, "float"),
    ("min_solidity", "Мин. solidity", 0.0, 1.0, 0.01, 0.60, "float"),
    ("min_extent", "Мин. extent", 0.0, 1.0, 0.01, 0.30, "float"),
    ("min_contrast", "Мин. контраст", 0, 100, 1, 40, "int"),
    ("max_edge_width", "Фокус: граница <=, px", 0.5, 15.0, 0.1, 3.9, "float"),
    ("blur_edge_width", "Размыто: граница >=, px", 1.0, 25.0, 0.1, 6.0, "float"),
    ("padding", "Отступ при сохранении, px", 0, 50, 1, 25, "int"),
)

_CHECKBOX_DEFS: tuple[CheckboxDef, ...] = (
    ("Оценивать ширину границы", "estimate_edge", True),
    ("Оставлять только сфокусированные", "filter_edge", False),
    ("Фильтровать выбросы по площади", "filter_outliers", True),
    ("Не сохранять красные", "skip_blurry", True),
    ("Сохранять только сфокусированные", "save_only_focused", False),
)


def get_param_definitions() -> tuple[ParamDef, ...]:
    return _PARAM_DEFS


def get_checkbox_definitions() -> tuple[CheckboxDef, ...]:
    return _CHECKBOX_DEFS


def get_default_params() -> dict[str, Any]:
    """Return default processing parameters (used by GUI and batch)."""
    params: dict[str, Any] = {}
    for key, _, _, _, _, default, _ in _PARAM_DEFS:
        params[key] = default
    for _, key, default in _CHECKBOX_DEFS:
        params[key] = default
    params["mm_per_px"] = None
    return params


def normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Coerce parameters into a consistent, valid state."""
    merged = {**get_default_params(), **(params or {})}
    merged["blur_kernel"] = ensure_odd(int(float(merged["blur_kernel"])), 1)
    if int(merged["max_area"]) <= int(merged["min_area"]):
        merged["max_area"] = int(merged["min_area"]) + 100
    if float(merged["max_ratio"]) <= float(merged["min_ratio"]):
        merged["max_ratio"] = float(merged["min_ratio"]) + 0.5
    if float(merged["blur_edge_width"]) <= float(merged["max_edge_width"]):
        merged["blur_edge_width"] = float(merged["max_edge_width"]) + 0.1
    return merged


def parse_mm_per_px_text(text: str) -> float | None:
    """Parse mm/px from user input. Empty string → None (pixel measurements)."""
    if not (text or "").strip():
        return None
    value = parse_float(text)
    if value is None:
        raise ValueError("Масштаб мм/px должен быть числом, например 0.025") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Масштаб мм/px должен быть положительным числом")
    return value


def ensure_odd(value: float, min_value: int = 1) -> int:
    """Return the nearest odd integer ≥ *min_value*."""
    v = int(round(float(value)))
    v = max(v, min_value)
    if v % 2 == 0:
        v += 1
    return v


def round_param_value(value: Any, low: float, high: float, step: float, kind: str) -> Any:
    """Round a parameter value to the allowed step and clamp to [low, high]."""
    v = float(value)
    lo, hi, st = float(low), float(high), float(step)
    if kind == "float":
        if st <= 0:
            return max(lo, min(hi, v))
        decimals = 2 if st < 0.1 else 1
        return max(lo, min(hi, round(round(v / st) * st, decimals)))
    low_i, high_i = int(lo), int(hi)
    v = int(round(v)) if st <= 0 else int(round(v / st) * st)
    v = max(low_i, min(high_i, v))
    if kind == "odd" and v % 2 == 0:
        v = v + 1 if v + 1 <= high_i else v - 1
        v = max(low_i, v)
    return v


def format_param_label(title: str, value: Any, kind: str, step: float) -> str:
    """Format a parameter label for GUI display."""
    if kind == "float":
        decimals = 2 if float(step) < 0.1 else 1
        return f"{title}: {float(value):.{decimals}f}"
    return f"{title}: {round(float(value))}"