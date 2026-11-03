"""Export seed crops, contours, and CSV rows."""
from __future__ import annotations

import os
import re
from typing import Any

import cv2

from core.contour_io import save_contour_json
from core.csv_utils import update_seeds_csv
from core.io_utils import fmt_float, imwrite_unicode, sanitize_filename
from core.models import ExportSummary, SeedItem
from core.seed_geometry import compute_median_color_hex, groove_presence


def make_source_name(image_path: str) -> str:
    """Build a safe group name from an image path."""
    if image_path:
        return sanitize_filename(os.path.splitext(os.path.basename(image_path))[0])
    return "image"


def select_items_for_export(
    items: list[SeedItem],
    *,
    save_only_focused: bool = False,
    skip_blurry: bool = True,
) -> list[SeedItem]:
    items = list(items or [])
    if save_only_focused:
        return [it for it in items if it.get("status") == "focused"]
    if skip_blurry:
        return [it for it in items if it.get("status") != "blurry"]
    return items


def assign_unique_seed_indices(items: list[SeedItem]) -> list[tuple[SeedItem, int]]:
    def sort_key(item: SeedItem) -> int:
        try:
            return int(item.get("index", 0))
        except Exception:
            return 0

    used: set[int] = set()
    result: list[tuple[SeedItem, int]] = []
    for item in sorted(items, key=sort_key):
        try:
            idx = int(item.get("index", 0))
        except Exception:
            idx = 0
        if idx <= 0:
            idx = max(used) + 1 if used else 1
        while idx in used:
            idx += 1
        used.add(idx)
        result.append((item, idx))
    return result


def compute_crop_box(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
    padding: int = 0,
) -> tuple[int, int, int, int]:
    h, w = image_shape[:2]
    x, y, bw, bh = map(int, bbox)
    pad = max(0, int(padding))
    return max(0, x - pad), max(0, y - pad), min(int(w), x + bw + pad), min(int(h), y + bh + pad)


def _px_to_mm(value_px: Any, mm_per_px: float | None, decimals: int = 4) -> str:
    if value_px is None or mm_per_px is None:
        return ""
    try:
        return fmt_float(float(value_px) * float(mm_per_px), decimals)
    except (TypeError, ValueError):
        return ""


def build_seed_csv_row(
    item: SeedItem,
    file_index: str,
    seed_index: int,
    mm_per_px: float | None,
    color_hex: str = "",
    has_groove: int | None = None,
) -> dict[str, Any]:
    length_px = item.get("length_px")
    width_px = item.get("width_px")
    c_hex = color_hex or str(item.get("color_hex") or "")
    return {
        "file_index": file_index,
        "index": seed_index,
        "status": str(item.get("status", "unknown")),
        "color_hex": c_hex,
        "edge_width_px": fmt_float(item.get("edge_width"), 3),
        "length_px": fmt_float(length_px, 3),
        "width_px": fmt_float(width_px, 3),
        "angle_deg": fmt_float(item.get("angle_deg", item.get("angle")), 3),
        "area_px": fmt_float(item.get("area"), 1),
        # Бинарный признак бороздки: 1 — есть, 0 — нет
        "has_groove": "" if has_groove is None else int(has_groove),
        "mm_per_px": fmt_float(mm_per_px, 8) if mm_per_px is not None else "",
    }


def remove_group_files_except(dir_path: str, group_prefix: str, keep_indexes: list[str]) -> None:
    if not group_prefix or not os.path.isdir(dir_path):
        return
    keep = set(str(x) for x in keep_indexes)
    img_pat = re.compile(r"^" + re.escape(group_prefix) + r"_(\d+)\.(jpe?g|png)$", re.IGNORECASE)
    json_pat = re.compile(r"^" + re.escape(group_prefix) + r"_(\d+)_contour\.json$", re.IGNORECASE)
    for name in os.listdir(dir_path):
        file_index: str | None = None
        remove = False
        m = img_pat.match(name)
        if m:
            file_index = f"{group_prefix}_{m.group(1)}"
            remove = m.group(2).lower() != "jpg" or file_index not in keep
        else:
            m = json_pat.match(name)
            if m:
                file_index = f"{group_prefix}_{m.group(1)}"
                remove = file_index not in keep
        if remove:
            try:
                os.remove(os.path.join(dir_path, name))
            except Exception:
                pass


def export_seed_items(
    image_bgr: Any,
    items: list[SeedItem],
    out_dir: str,
    source_name: str,
    *,
    padding: int = 0,
    mm_per_px: float | None = None,
) -> ExportSummary:
    """Export crops + contours + CSV. Returns summary dict."""
    items = list(items or [])
    summary: ExportSummary = {
        "saved": 0, "selected": len(items), "contour_errors": 0,
        "saved_file_indexes": [], "csv_error": None,
    }
    if not items or image_bgr is None:
        return summary
    os.makedirs(out_dir, exist_ok=True)
    pad = max(0, int(padding))
    saved_rows: list[dict[str, Any]] = []

    for item, seed_index in assign_unique_seed_indices(items):
        bbox = item.get("bbox")
        if not bbox:
            continue
        try:
            x1, y1, x2, y2 = compute_crop_box(bbox, image_bgr.shape, pad)
        except Exception:
            continue
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        file_index = f"{source_name}_{seed_index:03d}"
        if not imwrite_unicode(os.path.join(out_dir, f"{file_index}.jpg"), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            continue
        try:
            save_contour_json(
                path=os.path.join(out_dir, f"{file_index}_contour.json"),
                file_index=file_index, seed_index=seed_index,
                contour=item.get("contour"), crop_origin=(x1, y1),
                crop_shape=crop.shape, bbox=bbox, padding=pad, source_name=source_name,
            )
        except Exception:
            summary["contour_errors"] += 1
        summary["saved"] += 1
        summary["saved_file_indexes"].append(file_index)
        color_hex = str(item.get("color_hex") or "")
        if not color_hex and item.get("contour") is not None:
            color_hex = compute_median_color_hex(image_bgr, item.get("contour"))
        # Признак бороздки: берём из уже посчитанных метрик, а если его нет —
        # считаем прямо по фрагменту исходного изображения
        has_groove = item.get("groove")
        if has_groove not in (0, 1) and item.get("contour") is not None:
            has_groove = groove_presence(image_bgr, item.get("contour"))
        saved_rows.append(build_seed_csv_row(
            item, file_index, seed_index, mm_per_px, color_hex=color_hex, has_groove=has_groove,
        ))

    try:
        remove_group_files_except(out_dir, source_name, summary["saved_file_indexes"])
    except Exception:
        pass
    try:
        update_seeds_csv(os.path.join(out_dir, "seeds.csv"), saved_rows, source_name)
    except Exception as exc:
        summary["csv_error"] = str(exc)
    return summary


def format_export_message(saved: int, selected: int, source_name: str, out_dir: str, contour_errors: int = 0) -> str:
    msg = f"Сохранено {saved} из {selected} фрагментов для '{source_name}' в: {out_dir}"
    if contour_errors:
        msg += f", ошибки сохранения контуров: {contour_errors}"
    return msg