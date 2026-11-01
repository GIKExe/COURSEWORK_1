"""Image segmentation pipeline: mask creation, filtering, edge estimation."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from core.models import ProcessResult, SeedItem, SeedStatus, STATUS_COLORS
from core.params import ensure_odd


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    dst = np.empty_like(img)
    cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)
    return dst.astype(np.uint8)


def to_gray_uint8(image: np.ndarray | None) -> np.ndarray:
    """Convert BGR/BGRA/gray image to grayscale uint8."""
    if image is None:
        raise ValueError("Изображение не передано")
    if image.ndim == 3:
        ch = image.shape[2]
        if ch == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif ch == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = image[:, :, 0].copy()
    else:
        gray = image.copy()
    return to_uint8(gray)


def _morph(mask: np.ndarray, kernel_size: int, op: int, iterations: int = 1) -> np.ndarray:
    if kernel_size <= 1:
        return mask
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, op, kern, iterations=iterations)


def make_mask(gray: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Create a binary seed mask using Otsu thresholding + morphology."""
    k = ensure_odd(params["blur_kernel"], 1)
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_count = int(np.count_nonzero(mask))
    if 0 < white_count < mask.size:
        white_mask = mask > 0
        if float(np.mean(gray[white_mask])) < float(np.mean(gray[~white_mask])) or white_count > mask.size * 0.6:
            mask = cv2.bitwise_not(mask)
    elif white_count == mask.size:
        mask = np.zeros_like(mask)

    close_k = max(1, int(params["close_kernel"]))
    open_k = max(1, int(params["open_kernel"]))
    if close_k > 1:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
    if open_k > 1:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
    return mask


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    keep = np.zeros(num, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(1, int(min_area))
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def fill_external_contours(mask: np.ndarray, min_area: int) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)
    return filled


def object_contrast(gray: np.ndarray, contour: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    x, y, w, h = bbox
    img_h, img_w = gray.shape[:2]
    pad = max(6, min(20, int(min(w, h) * 0.25)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img_w, x + w + pad), min(img_h, y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    roi = gray[y0:y1, x0:x1]
    m = np.zeros(roi.shape, dtype=np.uint8)
    shifted = contour - np.array([x0, y0], dtype=np.int32)
    cv2.drawContours(m, [shifted.astype(np.int32)], -1, 255, thickness=cv2.FILLED)
    if cv2.countNonZero(m) == 0:
        return 0.0
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inner = cv2.erode(m, kern, iterations=1)
    if cv2.countNonZero(inner) < 3:
        inner = m
    outer = cv2.dilate(m, kern, iterations=4)
    band = cv2.subtract(outer, m)
    if cv2.countNonZero(band) < 3:
        band = cv2.bitwise_not(m)
    return abs(float(cv2.mean(roi, mask=inner)[0]) - float(cv2.mean(roi, mask=band)[0]))


def filter_candidates_by_outliers(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(candidates) < 6:
        return candidates
    areas = np.array([c["area"] for c in candidates], dtype=np.float32)
    dims = np.array([min(c["bbox"][2], c["bbox"][3]) for c in candidates], dtype=np.float32)
    med_a, med_d = float(np.median(areas)), float(np.median(dims))
    if med_a <= 0 or med_d <= 0:
        return candidates
    return [
        c for c in candidates
        if med_a * 0.35 <= c["area"] <= med_a * 3.0
        and med_d * 0.45 <= min(c["bbox"][2], c["bbox"][3]) <= med_d * 3.0
    ]


def find_descending_crossing(x: np.ndarray, y: np.ndarray, level: float) -> float | None:
    if len(x) < 2:
        return None
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
    for i in range(len(x) - 1):
        y0, y1 = y[i], y[i + 1]
        if (y0 - level) * (y1 - level) <= 0.0 and y0 >= y1:
            if abs(y1 - y0) < 1e-6:
                return float(x[i])
            return float(x[i] + (level - y0) * (x[i + 1] - x[i]) / (y1 - y0))
    if np.any(y >= level) and np.any(y < level):
        return float(x[int(np.argmin(np.abs(y - level)))])
    return None


def estimate_edge_width(gray: np.ndarray, contour: np.ndarray, bbox: tuple[int, int, int, int], max_points: int = 160) -> float | None:
    x, y, w, h = bbox
    if w < 4 or h < 4:
        return None
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 12:
        return None
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points, endpoint=False).astype(np.int32)
        pts = pts[idx]

    prev, nxt = np.roll(pts, -1, axis=0), np.roll(pts, 1, axis=0)
    tangents = prev - nxt
    norms = np.linalg.norm(tangents, axis=1)
    valid = norms > 1e-3
    if int(valid.sum()) < 12:
        return None
    pts, tangents = pts[valid], tangents[valid] / norms[valid, None]
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))

    img_h, img_w = gray.shape[:2]
    min_dim = max(3, min(w, h))
    check_offset = float(np.clip(min_dim * 0.20, 2.0, 5.0))

    def mean_at(offset: float) -> float:
        coords = pts + normals * offset
        xi = np.rint(coords[:, 0]).astype(np.int32)
        yi = np.rint(coords[:, 1]).astype(np.int32)
        ok = (xi >= 0) & (xi < img_w) & (yi >= 0) & (yi < img_h)
        if not ok.any():
            return np.nan
        return float(np.mean(gray[yi[ok], xi[ok]]))

    plus, minus = mean_at(check_offset), mean_at(-check_offset)
    if np.isnan(plus) or np.isnan(minus):
        return None
    outward = 1.0 if plus < minus else -1.0

    inside_start = -int(np.clip(min_dim * 0.15, 2, 5))
    max_out = int(np.clip(min_dim * 0.90, 8, 18))
    distances = np.arange(inside_start, max_out + 1, 1, dtype=np.float32)
    profile = np.array([mean_at(outward * d) for d in distances], dtype=np.float32)
    good = ~np.isnan(profile)
    if int(good.sum()) < max(6, len(distances) // 2):
        return None
    profile = np.interp(distances, distances[good], profile[good])

    if len(profile) >= 5:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        padded = np.pad(profile, 2, mode="edge")
        profile_smooth = np.convolve(padded, kernel, mode="valid")
    else:
        profile_smooth = profile

    inside_region = profile_smooth[distances <= (inside_start + 1)]
    outside_region = profile_smooth[distances >= (max_out - 2)]
    if len(inside_region) == 0 or len(outside_region) == 0:
        return None
    inside_val, outside_val = float(np.mean(inside_region)), float(np.mean(outside_region))
    contrast = inside_val - outside_val
    if abs(contrast) < 5.0:
        return None

    norm_profile = (profile_smooth - outside_val) / contrast if contrast > 0 else (outside_val - profile_smooth) / (-contrast)
    norm_profile = np.clip(norm_profile, -0.5, 1.5)
    d80 = find_descending_crossing(distances, norm_profile, 0.80)
    d20 = find_descending_crossing(distances, norm_profile, 0.20)
    if d80 is None or d20 is None:
        return None
    width = abs(float(d20 - d80))
    if width < 0.3 or width > float(max_out - inside_start):
        return None
    return width


def classify_edge(edge_width: float | None, params: dict[str, Any]) -> str:
    if edge_width is None:
        return SeedStatus.UNKNOWN
    if edge_width <= float(params["max_edge_width"]):
        return SeedStatus.FOCUSED
    if edge_width >= float(params["blur_edge_width"]):
        return SeedStatus.BLURRY
    return SeedStatus.TRANSITION


def sort_items_reading_order(items: list[SeedItem], row_height_ratio: float = 0.55) -> list[SeedItem]:
    if len(items) <= 1:
        return items
    decorated = []
    heights = []
    for item in items:
        x, y, w, h = item["bbox"]
        decorated.append((float(y) + float(h) * 0.5, float(x) + float(w) * 0.5, item))
        heights.append(float(h))
    decorated.sort(key=lambda t: t[0])
    med_h = float(np.median(np.array(heights, dtype=np.float32)))
    row_tol = max(3.0, med_h * row_height_ratio)

    rows: list[list[tuple[float, float, SeedItem]]] = [[decorated[0]]]
    prev_cy = decorated[0][0]
    for entry in decorated[1:]:
        if entry[0] - prev_cy > row_tol:
            rows.append([])
        rows[-1].append(entry)
        prev_cy = entry[0]

    result: list[SeedItem] = []
    for row in rows:
        row.sort(key=lambda t: (t[1], t[0]))
        result.extend(item for _, _, item in row)
    return result


def process_image(img_bgr: np.ndarray, params: dict[str, Any]) -> ProcessResult:
    """Full segmentation pipeline: mask → contours → filter → measure → annotate."""
    gray = to_gray_uint8(img_bgr)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    mask = make_mask(gray, params)

    min_area = max(1, int(params["min_area"]))
    max_area = max(min_area + 1, int(params["max_area"]))
    mask = remove_small_components(mask, max(20, min_area // 4))
    mask = fill_external_contours(mask, max(20, min_area // 4))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_ratio = max(1.0, float(params["min_ratio"]))
    max_ratio = max(min_ratio, float(params["max_ratio"]))
    min_solidity = max(0.0, min(1.0, float(params["min_solidity"])))
    min_extent = max(0.0, min(1.0, float(params["min_extent"])))
    min_contrast = max(0.0, float(params["min_contrast"]))

    p = dict(params)
    if float(p["blur_edge_width"]) <= float(p["max_edge_width"]):
        p["blur_edge_width"] = float(p["max_edge_width"]) + 0.1

    estimate_edge = bool(p["estimate_edge"]) or bool(p["filter_edge"])
    filter_edge = bool(p["filter_edge"])
    filter_outliers = bool(p.get("filter_outliers", True))
    img_h, img_w = gray.shape[:2]
    MIN_COMPACTNESS = 0.08

    raw: list[dict[str, Any]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 5 or bh < 5:
            continue
        rect_area = bw * bh
        if area <= 0:
            area = rect_area
        if area < min_area or area > max_area:
            continue
        if bw > img_w * 0.9 or bh > img_h * 0.9:
            continue
        ratio = float(max(bw, bh)) / float(max(1, min(bw, bh)))
        if ratio < min_ratio or ratio > max_ratio:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = area / hull_area if hull_area > 0 else 0.0
        extent = area / float(rect_area)
        if solidity < min_solidity or extent < min_extent:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        compactness = (4.0 * np.pi * area) / (perimeter * perimeter)
        if compactness < MIN_COMPACTNESS:
            continue
        contrast = object_contrast(gray, cnt, (x, y, bw, bh))
        if contrast < min_contrast:
            continue
        raw.append({
            "bbox": (x, y, bw, bh), "area": float(area), "contour": cnt,
            "solidity": float(solidity), "extent": float(extent),
            "contrast": float(contrast), "compactness": float(compactness),
        })

    if filter_outliers:
        raw = filter_candidates_by_outliers(raw)

    items: list[SeedItem] = []
    for cand in raw:
        ew = estimate_edge_width(gray, cand["contour"], cand["bbox"]) if estimate_edge else None
        status = classify_edge(ew, p)
        if filter_edge and status != SeedStatus.FOCUSED:
            continue
        rect = cv2.minAreaRect(cand["contour"])
        box = cv2.boxPoints(rect)
        side_a, side_b = rect[1]
        edge1, edge2 = box[1] - box[0], box[2] - box[1]
        len1, len2 = float(np.linalg.norm(edge1)), float(np.linalg.norm(edge2))
        major = edge1 if len1 >= len2 else edge2
        angle = float(np.degrees(np.arctan2(major[1], major[0]))) % 180.0 if np.linalg.norm(major) > 1e-6 else 0.0

        item: SeedItem = dict(cand)
        item.update(
            rect=rect, box=box, center_x=float(rect[0][0]), center_y=float(rect[0][1]),
            length_px=float(max(side_a, side_b)), width_px=float(min(side_a, side_b)),
            angle_deg=angle, edge_width=ew, status=status,
        )
        items.append(item)

    items = sort_items_reading_order(items)
    for i, item in enumerate(items, 1):
        item["index"] = i

    annotated = img_bgr.copy()
    mm_per_px = params.get("mm_per_px")
    for item in items:
        x, y, bw, bh = item["bbox"]
        color = STATUS_COLORS.get(SeedStatus(item["status"]), (255, 255, 255))
        cv2.drawContours(annotated, [item["contour"]], -1, color, 1, lineType=cv2.LINE_AA)
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
        cv2.polylines(annotated, [cv2.boxPoints(item["rect"]).astype(np.int32)], True, (255, 0, 255), 1, cv2.LINE_AA)
        label = f"{item['index']}"
        if item["edge_width"] is not None:
            label += f" {item['edge_width']:.1f}px"
        if mm_per_px:
            label += f" | {item['length_px'] * mm_per_px:.2f}x{item['width_px'] * mm_per_px:.2f}mm"
        else:
            label += f" | {item['length_px']:.0f}x{item['width_px']:.0f}px"
        ty = y - 6 if y - 20 > 0 else y + bh + 16
        cv2.putText(annotated, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return {"items": items, "annotated": annotated, "mask": mask}