"""Contour-based seed geometry and photometric measurements.

All functions here measure the seed *inside its contour* — never the bounding
box — so they work both on full-image OpenCV contours and on user-drawn
contours coming from ``*_contour.json`` files.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import cv2
import numpy as np

from .contour_mask import contour_mask
from .io_utils import разобрать_точку, to_gray


def points_to_contour(points: Iterable[Any] | None) -> np.ndarray | None:
    """Convert a heterogeneous point list to an (N, 1, 2) int32 contour."""
    if points is None:
        return None
    if isinstance(points, np.ndarray):
        arr = points.reshape(-1, 2)
        if len(arr) >= 3:
            return arr.astype(np.int32).reshape(-1, 1, 2)
        return None
    pts: list[tuple[int, int]] = []
    for point in points:
        parsed = разобрать_точку(point)
        if parsed is not None:
            x, y = parsed
            pts.append((int(round(x)), int(round(y))))
    if len(pts) < 3:
        return None
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def simplify_contour(contour: np.ndarray, max_points: int = 480) -> np.ndarray:
    """Decimate long contours so brute-force searches stay fast."""
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(pts) <= max_points:
        return pts
    idx = np.linspace(0, len(pts) - 1, max_points, endpoint=False).astype(np.int64)
    return pts[idx]


def farthest_pair(pts: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return the two mutually most distant contour points (the "diameter")."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return None
    if len(pts) > 24:
        try:
            hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
            if len(hull) >= 2:
                pts = hull
        except cv2.error:
            pass
    if len(pts) > 300:
        idx = np.linspace(0, len(pts) - 1, 300, endpoint=False).astype(np.int64)
        pts = pts[idx]
    dist2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
    i, j = np.unravel_index(int(np.argmax(dist2)), dist2.shape)
    return (float(pts[i][0]), float(pts[i][1])), (float(pts[j][0]), float(pts[j][1]))


def _segment_crossings(a: tuple[float, float], b: tuple[float, float], poly: np.ndarray) -> int:
    """Count proper intersections of segment a-b with polygon edges (strict)."""
    ax, ay = a
    bx, by = b
    rx, ry = bx - ax, by - ay
    eps = 1e-9
    count = 0
    n = len(poly)
    for k in range(n):
        cx, cy = poly[k]
        dx, dy = poly[(k + 1) % n]
        sx, sy = dx - cx, dy - cy
        den = rx * sy - ry * sx
        if abs(den) < eps:
            continue
        qx, qy = cx - ax, cy - ay
        t = (qx * sy - qy * sx) / den
        u = (qx * ry - qy * rx) / den
        if eps < t < 1.0 - eps and eps < u < 1.0 - eps:
            count += 1
    return count


def _chord_is_internal(a: tuple[float, float], b: tuple[float, float], pts: np.ndarray) -> bool:
    """True when the straight chord a-b does not cross the seed contour.

    Combines an exact segment-vs-edge intersection test with a
    point-in-polygon test on samples along the chord: a chord that leaves and
    re-enters the shape (possible for concave outlines) has outside samples.
    """
    if _segment_crossings(a, b, pts) > 0:
        return False
    ax, ay = a
    bx, by = b
    poly = pts.astype(np.float32).reshape(-1, 1, 2)
    for f in (0.5, 0.3, 0.7):
        mx, my = ax + (bx - ax) * f, ay + (by - ay) * f
        try:
            if float(cv2.pointPolygonTest(poly, (float(mx), float(my)), False)) < 0:
                return False
        except cv2.error:
            break
    return True


def length_axis(contour: np.ndarray | None) -> tuple[float, float, float] | None:
    """Seed length axis measured between the two most distant contour points.

    Returns ``(length_px, angle_deg, end_to_end_distance)`` where *angle_deg*
    is the direction of the chord in image coordinates (y grows downwards),
    normalised to [0, 180).  The diameter chord connects the extreme contour
    points; when it would leave the shape (strongly concave outline) the
    longest internal chord found by a local search is used instead, so the
    length line never crosses the contour.
    """
    if contour is None:
        return None
    pts = simplify_contour(np.asarray(contour).reshape(-1, 2))
    if len(pts) < 2:
        return None
    pair = farthest_pair(pts)
    if pair is None:
        return None
    a, b = pair
    dist = float(math.hypot(b[0] - a[0], b[1] - a[1]))
    if dist <= 0:
        return None
    if not _chord_is_internal(a, b, pts):
        # Local search over candidate endpoints near the extremes: keep the
        # longest chord that stays inside the contour.
        cand_a = pts[np.argsort(((pts - np.array(a)) ** 2).sum(axis=1))[:24]]
        cand_b = pts[np.argsort(((pts - np.array(b)) ** 2).sum(axis=1))[:24]]
        best_len = -1.0
        best_pair: tuple[tuple[float, float], tuple[float, float]] | None = None
        for ca in cand_a:
            for cb in cand_b:
                d = float(math.hypot(float(cb[0] - ca[0]), float(cb[1] - ca[1])))
                if d <= best_len or d < dist * 0.5:
                    continue
                pa, pb = (float(ca[0]), float(ca[1])), (float(cb[0]), float(cb[1]))
                if _chord_is_internal(pa, pb, pts):
                    best_len, best_pair = d, (pa, pb)
        if best_pair is not None and best_len > 0:
            a, b = best_pair
            dist = best_len
    angle = float(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))) % 180.0
    return dist, angle, dist


def compute_median_rgb(
    image: np.ndarray | Any,
    contour: Any,
    *,
    is_rgb: bool = False,
) -> tuple[int, int, int] | None:
    """Compute median R, G, B of pixels inside the contour.

    Robust to specular reflections, shadows and noise.
    Accepts cv2 BGR/RGB ndarray or PIL Image, and contour as (N, 1, 2) / (N, 2) ndarray
    or list of points/dicts.
    """
    if image is None or contour is None:
        return None
    if hasattr(image, "convert") and not isinstance(image, np.ndarray):
        arr = np.asarray(image.convert("RGB"))
        is_rgb = True
    else:
        arr = np.asarray(image)
        if arr.ndim < 3 or arr.shape[2] < 3:
            return None

    pts = points_to_contour(contour)
    if pts is None:
        return None
    pts = pts.reshape(-1, 2)
    if len(pts) < 3:
        return None

    region = contour_mask(arr.shape, pts)
    if region is None:
        return None
    x0, y0, x1, y1, mask = region
    crop = arr[y0:y1, x0:x1]

    valid_pixels = crop[mask > 0]
    if valid_pixels.size == 0 or len(valid_pixels) == 0:
        return None

    med = np.median(valid_pixels[:, :3], axis=0)
    if is_rgb:
        r, g, b = int(round(med[0])), int(round(med[1])), int(round(med[2]))
    else:
        b, g, r = int(round(med[0])), int(round(med[1])), int(round(med[2]))

    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return r, g, b


def compute_median_color_hex(
    image: np.ndarray | Any,
    contour: Any,
    *,
    is_rgb: bool = False,
) -> str:
    """Compute median color of seed inside contour and format as uppercase '#RRGGBB'."""
    rgb = compute_median_rgb(image, contour, is_rgb=is_rgb)
    if rgb is None:
        return ""
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def rgb_averaged(image_bgr: np.ndarray, contour: np.ndarray | None) -> tuple[float, float, float] | None:
    """Mean R, G, B over the pixels strictly *inside* the contour."""
    if image_bgr is None or contour is None:
        return None
    pts = np.asarray(contour, dtype=np.int32).reshape(-1, 2)
    if len(pts) < 3:
        return None
    region = contour_mask(image_bgr.shape, pts)
    if region is None:
        return None
    x0, y0, x1, y1, mask = region
    crop = image_bgr[y0:y1, x0:x1]
    if cv2.countNonZero(mask) == 0:
        return None
    mean = cv2.mean(crop, mask=mask)  # BGR(A) order
    return float(mean[2]), float(mean[1]), float(mean[0])


# --- Детектор бороздки -------------------------------------------------------
# Порог бинаризации признака "has_groove". Подобран на размеченных снимках
# images/2_1.jpg (без бороздки) и images/2_1p.jpg (с бороздкой): на полосе
# значений 10...13 точность обеих групп держится выше 92%.
GROOVE_SCORE_THRESHOLD = 11.0

# Поперёчная геометрия анализа (в долях полуширины зерна в данной колонке):
_GROOVE_ZONE = 0.60    # зона анализа: края зерна отсекаем — они тёмные из-за
                       # трёхмерного затенения и дали бы ложный "гребень"
_GROOVE_INNER = 0.35   # центральная полоса, в которой ищется минимум профиля
_GROOVE_OUTER = 0.42   # отсюда берётся уровень соседних участков для сравнения


def _axis_angle(mask: np.ndarray, angle_hint: float | None) -> float | None:
    """Направление длинной оси зерна в градусах: подсказка или PCA по маске."""
    if angle_hint is not None:
        try:
            value = float(angle_hint)
        except (TypeError, ValueError):
            value = math.nan
        if math.isfinite(value):
            return value % 180.0
    ys, xs = np.where(mask > 0)
    if xs.size < 50:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    cov_xx = float(np.var(xs))
    cov_yy = float(np.var(ys))
    cov_xy = float(np.mean((xs - cx) * (ys - cy)))
    return math.degrees(0.5 * math.atan2(2.0 * cov_xy, cov_xx - cov_yy))


def _rotate_to_axis(
    gray: np.ndarray, mask: np.ndarray, angle_deg: float, center: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Развернуть изображение и маску так, чтобы длинная ось стала горизонтальной.

    Холст выбирается с запасом по диагонали, чтобы при повороте ничего
    не обрезалось. Если после поворота зерно оказалось вертикальным —
    доворачиваем ещё на 90°.
    """
    side = int(math.hypot(mask.shape[1], mask.shape[0])) + 8
    if side < 16:
        return None
    cxy = side // 2
    ox, oy = cxy - int(round(center[0])), cxy - int(round(center[1]))
    if ox < 0 or oy < 0 or ox + gray.shape[1] > side or oy + gray.shape[0] > side:
        return None
    big_gray = np.zeros((side, side), dtype=np.uint8)
    big_mask = np.zeros((side, side), dtype=np.uint8)
    big_gray[oy:oy + gray.shape[0], ox:ox + gray.shape[1]] = gray
    big_mask[oy:oy + mask.shape[0], ox:ox + mask.shape[1]] = mask

    for ang in (angle_deg, angle_deg + 90.0):
        M = cv2.getRotationMatrix2D((float(cxy), float(cxy)), float(ang), 1.0)
        g = cv2.warpAffine(big_gray, M, (side, side), flags=cv2.INTER_LINEAR, borderValue=0)
        m = cv2.warpAffine(big_mask, M, (side, side), flags=cv2.INTER_NEAREST, borderValue=0)
        ys, xs = np.where(m > 0)
        if xs.size > 10 and (xs.max() - xs.min()) >= (ys.max() - ys.min()):
            return cv2.GaussianBlur(g, (3, 3), 0), m
    return None


def groove_score(
    image_bgr: np.ndarray,
    contour: np.ndarray | None,
    angle_hint: float | None = None,
) -> float | None:
    """Сила продольной бороздки зерна (60-й процентиль контраста тёмного гребня).

    Принцип работы:
      1. Зерно разворачивается так, чтобы его длинная ось была горизонтальной
         (ориентация не важна — снимки могут быть любыми).
      2. Для каждой колонки вдоль оси строится профиль яркости поперёк зерна.
      3. Внутри центральной зоны (|u| <= 0.35 полуширины) ищется минимум
         профиля; его контраст к соседним участкам (|u| >= 0.42) и есть отклик
         на бороздку.
      4. Итоговый балл — 60-й процентиль контрастов по всем колонкам.

    Такая схема устойчива к ориентации и изгибу зерна: бороздка повторяет
    центральную линию формы, а её тёмное дно даёт устойчивый минимум. Затемнённые
    края отсечены зоной 0.6, поперечная трещина затемняет колонку целиком
    (и минимум, и соседи темнеют вместе — контраста не возникает), случайная
    косая трещина пересекает центр лишь в нескольких колонках и на средний
    процентиль почти не влияет.
    """
    if image_bgr is None or contour is None:
        return None
    pts = np.asarray(contour, dtype=np.int32).reshape(-1, 2)
    if len(pts) < 3:
        return None
    x, y, w, h = cv2.boundingRect(pts.reshape(-1, 1, 2))
    if w < 8 or h < 8:
        return None
    region = contour_mask(image_bgr.shape, pts, pad=2, bbox=(x, y, w, h))
    if region is None:
        return None
    x0, y0, x1, y1, mask = region
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None

    gray = to_gray(image_bgr[y0:y1, x0:x1])
    if gray is None:
        return None
    if cv2.countNonZero(mask) < 100:
        return None

    ys_m, xs_m = np.where(mask > 0)
    angle = _axis_angle(mask, angle_hint)
    if angle is None:
        return None
    rotated = _rotate_to_axis(gray, mask, angle, (float(xs_m.mean()), float(ys_m.mean())))
    if rotated is None:
        return None
    gray_r, mask_r = rotated

    cols = np.where(mask_r.any(axis=0))[0]
    if len(cols) < 8:
        return None
    contrasts: list[float] = []
    for xc in cols:
        ys = np.where(mask_r[:, xc] > 0)[0]
        if len(ys) < 8:
            continue
        center = 0.5 * (ys.min() + ys.max())
        half = 0.5 * (ys.max() - ys.min())
        if half < 5:
            continue
        u = (ys - center) / half
        sel = np.abs(u) <= _GROOVE_ZONE
        if int(sel.sum()) < 6:
            continue
        uu = u[sel]
        prof = gray_r[ys[sel], xc].astype(np.float64)
        order = np.argsort(uu)
        uu, prof = uu[order], prof[order]
        if len(prof) >= 5:  # лёгкое сглаживание шума резкости
            prof = np.convolve(np.pad(prof, (1, 1), mode="edge"), np.ones(3) / 3.0, mode="valid")
        inner = np.abs(uu) <= _GROOVE_INNER
        if int(inner.sum()) < 3:
            continue
        outer = np.abs(uu) >= _GROOVE_OUTER
        if int(outer.sum()) < 2:
            continue
        i_min = int(np.argmin(np.where(inner, prof, np.inf)))
        contrasts.append(float(prof[outer].mean()) - float(prof[i_min]))
    if len(contrasts) < 8:
        return None
    return float(np.percentile(np.asarray(contrasts, dtype=np.float64), 60))


def groove_presence(
    image_bgr: np.ndarray,
    contour: np.ndarray | None,
    *,
    angle_hint: float | None = None,
    threshold: float = GROOVE_SCORE_THRESHOLD,
) -> int | None:
    """Бинарный признак наличия бороздки: 1 — есть, 0 — нет, None — не определить."""
    score = groove_score(image_bgr, contour, angle_hint=angle_hint)
    if score is None:
        return None
    return 1 if score >= threshold else 0


def compute_seed_metrics(
    image_bgr: np.ndarray,
    contour: np.ndarray | None,
    *,
    angle_hint: float | None = None,
    groove_threshold: float = GROOVE_SCORE_THRESHOLD,
) -> dict[str, Any]:
    """Вычислить метрики одного зерна по его контуру.

    Возвращает словарь с ``rgb_averaged``, ``groove`` (1 — бороздка есть,
    0 — нет, None — признак определить не удалось), ``groove_score`` и
    измерениями длинной оси (``length_px`` / ``angle_deg``).
    """
    result: dict[str, Any] = {"rgb_averaged": None, "color_hex": "", "groove": None, "groove_score": None}
    axis = length_axis(contour)
    if axis is not None:
        length_px, angle_deg, dist = axis
        result["length_px"] = round(length_px, 3)
        result["angle_deg"] = round(angle_deg, 3)
        result["length_endpoints"] = dist
    rgb = rgb_averaged(image_bgr, contour)
    if rgb is not None:
        result["rgb_averaged"] = f"{rgb[0]:.1f};{rgb[1]:.1f};{rgb[2]:.1f}"
    result["color_hex"] = compute_median_color_hex(image_bgr, contour, is_rgb=False)
    score = groove_score(image_bgr, contour, angle_hint=angle_hint)
    if score is not None:
        result["groove_score"] = round(score, 3)
        result["groove"] = 1 if score >= groove_threshold else 0
    return result
