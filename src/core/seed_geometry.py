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


def points_to_contour(points: Iterable[Any] | None) -> np.ndarray | None:
    """Convert a heterogeneous point list to an (N, 1, 2) int32 contour."""
    if points is None:
        return None
    pts: list[tuple[int, int]] = []
    for point in points:
        try:
            if isinstance(point, dict):
                x, y = float(point["x"]), float(point["y"])
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                x, y = float(point[0]), float(point[1])
            else:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
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


def rgb_averaged(image_bgr: np.ndarray, contour: np.ndarray | None) -> tuple[float, float, float] | None:
    """Mean R, G, B over the pixels strictly *inside* the contour."""
    if image_bgr is None or contour is None:
        return None
    pts = np.asarray(contour, dtype=np.int32).reshape(-1, 2)
    if len(pts) < 3:
        return None
    x, y, w, h = cv2.boundingRect(pts.reshape(-1, 1, 2))
    img_h, img_w = image_bgr.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(img_w, int(x + w)), min(img_h, int(y + h))
    if x1 <= x0 or y1 <= y0:
        return None
    crop = image_bgr[y0:y1, x0:x1]
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    shifted = (pts - np.array([x0, y0], dtype=np.int32)).reshape(-1, 1, 2)
    cv2.drawContours(mask, [shifted], -1, 255, thickness=cv2.FILLED)
    if cv2.countNonZero(mask) == 0:
        return None
    mean = cv2.mean(crop, mask=mask)  # BGR(A) order
    return float(mean[2]), float(mean[1]), float(mean[0])


def groove_score(image_bgr: np.ndarray, contour: np.ndarray | None, angle_hint: float | None = None) -> float:
    """Strength of the longitudinal groove (dark stripe along the seed axis).

    Only pixels inside the eroded contour are analysed: the brightness of a
    narrow band around the length axis is compared against the flanks on both
    sides in four segments along the length.  A persistent central darkening
    (the groove facing the camera) gives a high score; a smooth back gives
    roughly zero.
    """
    if image_bgr is None or contour is None:
        return 0.0
    axis = length_axis(contour)
    if axis is None:
        return 0.0
    _, ang, _ = axis
    if angle_hint is not None:
        try:
            ang = float(angle_hint) % 180.0
        except (TypeError, ValueError):
            pass
    pts = np.asarray(contour, dtype=np.int32).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    x, y, w, h = cv2.boundingRect(pts.reshape(-1, 1, 2))
    img_h, img_w = image_bgr.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(img_w, int(x + w)), min(img_h, int(y + h))
    if x1 - x0 < 12 or y1 - y0 < 12:
        return 0.0
    crop = image_bgr[y0:y1, x0:x1]
    cw, chh = x1 - x0, y1 - y0
    mask = np.zeros((chh, cw), dtype=np.uint8)
    cv2.drawContours(mask, [(pts - np.array([x0, y0])).reshape(-1, 1, 2)], -1, 255, thickness=cv2.FILLED)
    erode_k = max(3, int(min(cw, chh) * 0.12) | 1)
    inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k)))
    if cv2.countNonZero(inner) < 60:
        inner = mask
    if cv2.countNonZero(inner) < 60:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    rad = math.radians(ang)
    ux, uy = math.cos(rad), math.sin(rad)
    vx, vy = -uy, ux
    ys, xs = np.mgrid[0:chh, 0:cw].astype(np.float32)
    t = xs * ux + ys * uy   # coordinate along the seed length
    s = xs * vx + ys * vy   # coordinate across the seed width
    sel_m = mask > 0
    sv_all = s[sel_m]
    if sv_all.size < 20:
        return 0.0
    s_center = float(np.median(sv_all))
    s_span = float(np.percentile(sv_all, 97) - np.percentile(sv_all, 3))
    half_core = max(3.0, s_span * 0.12)
    half_flank = max(3.0, s_span * 0.30)
    ds = np.abs(s - s_center)
    core = (inner > 0) & (ds <= half_core)
    flank = (inner > 0) & (ds > half_flank)
    if core.sum() < 20 or flank.sum() < 20:
        return 0.0
    tv = t[core]
    gv = gray[core]
    flank_mean = float(np.mean(gray[flank]))
    bins = np.quantile(tv, [0.0, 0.25, 0.5, 0.75, 1.0])
    deltas: list[float] = []
    dark_segments = 0
    for k in range(4):
        seg = (tv >= bins[k]) & (tv <= bins[k + 1])
        if seg.sum() < 5:
            continue
        delta = flank_mean - float(np.mean(gv[seg]))  # > 0 when the core is darker
        deltas.append(delta)
        if delta > 1.0:
            dark_segments += 1
    if not deltas:
        return 0.0
    fraction = dark_segments / len(deltas)
    depth = float(np.mean(deltas))
    if fraction < 0.5 or depth <= 0.0:
        return 0.0
    return depth * fraction


def compute_seed_metrics(
    image_bgr: np.ndarray,
    contour: np.ndarray | None,
    *,
    angle_hint: float | None = None,
    groove_threshold: float = 3.0,
) -> dict[str, Any]:
    """Compute contour-based metrics for one seed.

    Returns a dict with ``rgb_averaged``, ``groove`` (0/1) and the length-axis
    measurements (``length_px`` / ``angle_deg`` plus the endpoint coordinates).
    """
    result: dict[str, Any] = {"rgb_averaged": None, "groove": None, "groove_score": 0.0}
    axis = length_axis(contour)
    if axis is not None:
        length_px, angle_deg, dist = axis
        result["length_px"] = round(length_px, 3)
        result["angle_deg"] = round(angle_deg, 3)
        result["length_endpoints"] = dist
    rgb = rgb_averaged(image_bgr, contour)
    if rgb is not None:
        result["rgb_averaged"] = f"{rgb[0]:.1f};{rgb[1]:.1f};{rgb[2]:.1f}"
    score = groove_score(image_bgr, contour, angle_hint=angle_hint)
    result["groove_score"] = round(score, 3)
    if rgb is not None or axis is not None:
        result["groove"] = 1 if score >= groove_threshold else 0
    return result
