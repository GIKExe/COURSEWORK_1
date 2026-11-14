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


def _longest_internal_chord(
    pts: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    """Longest chord between contour points that does not leave the contour."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
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
    return a, b, dist


def length_axis(contour: np.ndarray | None) -> tuple[float, float, float] | None:
    """Return ``(length, angle_deg, length)`` for the longest internal chord.

    The third element is kept for backward compatibility; use
    :func:`measure_contour_geometry` when the actual chord endpoints are needed.
    """
    if contour is None:
        return None
    pts = simplify_contour(np.asarray(contour).reshape(-1, 2))
    if len(pts) < 2:
        return None
    chord = _longest_internal_chord(pts)
    if chord is None:
        return None
    a, b, dist = chord
    angle = float(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))) % 180.0
    return dist, angle, dist


# --- Ширина как максимальная внутренняя хорда -------------------------------
# Ширина измеряется поперёк длинной оси: для каждого положения вдоль хорды
# строится перпендикулярная прямая, находятся все её пересечения с контуром и
# берётся максимальный интервал, целиком лежащий внутри контура. Максимум по
# всем положениям и есть ширина. Это ровно «максимальное расстояние двух
# противоположных точек, не пересекающее контур», без эллипсов, прямоугольников
# и усреднений по площади.
_WIDTH_SAMPLES = 192
_WIDTH_MERGE_EPS = 1e-7


def maximum_internal_width(
    pts: np.ndarray,
    a: tuple[float, float] | np.ndarray,
    b: tuple[float, float] | np.ndarray,
    *,
    samples: int = _WIDTH_SAMPLES,
) -> tuple[float, tuple[tuple[float, float], tuple[float, float]], np.ndarray, np.ndarray, float] | None:
    """Maximum chord perpendicular to the ``a``-``b`` axis, inside the polygon.

    Returns ``(width, (p1, p2), profile, x, s_max)`` where ``profile`` is the
    widest internal interval at each sampled position along the axis,
    ``x`` are normalised positions in ``[0, 1]`` and ``s_max`` is the position
    of the maximum.  All coordinates are in the same space as ``pts``.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    length = float(math.hypot(float(b[0] - a[0]), float(b[1] - a[1])))
    if length <= 1e-9 or len(pts) < 3:
        return None

    u = (b - a) / length
    n = np.array([-u[1], u[0]], dtype=np.float64)

    p0 = pts
    p1 = np.roll(pts, -1, axis=0)
    d = p1 - p0
    den = n[0] * d[:, 1] - n[1] * d[:, 0]
    edge_ok = np.abs(den) > 1e-12

    s = np.linspace(0.0, length, max(8, int(samples)))
    q = a[None, :] + s[:, None] * u[None, :]
    qp = p0[None, :, :] - q[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        # r -- parameter along the polygon edge, t -- coordinate along n.
        r = (qp[..., 0] * n[1] - qp[..., 1] * n[0]) / den[None, :]
        t = (qp[..., 0] * d[None, :, 1] - qp[..., 1] * d[None, :, 0]) / den[None, :]

    ok = edge_ok[None, :] & (r >= 0.0) & (r < 1.0) & np.isfinite(t)
    ts = np.where(ok, t, np.inf)
    ts.sort(axis=1)

    # A scanline through a polygon vertex produces two coincident intersections.
    # Merge them, otherwise the even-odd pairing of the intervals breaks.
    keep = np.isfinite(ts)
    with np.errstate(invalid="ignore"):
        delta = np.diff(ts, axis=1)
    keep[:, 1:] &= np.isfinite(delta) & (delta > (_WIDTH_MERGE_EPS * max(length, 1.0)))

    # Pack kept values to the left, preserving ascending order. Then the kept
    # intersections of every row occupy columns 0..count-1 and the inside
    # intervals are (0,1), (2,3), ... pairs.
    order = np.argsort(~keep, axis=1, kind="stable")
    tk = np.take_along_axis(ts, order, axis=1)
    counts = keep.sum(axis=1)

    cols = ts.shape[1]
    even_cols = np.arange(0, cols - 1, 2)
    odd_cols = even_cols + 1
    starts = tk[:, even_cols]
    ends = tk[:, odd_cols]
    pair_ok = odd_cols[None, :] < counts[:, None]
    with np.errstate(invalid="ignore"):
        pair_widths = ends - starts
    pair_widths = np.where(pair_ok & np.isfinite(pair_widths), pair_widths, -1.0)
    row_widths = pair_widths.max(axis=1)

    best_row = int(np.argmax(row_widths))
    if row_widths[best_row] <= 0:
        return None
    best_col = int(np.argmax(pair_widths[best_row]))
    width = float(pair_widths[best_row, best_col])
    if width <= 0:
        return None

    t0 = float(starts[best_row, best_col])
    t1 = float(ends[best_row, best_col])
    pmin = a + s[best_row] * u + t0 * n
    pmax = a + s[best_row] * u + t1 * n

    profile = np.where(row_widths > 0, row_widths, 0.0)
    x = s / length
    return (
        width,
        ((float(pmin[0]), float(pmin[1])), (float(pmax[0]), float(pmax[1]))),
        profile,
        x,
        float(s[best_row] / length),
    )


# --- Классификация формы -----------------------------------------------------
# Литература по морфологии зерновки ячменя:
#   * основные ботанические формы — ромбическая, эллиптическая, удлинённая
#     (activestudy.info «Строение и свойства зерновки ячменя»);
#   * у удлинённой формы наиболее широкая часть находится несколько выше
#     середины, а сужение кверху резче, чем книзу (БГСХА, «Сортовые признаки
#     ячменя»);
#   * зерновка эллиптическая, с продольной неглубокой бороздкой, заострена с
#     обеих сторон (activestudy.info «Ботаническая характеристика ячменя»).
# Прикладные классы «грушевидная / цилиндрическая / игольчатая» не являются
# стандартными ботаническими терминами. Они сопоставлены с профилем ширины
# вдоль длинной оси: грушевидная — асимметричный профиль с более широкой
# частью ближе к одному концу; цилиндрическая — симметричный профиль с
# выровненной серединой; игольчатая — удлинённое зерно с узким профилем и
# заострёнными концами. Шаблоны ниже заданы в тех же единицах, что и
# нормированный профиль ширины (максимум = 1).
SHAPE_PEAR = "грушевидная"
SHAPE_CYLINDRICAL = "цилиндрическая"
SHAPE_NEEDLE = "игольчатая"
SHAPE_OPTIONS: tuple[str, ...] = (SHAPE_PEAR, SHAPE_CYLINDRICAL, SHAPE_NEEDLE)

_SHAPE_PROFILE_POINTS = 64
_SHAPE_GRID = np.linspace(0.04, 0.96, _SHAPE_PROFILE_POINTS)
_SHAPE_ANCHOR_X = np.array([
    0.06, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.94,
])

_SHAPE_TEMPLATE_ANCHORS: dict[str, np.ndarray] = {
    # Симметричный, с выровненной серединой и округлыми концами.
    SHAPE_CYLINDRICAL: np.array([
        0.421, 0.509, 0.617, 0.697, 0.775, 0.866, 0.966, 0.993,
        0.966, 0.866, 0.775, 0.697, 0.617, 0.509, 0.421,
    ]),
    # Асимметричный: максимум смещён влево, к правому концу сужение резче.
    SHAPE_PEAR: np.array([
        0.515, 0.640, 0.771, 0.844, 0.904, 0.965, 0.995, 0.954,
        0.853, 0.694, 0.590, 0.514, 0.447, 0.376, 0.329,
    ]),
    # Удлинённый, постепенно сужающийся к обоим концам.
    SHAPE_NEEDLE: np.array([
        0.397, 0.480, 0.587, 0.668, 0.748, 0.844, 0.957, 0.992,
        0.957, 0.844, 0.748, 0.668, 0.587, 0.480, 0.397,
    ]),
}

_SHAPE_TEMPLATES: dict[str, np.ndarray] = {
    name: np.interp(_SHAPE_GRID, _SHAPE_ANCHOR_X, anchors)
    for name, anchors in _SHAPE_TEMPLATE_ANCHORS.items()
}

# Пороги и веса подобраны на предоставленном наборе из 2600 контуров:
# у 90% зёрен вытянутость L/W < 2.40, у 90% асимметрия профиля < 1.36.
_NEEDLE_ASPECT_MIN = 2.40
_NEEDLE_FLAT_MAX = 0.82
_NEEDLE_ASPECT_WEIGHT = 0.25
_NEEDLE_FLAT_WEIGHT = 0.15
_PEAR_ASYM_MIN = 1.36
_PEAR_ASYM_WEIGHT = 0.20


def _normalized_shape_profile(profile: np.ndarray, x: np.ndarray) -> np.ndarray | None:
    """Resample, smooth and normalise a width profile to a unit maximum."""
    profile = np.asarray(profile, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if profile.size < 3 or x.size != profile.size:
        return None
    p = np.interp(_SHAPE_GRID, x, profile)
    p = np.maximum(p, 0.0)
    peak = float(p.max())
    if peak <= 1e-9:
        return None
    p /= peak
    # Лёгкое сглаживание: на вогнутом контуре профиль может «дрожать».
    if p.size >= 5:
        kernel = np.ones(5, dtype=np.float64) / 5.0
        p = np.convolve(np.pad(p, (2, 2), mode="edge"), kernel, mode="valid")
        p /= max(float(p.max()), 1e-9)
    return p


def classify_seed_shape(profile: np.ndarray, x: np.ndarray, aspect: float) -> str:
    """Choose the best of the three practical barley-grain shape templates.

    The contour's width profile is compared with each template by mean absolute
    distance.  Aspect-ratio and asymmetry priors resolve cases where the profiles
    are close: a needle-shaped grain must be elongated, a pear-shaped one must be
    noticeably asymmetric.
    """
    p = _normalized_shape_profile(profile, x)
    if p is None:
        return SHAPE_CYLINDRICAL

    left = float(p[_SHAPE_GRID < 0.35].mean())
    right = float(p[_SHAPE_GRID > 0.65].mean())
    asymmetry = max(left, right) / max(min(left, right), 1e-9)
    flat = float(np.mean(p[(_SHAPE_GRID > 0.25) & (_SHAPE_GRID < 0.75)] >= 0.85))

    def distance(template: np.ndarray) -> float:
        return float(np.mean(np.abs(p - template)))

    d_cyl = distance(_SHAPE_TEMPLATES[SHAPE_CYLINDRICAL])
    d_needle = distance(_SHAPE_TEMPLATES[SHAPE_NEEDLE])
    # The pear template is asymmetric; the grain can be oriented either way.
    pear = _SHAPE_TEMPLATES[SHAPE_PEAR]
    d_pear = min(distance(pear), distance(pear[::-1]))

    scores = {
        SHAPE_CYLINDRICAL: d_cyl,
        SHAPE_PEAR: d_pear + _PEAR_ASYM_WEIGHT * max(0.0, _PEAR_ASYM_MIN - asymmetry),
        SHAPE_NEEDLE: (
            d_needle
            + _NEEDLE_ASPECT_WEIGHT * max(0.0, _NEEDLE_ASPECT_MIN - float(aspect))
            + _NEEDLE_FLAT_WEIGHT * max(0.0, flat - _NEEDLE_FLAT_MAX)
        ),
    }
    return min(scores, key=scores.get)


def _aligned_endpoints(
    a1: tuple[float, float], b1: tuple[float, float],
    a2: tuple[float, float], b2: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Average two chords after giving their endpoints a consistent order."""
    p1 = np.asarray(a1, dtype=np.float64)
    q1 = np.asarray(b1, dtype=np.float64)
    p2 = np.asarray(a2, dtype=np.float64)
    q2 = np.asarray(b2, dtype=np.float64)
    if float(np.dot(q1 - p1, q2 - p2)) < 0.0:
        p2, q2 = q2, p2
    return (p1 + p2) / 2.0, (q1 + q2) / 2.0


def measure_contour_geometry(contour: np.ndarray | None) -> dict[str, Any]:
    """Measure the seed from its contour alone (no image needed).

    The measurement is a perpendicular cross built from two bases:

    1. the longest internal chord gives the length base; the maximum chord
       perpendicular to it gives the width base;
    2. the width base in turn gives a second length base (perpendicular to it),
       and that second length base gives a second width base.

    The final length and width endpoints are the averages of the corresponding
    endpoints of both bases.  Since the two length bases are parallel and the
    two width bases are parallel, the averaged segments stay perpendicular.

    Returns ``length_px``, ``angle_deg``, ``length_endpoints``, ``width_px``,
    ``width_endpoints``, ``width_position`` and ``shape``.  Missing keys mean
    that the contour was degenerate.
    """
    if contour is None:
        return {}
    pts = simplify_contour(np.asarray(contour).reshape(-1, 2))
    if len(pts) < 3:
        return {}
    # Measure in a local frame.  The contour JSON keeps crop-local coordinates
    # while the segmentation pipeline measures source-image coordinates; the
    # shift between them must not change any result (including shape, which can
    # flip when a contour lies exactly on a template boundary).
    origin = pts[0].copy()
    pts = pts - origin
    chord = _longest_internal_chord(pts)
    if chord is None:
        return {}
    a1, b1, length1 = chord

    width_base = maximum_internal_width(pts, a1, b1)
    if width_base is None:
        angle = float(math.degrees(math.atan2(b1[1] - a1[1], b1[0] - a1[0]))) % 180.0
        return {
            "length_px": round(float(length1), 3),
            "angle_deg": round(angle, 3),
            "length_endpoints": (
                (float(a1[0] + origin[0]), float(a1[1] + origin[1])),
                (float(b1[0] + origin[0]), float(b1[1] + origin[1])),
            ),
        }

    width1, (c1, d1), profile, x, position = width_base
    a_final, b_final = np.asarray(a1, dtype=np.float64), np.asarray(b1, dtype=np.float64)
    c_final, d_final = np.asarray(c1, dtype=np.float64), np.asarray(d1, dtype=np.float64)
    length_final = float(length1)
    width_final = float(width1)

    # Second base: longest chord perpendicular to the width base.  It is
    # parallel to the first length chord, but sampled independently.
    length_base2 = maximum_internal_width(pts, c1, d1)
    if length_base2 is not None:
        length2, (a2, b2), _, _, _ = length_base2
        width_base2 = maximum_internal_width(pts, a2, b2)
        if width_base2 is not None:
            width2, (c2, d2), _, _, _ = width_base2
            a_final, b_final = _aligned_endpoints(a1, b1, a2, b2)
            c_final, d_final = _aligned_endpoints(c1, d1, c2, d2)
            length_final = float(np.linalg.norm(b_final - a_final))
            width_final = float(np.linalg.norm(d_final - c_final))

    if length_final <= 1e-9 or width_final <= 1e-9:
        return {}

    # Shape is classified from the width profile across the final length axis.
    profile_result = maximum_internal_width(pts, a_final, b_final)
    if profile_result is not None:
        _, _, profile, x, position = profile_result
    shape = classify_seed_shape(profile, x, length_final / max(width_final, 1e-9))
    angle = float(math.degrees(math.atan2(
        float(b_final[1] - a_final[1]), float(b_final[0] - a_final[0])
    ))) % 180.0
    return {
        "length_px": round(length_final, 3),
        "angle_deg": round(angle, 3),
        "length_endpoints": (
            (float(a_final[0] + origin[0]), float(a_final[1] + origin[1])),
            (float(b_final[0] + origin[0]), float(b_final[1] + origin[1])),
        ),
        "width_px": round(width_final, 3),
        "width_endpoints": (
            (float(c_final[0] + origin[0]), float(c_final[1] + origin[1])),
            (float(d_final[0] + origin[0]), float(d_final[1] + origin[1])),
        ),
        "width_position": round(float(position), 4),
        "shape": shape,
    }


def compute_median_rgb(
    image: np.ndarray | Any,
    contour: Any,
    *,
    is_rgb: bool = False,
) -> tuple[int, int, int] | None:
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
    result: dict[str, Any] = {
        "rgb_averaged": None, "color_hex": "", "groove": None, "groove_score": None,
    }
    result.update(measure_contour_geometry(contour))
    rgb = rgb_averaged(image_bgr, contour)
    if rgb is not None:
        result["rgb_averaged"] = f"{rgb[0]:.1f};{rgb[1]:.1f};{rgb[2]:.1f}"
    result["color_hex"] = compute_median_color_hex(image_bgr, contour, is_rgb=False)
    score = groove_score(image_bgr, contour, angle_hint=angle_hint)
    if score is not None:
        result["groove_score"] = round(score, 3)
        result["groove"] = 1 if score >= groove_threshold else 0
    return result
