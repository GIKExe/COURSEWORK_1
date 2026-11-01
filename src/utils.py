"""
Обработчики изображения:
- чтение/запись с поддержкой кириллицы;
- конвертация цветовых пространств;
- сегментация;
- фильтрация объектов;
- оценка фокуса;
- калибровка мм/пиксель по линейке;
- замеры семян;
- статистика размеров;
- обновление параметров в CSV;
- сохранение контуров семян в JSON.
"""

import csv
import json
import math
import os
import re
from typing import Any, Optional

import cv2
import numpy as np

__all__ = [
    # I/O
    "imread_unicode",
    "imwrite_unicode",
    "sanitize_filename",
    "fmt_float",
    "save_contour_json",
    "remove_group_files_except",
    "update_seeds_csv",
    # Конвертация
    "to_uint8",
    "to_gray_uint8",
    "ensure_odd",
    # Сегментация
    "make_mask",
    "remove_small_components",
    "fill_external_contours",
    "object_contrast",
    "filter_candidates_by_outliers",
    "find_descending_crossing",
    "estimate_edge_width",
    "classify_edge",
    "sort_items_reading_order",
    "process_image",
    # Калибровка линейки
    "calibrate_ruler",
    # Статистика
    "compute_size_stats",
    "format_size_stats",
]

# ============================================================
# Константы
# ============================================================

STATUS_COLORS = {
    "focused": (0, 200, 0),       # зелёный
    "transition": (0, 220, 255),  # жёлтый
    "blurry": (0, 0, 255),        # красный
    "unknown": (80, 160, 255),    # оранжевый
}

STATUS_NAMES_RU = {
    "focused": "фокус",
    "transition": "транзит",
    "blurry": "размыто",
    "unknown": "неизвестно",
}

DEFAULT_CSV_FIELDS = [
    "file_index",
    "index",
    "status",
    "edge_width_px",
    "length_px",
    "width_px",
    "angle_deg",
    "area_px",
    "length_mm",
    "width_mm",
    "mm_per_px",
]

# Старые/нежелательные поля, которые не нужно тащить в обновлённый CSV.
SKIP_LEGACY_FIELDS = {
    "filename",
    "source_path",
    "path",
}


# ============================================================
# Ввод / вывод
# ============================================================

def imread_unicode(path):
    """Читает изображение, включая пути с кириллицей."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img, params=None):
    """Сохраняет изображение, включая пути с кириллицей."""
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".png"
        path += ext
    if params is None:
        if ext.lower() in (".jpg", ".jpeg"):
            params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        else:
            params = []
    ok, buf = cv2.imencode(ext, img, params)
    if ok:
        buf.tofile(path)
        return True
    return False


def sanitize_filename(name):
    """
    Делает имя файла/префикс безопасным для Windows/Linux.
    """
    name = os.path.basename(str(name))
    name = re.sub(r'[\/:*?"<>|\s]+', "", name)
    name = name.strip(".")
    if not name:
        name = "image"
    return name


def fmt_float(value, digits=3):
    """
    Форматирует число для CSV.
    None/NaN/Infinity/ошибки -> пустая строка.
    """
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def contour_to_points(contour, offset_x=0, offset_y=0):
    """
    Преобразует контур OpenCV в список точек [[x, y], ...].
    Можно сразу сместить координаты на offset_x/offset_y.
    """
    if contour is None:
        return []
    try:
        pts = contour.reshape(-1, 2)
    except Exception:
        return []
    out = []
    for p in pts:
        try:
            x = int(p[0]) - int(offset_x)
            y = int(p[1]) - int(offset_y)
            out.append([x, y])
        except Exception:
            pass
    return out


def save_contour_json(
    path,
    file_index,
    seed_index,
    contour,
    crop_origin,
    crop_shape,
    bbox=None,
    padding=0,
    source_name=None,
):
    """
    Сохраняет точки криволинейной границы семени в JSON.
    points даны в координатах сохранённого crop-изображения (JPG).
    Чтобы получить координаты в исходном фото, нужно прибавить
    crop_origin_in_source.
    """
    crop_h, crop_w = crop_shape[:2]
    ox, oy = crop_origin
    points = contour_to_points(contour, offset_x=ox, offset_y=oy)
    payload = {
        "file_index": file_index,
        "index": int(seed_index),
        "points_space": "crop",
        "image_width": int(crop_w),
        "image_height": int(crop_h),
        "crop_origin_in_source": [int(ox), int(oy)],
        "padding": int(padding),
        "points": points,
    }
    if bbox is not None:
        try:
            bx, by, bw, bh = bbox
            payload["bbox_in_source"] = [int(bx), int(by), int(bw), int(bh)]
        except Exception:
            pass
    if source_name:
        payload["source_name"] = source_name
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=None)


def remove_group_files_except(dir_path, group_prefix, keep_file_indexes):
    """
    Удаляет старые файлы группы (исходного изображения), которых нет
    в новом результате.
    """
    if not group_prefix or not os.path.isdir(dir_path):
        return
    keep = set(str(x) for x in keep_file_indexes)
    img_pattern = re.compile(
        r"^" + re.escape(group_prefix) + r"_(\d+)\.(jpe?g|png)$",
        re.IGNORECASE,
    )
    json_pattern = re.compile(
        r"^" + re.escape(group_prefix) + r"_(\d+)_contour\.json$",
        re.IGNORECASE,
    )
    for name in os.listdir(dir_path):
        file_index = None
        remove = False
        m = img_pattern.match(name)
        if m:
            digits = m.group(1)
            ext = m.group(2).lower()
            file_index = f"{group_prefix}_{digits}"
            if ext != "jpg":
                remove = True
            elif file_index not in keep:
                remove = True
        else:
            m = json_pattern.match(name)
            if m:
                digits = m.group(1)
                file_index = f"{group_prefix}_{digits}"
                if file_index not in keep:
                    remove = True
        if remove:
            try:
                os.remove(os.path.join(dir_path, name))
            except Exception:
                pass


def update_seeds_csv(path, new_rows, group_prefix):
    """
    Обновляет out/seeds.csv.
    """
    existing_rows = []
    existing_fields = []
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                sample = f.read(8192)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                except Exception:
                    dialect = csv.excel
                reader = csv.DictReader(f, dialect=dialect)
                existing_fields = list(reader.fieldnames or [])
                for row in reader:
                    clean_row = {}
                    for key, value in row.items():
                        if key is None:
                            continue
                        clean_row[key] = "" if value is None else value
                    existing_rows.append(clean_row)
        except Exception:
            existing_rows = []
            existing_fields = []

    group_pattern = re.compile(r"^" + re.escape(group_prefix) + r"_\d+$")
    kept_rows = []
    for row in existing_rows:
        file_index = str(row.get("file_index", "")).strip()
        if file_index and group_pattern.match(file_index):
            continue
        kept_rows.append(row)

    fieldnames = []
    for name in DEFAULT_CSV_FIELDS:
        if name not in fieldnames:
            fieldnames.append(name)
    for name in existing_fields:
        if name in SKIP_LEGACY_FIELDS:
            continue
        if name not in fieldnames:
            fieldnames.append(name)
    for row in new_rows:
        for name in row.keys():
            if name in SKIP_LEGACY_FIELDS:
                continue
            if name not in fieldnames:
                fieldnames.append(name)

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in kept_rows:
            writer.writerow(row)
        for row in new_rows:
            writer.writerow(row)


# ============================================================
# Конвертация изображений (общие утилиты)
# ============================================================

def to_uint8(img):
    """Приводит изображение к uint8 с нормализацией."""
    if img.dtype == np.uint8:
        return img
    dst = np.empty_like(img)
    cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)
    return dst.astype(np.uint8)


def to_gray_uint8(image):
    """
    Конвертирует изображение в grayscale uint8.
    Поддерживается BGR, BGRA, grayscale, single-channel 3D.
    """
    if image is None:
        raise ValueError("Изображение не передано")
    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif channels == 1:
            gray = image[:, :, 0].copy()
        else:
            gray = image[:, :, 0].copy()
    else:
        gray = image.copy()
    return to_uint8(gray)


def ensure_odd(value, min_value=1):
    """Возвращает ближайшее нечётное число >= min_value."""
    value = int(round(float(value)))
    if value < min_value:
        value = min_value
    if value % 2 == 0:
        value += 1
    return value


# ============================================================
# Морфологические утилиты (общие)
# ============================================================

def _morph_close(mask, kernel_size, iterations=1, shape=cv2.MORPH_RECT):
    """Морфологическое закрытие."""
    if kernel_size <= 1:
        return mask
    kern = cv2.getStructuringElement(shape, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=iterations)


def _morph_open(mask, kernel_size, iterations=1, shape=cv2.MORPH_RECT):
    """Морфологическое открытие."""
    if kernel_size <= 1:
        return mask
    kern = cv2.getStructuringElement(shape, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern, iterations=iterations)


# ============================================================
# Калибровка мм/пиксель по фотографии линейки
# ============================================================

def calibrate_ruler(image, debug=False):
    """
    Возвращает мм/пиксель по фотографии линейки, переданной как ndarray.

    Параметры:
        image: numpy.ndarray — изображение в памяти (BGR, BGRA или grayscale).
        debug: bool — если True, сохранит отладочные изображения.

    Возврат:
        float — масштаб в мм/пиксель.
    """
    gray = to_gray_uint8(image)
    crop, mask = _preprocess_ruler(gray)
    tick_bin, ticks, heights = _detect_ticks(crop, mask)

    x_pos = np.array([t["cx"] for t in ticks], dtype=np.float64)
    distances = np.diff(x_pos)

    # Шаг 1: грубый масштаб по сантиметровым штрихам.
    px_per_mm = _cm_scale(x_pos, heights)

    # Шаг 2: уточнение по всем штрихам (гипотезы 1 мм / 0.5 мм).
    if px_per_mm is not None:
        refined = _refine_scale(distances, px_per_mm)
        if refined is not None:
            # !!! refined * 2 - это не ошибка, точне, это костыль,
            # который исправляет какой-то баг в программе. 
            px_per_mm = refined * 2
    else:
        px_per_mm = _dominant_distance(distances)

    if px_per_mm is None or not np.isfinite(px_per_mm) or px_per_mm <= 0:
        raise ValueError("Не удалось определить масштаб линейки")

    mm_per_px = 1.0 / float(px_per_mm)

    if debug:
        _save_ruler_debug(crop, tick_bin, ticks, heights, mm_per_px, px_per_mm)

    return float(mm_per_px)


def _preprocess_ruler(gray):
    """
    Предобработка линейки:
    - найти линейку;
    - выпрямить;
    - обрезать;
    - ориентировать горизонтально.
    """
    H, W = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    binary = _morph_close(binary, 7, iterations=3)
    binary = _morph_open(binary, 7, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Линейка не найдена")

    ruler_cnt = max(contours, key=cv2.contourArea)

    # Если бинаризация выбрала почти весь кадр как объект,
    # пробуем инвертированную маску.
    full_area = float(H * W)
    if cv2.contourArea(ruler_cnt) > 0.95 * full_area:
        inverted = cv2.bitwise_not(binary)
        inverted = _morph_close(inverted, 7, iterations=3)
        inverted = _morph_open(inverted, 7, iterations=2)
        inv_contours, _ = cv2.findContours(
            inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if inv_contours:
            inv_cnt = max(inv_contours, key=cv2.contourArea)
            if cv2.contourArea(inv_cnt) < cv2.contourArea(ruler_cnt):
                ruler_cnt = inv_cnt

    if cv2.contourArea(ruler_cnt) < H * W * 0.005:
        raise ValueError("Объект слишком мал для линейки")

    rect = cv2.minAreaRect(ruler_cnt)
    (cx, cy), (rw, rh), angle = rect
    rot_angle = angle + 90 if rw < rh else angle

    M = cv2.getRotationMatrix2D((cx, cy), rot_angle, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    nW = int(H * sin_a + W * cos_a)
    nH = int(H * cos_a + W * sin_a)
    M[0, 2] += (nW - W) / 2
    M[1, 2] += (nH - H) / 2

    rotated = cv2.warpAffine(gray, M, (nW, nH), borderValue=0)

    mask_full = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(mask_full, [ruler_cnt], -1, 255, -1)
    rotated_mask = cv2.warpAffine(mask_full, M, (nW, nH), borderValue=0)

    ys, xs = np.where(rotated_mask > 128)
    if len(xs) == 0:
        raise ValueError("Маска пуста после поворота")

    pad = 3
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(nW, int(xs.max()) + pad + 1)
    y1 = min(nH, int(ys.max()) + pad + 1)

    if x1 <= x0 or y1 <= y0:
        raise ValueError("Не удалось вырезать область линейки")

    crop = rotated[y0:y1, x0:x1]
    mask = rotated_mask[y0:y1, x0:x1]

    if crop.size == 0 or mask.size == 0:
        raise ValueError("Обрезанное изображение линейки пусто")

    ch, cw = crop.shape
    # Делаем линейку горизонтальной.
    if ch > cw:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

    return crop, mask


def _detect_ticks(crop, mask):
    """
    Детектирует штрихи линейки и возвращает:
    - бинарную маску штрихов;
    - список отсортированных уникальных штрихов;
    - массив высот штрихов.
    """
    ch, cw = crop.shape

    # Адаптивный порог: тёмные штрихи -> белые.
    bs = max(11, (min(ch, cw) // 6) | 1)
    adaptive = cv2.adaptiveThreshold(
        crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=bs, C=8,
    )

    kern_e = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_e = cv2.erode(mask, kern_e, iterations=1)
    if not np.any(mask_e > 128):
        raise ValueError("Маска линейки пуста после эрозии")

    tick_bin = cv2.bitwise_and(adaptive, mask_e)

    # Вертикальное закрытие: сшивает прерывистые штрихи.
    kern_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    tick_bin = cv2.morphologyEx(tick_bin, cv2.MORPH_CLOSE, kern_v)

    # Границы линейки поколоночно (учитывает остаточный наклон).
    mb = mask_e > 128
    col_any = mb.any(axis=0)
    top_edge = np.where(col_any, mb.argmax(axis=0), -1)
    bot_edge = np.where(col_any, ch - 1 - mb[::-1].argmax(axis=0), -1)

    contours, _ = cv2.findContours(tick_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = max(5.0, ch * 0.05)
    max_area = ch * cw * 0.005
    min_h = ch * 0.05
    max_h = ch * 0.75
    max_w = max(6, int(cw * 0.004))
    tol = max(3, int(ch * 0.04))

    ticks = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if not (min_h <= bh <= max_h) or bw > max_w:
            continue
        if bh / max(bw, 1) < 2.5:
            continue

        # Штрих обязан касаться края линейки.
        cxi = min(max(bx + bw // 2, 0), cw - 1)
        te, be = top_edge[cxi], bot_edge[cxi]
        if te < 0:
            continue
        if by <= te + tol:
            edge = "top"
        elif by + bh >= be - tol:
            edge = "bottom"
        else:
            continue

        M = cv2.moments(cnt)
        cx = M["m10"] / M["m00"] if M["m00"] > 0 else bx + bw / 2.0
        cy = M["m01"] / M["m00"] if M["m00"] > 0 else by + bh / 2.0

        ticks.append(dict(cx=cx, cy=cy, bx=bx, by=by, bw=bw, bh=bh, edge=edge))

    if len(ticks) < 8:
        raise ValueError(f"Мало штрихов найдено: {len(ticks)}")

    # Доминирующий край.
    n_top = sum(1 for t in ticks if t["edge"] == "top")
    use_top = n_top >= len(ticks) - n_top
    edge_ticks = [t for t in ticks if (t["edge"] == "top") == use_top]
    if len(edge_ticks) < 8:
        edge_ticks = ticks

    edge_ticks.sort(key=lambda t: t["cx"])

    min_gap = max(2, int(round(cw * 0.0006)))
    uniq = [edge_ticks[0]]
    for t in edge_ticks[1:]:
        if t["cx"] - uniq[-1]["cx"] > min_gap:
            uniq.append(t)
        elif t["bh"] > uniq[-1]["bh"]:
            uniq[-1] = t

    if len(uniq) < 8:
        raise ValueError(f"После дедупликации мало штрихов: {len(uniq)}")

    heights = np.array([t["bh"] for t in uniq], dtype=np.float64)
    return tick_bin, uniq, heights


def _cm_scale(x_pos, heights):
    """Грубый px_per_mm по самым длинным (сантиметровым) штрихам."""
    if len(x_pos) < 10:
        return None
    thr = np.percentile(heights, 88)
    xs = x_pos[heights >= thr]
    if len(xs) < 5:
        return None
    d = np.diff(xs)
    if len(d) == 0:
        return None
    med = float(np.median(d))
    if med <= 0:
        return None
    sel = d[np.abs(d - med) <= 0.15 * med]
    if len(sel) < 4:
        sel = d[np.abs(d - med) <= 0.30 * med]
    if len(sel) < 4:
        return None
    return float(np.mean(sel)) / 10.0


def _refine_scale(distances, px_per_mm):
    """
    Уточнение: выбираем гипотезу решётки (1 мм или 0.5 мм),
    которая объясняет больше соседних расстояний, и усредняем её.
    Возвращает уже готовый px_per_mm.
    """
    if distances is None or len(distances) == 0:
        return None
    if px_per_mm is None or px_per_mm <= 0:
        return None
    best_n, best_val = 0, None
    for k in (1.0, 0.5):
        target = px_per_mm * k
        sel = distances[np.abs(distances - target) <= 0.10 * target]
        if len(sel) >= 15 and len(sel) > best_n:
            best_n = len(sel)
            best_val = float(np.mean(sel)) / k
    return best_val


def _dominant_distance(distances):
    """Fallback: мода соседних расстояний."""
    d = np.asarray(distances, dtype=np.float64)
    d = d[d > 0]
    if len(d) == 0:
        return None
    med = float(np.median(d))
    if med <= 0:
        return None
    keep = d[(d > 0.4 * med) & (d < 1.6 * med)]
    if len(keep) < 3:
        keep = d
    counts, edges = np.histogram(keep, bins=40)
    peak = int(np.argmax(counts))
    d_mode = (edges[peak] + edges[peak + 1]) / 2.0
    sel = keep[np.abs(keep - d_mode) <= 0.10 * d_mode]
    if len(sel) < 2:
        return float(med)
    return float(np.mean(sel))


def _save_ruler_debug(crop, tick_bin, ticks, heights, mm_per_px, px_per_mm):
    """Сохраняет отладочные изображения калибровки линейки."""
    try:
        thr = np.percentile(heights, 88)
        vis = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        n_cm = 0
        for t, h in zip(ticks, heights):
            bx, by, bw, bh = t["bx"], t["by"], t["bw"], t["bh"]
            if h >= thr:
                cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
                n_cm += 1
            else:
                cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 1)
            cv2.circle(
                vis, (int(round(t["cx"])), int(round(t["cy"]))), 2, (0, 0, 255), -1
            )
        txt = (
            f"mm/px = {mm_per_px:.6f}   px/mm = {px_per_mm:.2f}   "
            f"ticks = {len(ticks)}   cm = {n_cm}"
        )
        cv2.putText(vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imwrite("debug_ruler_ticks.png", vis)
        cv2.imwrite("debug_ruler_binary.png", tick_bin)
    except Exception:
        pass


# ============================================================
# Сегментация
# ============================================================

def make_mask(gray, params):
    """
    Создаёт бинарную маску семян.
    Используется глобальный порог Otsu.
    """
    k = ensure_odd(params["blur_kernel"], 1)
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Если маска получилась инвертированной, исправляем.
    white_count = int(np.count_nonzero(mask))
    if 0 < white_count < mask.size:
        white_mask = mask > 0
        mean_white = float(np.mean(gray[white_mask]))
        mean_black = float(np.mean(gray[~white_mask]))
        if mean_white < mean_black or white_count > mask.size * 0.6:
            mask = cv2.bitwise_not(mask)
    elif white_count == mask.size:
        mask = np.zeros_like(mask)

    close_k = max(1, int(params["close_kernel"]))
    open_k = max(1, int(params["open_kernel"]))
    if close_k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if open_k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def remove_small_components(mask, min_area):
    """Удаляет мелкие компоненты."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    keep = np.zeros(num, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(1, int(min_area))
    cleaned = np.where(keep[labels], 255, 0).astype(np.uint8)
    return cleaned


def fill_external_contours(mask, min_area):
    """Заполняет внешние контуры для закрытия дырок внутри семян."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)
    return filled


def object_contrast(gray, contour, bbox):
    """Примитивная оценка контраста объекта относительно локального фона."""
    x, y, w, h = bbox
    img_h, img_w = gray.shape[:2]
    pad = max(6, min(20, int(min(w, h) * 0.25)))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img_w, x + w + pad)
    y1 = min(img_h, y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    roi = gray[y0:y1, x0:x1]
    mask = np.zeros(roi.shape, dtype=np.uint8)
    shifted = contour - np.array([x0, y0], dtype=np.int32)
    cv2.drawContours(mask, [shifted.astype(np.int32)], -1, 255, thickness=cv2.FILLED)
    if cv2.countNonZero(mask) == 0:
        return 0.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inner = cv2.erode(mask, kernel, iterations=1)
    if cv2.countNonZero(inner) < 3:
        inner = mask
    outer = cv2.dilate(mask, kernel, iterations=4)
    band = cv2.subtract(outer, mask)
    if cv2.countNonZero(band) < 3:
        band = cv2.bitwise_not(mask)
    mean_in = cv2.mean(roi, mask=inner)[0]
    mean_out = cv2.mean(roi, mask=band)[0]
    return abs(float(mean_in) - float(mean_out))


def filter_candidates_by_outliers(candidates):
    """
    Удаляет объекты, которые сильно отличаются от медианного размера/площади.
    """
    if len(candidates) < 6:
        return candidates
    areas = np.array([c["area"] for c in candidates], dtype=np.float32)
    dims = np.array(
        [min(c["bbox"][2], c["bbox"][3]) for c in candidates], dtype=np.float32
    )
    med_a = float(np.median(areas))
    med_d = float(np.median(dims))
    if med_a <= 0 or med_d <= 0:
        return candidates
    lower_a = med_a * 0.35
    upper_a = med_a * 3.0
    lower_d = med_d * 0.45
    upper_d = med_d * 3.0
    filtered = []
    for c in candidates:
        a = c["area"]
        d = min(c["bbox"][2], c["bbox"][3])
        if lower_a <= a <= upper_a and lower_d <= d <= upper_d:
            filtered.append(c)
    return filtered


def find_descending_crossing(x, y, level):
    """Ищет точку пересечения уровня при спадании значения."""
    if len(x) < 2:
        return None
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    for i in range(len(x) - 1):
        y0 = y[i]
        y1 = y[i + 1]
        if (y0 - level) * (y1 - level) <= 0.0 and y0 >= y1:
            if abs(y1 - y0) < 1e-6:
                return float(x[i])
            return float(x[i] + (level - y0) * (x[i + 1] - x[i]) / (y1 - y0))
    if np.any(y >= level) and np.any(y < level):
        idx = int(np.argmin(np.abs(y - level)))
        return float(x[idx])
    return None


def estimate_edge_width(gray, contour, bbox, max_points=160):
    """Примерная оценка ширины переходной границы семя/фон."""
    x, y, w, h = bbox
    if w < 4 or h < 4:
        return None
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 12:
        return None
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points, endpoint=False).astype(np.int32)
        pts = pts[idx]

    prev = np.roll(pts, -1, axis=0)
    nxt = np.roll(pts, 1, axis=0)
    tangents = prev - nxt
    norms = np.linalg.norm(tangents, axis=1)
    valid = norms > 1e-3
    if int(valid.sum()) < 12:
        return None
    pts = pts[valid]
    tangents = tangents[valid] / norms[valid, None]
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))

    img_h, img_w = gray.shape[:2]
    min_dim = max(3, min(w, h))
    check_offset = float(np.clip(min_dim * 0.20, 2.0, 5.0))

    def mean_at(offset):
        coords = pts + normals * offset
        xi = np.rint(coords[:, 0]).astype(np.int32)
        yi = np.rint(coords[:, 1]).astype(np.int32)
        ok = (xi >= 0) & (xi < img_w) & (yi >= 0) & (yi < img_h)
        if not ok.any():
            return np.nan
        return float(np.mean(gray[yi[ok], xi[ok]]))

    plus = mean_at(check_offset)
    minus = mean_at(-check_offset)
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

    inside_val = float(np.mean(inside_region))
    outside_val = float(np.mean(outside_region))
    contrast = inside_val - outside_val
    if abs(contrast) < 5.0:
        return None

    if contrast > 0:
        norm_profile = (profile_smooth - outside_val) / contrast
    else:
        norm_profile = (outside_val - profile_smooth) / (-contrast)
    norm_profile = np.clip(norm_profile, -0.5, 1.5)

    d80 = find_descending_crossing(distances, norm_profile, 0.80)
    d20 = find_descending_crossing(distances, norm_profile, 0.20)
    if d80 is None or d20 is None:
        return None

    width = abs(float(d20 - d80))
    if width < 0.3 or width > float(max_out - inside_start):
        return None
    return width


def classify_edge(edge_width, params):
    """Классифицирует резкость по ширине переходной границы."""
    if edge_width is None:
        return "unknown"
    if edge_width <= float(params["max_edge_width"]):
        return "focused"
    if edge_width >= float(params["blur_edge_width"]):
        return "blurry"
    return "transition"


def sort_items_reading_order(items, row_height_ratio=0.55):
    """
    Сортирует объекты как при чтении: слева направо внутри строки,
    строки сверху вниз.
    """
    if len(items) <= 1:
        return items
    decorated = []
    heights = []
    for item in items:
        x, y, w, h = item["bbox"]
        cx = float(x) + float(w) * 0.5
        cy = float(y) + float(h) * 0.5
        decorated.append((cy, cx, item))
        heights.append(float(h))

    decorated.sort(key=lambda it: it[0])
    med_h = float(np.median(np.array(heights, dtype=np.float32)))
    row_tolerance = max(3.0, med_h * float(row_height_ratio))

    rows = []
    current_row = [decorated[0]]
    prev_cy = decorated[0][0]
    for cy, cx, item in decorated[1:]:
        if cy - prev_cy > row_tolerance:
            rows.append(current_row)
            current_row = []
        current_row.append((cy, cx, item))
        prev_cy = cy
    if current_row:
        rows.append(current_row)

    sorted_items = []
    for row in rows:
        row.sort(key=lambda it: (it[1], it[0]))
        sorted_items.extend([item for _, _, item in row])
    return sorted_items


# ============================================================
# Основная обработка изображения
# ============================================================

def process_image(img_bgr, params):
    """
    Основная обработка изображения: сегментация, фильтрация, замеры семян.
    """
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

    if float(params["blur_edge_width"]) <= float(params["max_edge_width"]):
        params = dict(params)
        params["blur_edge_width"] = float(params["max_edge_width"]) + 0.1

    estimate_edge = bool(params["estimate_edge"]) or bool(params["filter_edge"])
    filter_edge = bool(params["filter_edge"])
    filter_outliers = bool(params.get("filter_outliers", True))

    img_h, img_w = gray.shape[:2]
    raw_candidates = []
    MIN_COMPACTNESS = 0.08

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

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
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

        bbox = (x, y, bw, bh)
        contrast = object_contrast(gray, cnt, bbox)
        if contrast < min_contrast:
            continue

        raw_candidates.append({
            "bbox": bbox,
            "area": float(area),
            "contour": cnt,
            "solidity": float(solidity),
            "extent": float(extent),
            "contrast": float(contrast),
            "compactness": float(compactness),
        })

    if filter_outliers:
        raw_candidates = filter_candidates_by_outliers(raw_candidates)

    items = []
    for cand in raw_candidates:
        edge_width = None
        if estimate_edge:
            edge_width = estimate_edge_width(gray, cand["contour"], cand["bbox"])
        status = classify_edge(edge_width, params)
        if filter_edge and status != "focused":
            continue

        rect = cv2.minAreaRect(cand["contour"])
        box = cv2.boxPoints(rect)
        side_a, side_b = rect[1]
        length_px = float(max(side_a, side_b))
        width_px = float(min(side_a, side_b))

        edge1 = box[1] - box[0]
        edge2 = box[2] - box[1]
        len1 = float(np.linalg.norm(edge1))
        len2 = float(np.linalg.norm(edge2))
        major_edge = edge1 if len1 >= len2 else edge2
        if (
            np.isfinite(major_edge[0])
            and np.isfinite(major_edge[1])
            and np.linalg.norm(major_edge) > 1e-6
        ):
            angle_deg = float(np.degrees(np.arctan2(major_edge[1], major_edge[0]))) % 180.0
        else:
            angle_deg = 0.0

        cand["rect"] = rect
        cand["box"] = box
        cand["center_x"] = float(rect[0][0])
        cand["center_y"] = float(rect[0][1])
        cand["length_px"] = length_px
        cand["width_px"] = width_px
        cand["angle_deg"] = angle_deg
        cand["edge_width"] = edge_width
        cand["status"] = status
        items.append(cand)

    items = sort_items_reading_order(items)
    for i, item in enumerate(items, 1):
        item["index"] = i

    # Аннотация
    annotated = img_bgr.copy()
    mm_per_px = params.get("mm_per_px")
    for item in items:
        i = item["index"]
        x, y, bw, bh = item["bbox"]
        color = STATUS_COLORS.get(item["status"], (255, 255, 255))

        cv2.drawContours(annotated, [item["contour"]], -1, color, 1, lineType=cv2.LINE_AA)
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)

        box = cv2.boxPoints(item["rect"]).astype(np.int32)
        cv2.polylines(annotated, [box], True, (255, 0, 255), 1, cv2.LINE_AA)

        label = f"{i}"
        if item["edge_width"] is not None:
            label += f" {item['edge_width']:.1f}px"
        if mm_per_px:
            label += (
                f" | {item['length_px'] * mm_per_px:.2f}"
                f"x{item['width_px'] * mm_per_px:.2f}mm"
            )
        else:
            label += f" | {item['length_px']:.0f}x{item['width_px']:.0f}px"

        ty = y - 6 if y - 20 > 0 else y + bh + 16
        cv2.putText(
            annotated, label, (x, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return {
        "items": items,
        "annotated": annotated,
        "mask": mask,
    }


# ============================================================
# Статистика размеров
# ============================================================

def compute_size_stats(items, mm_per_px=None):
    """
    Считает мин / среднее / макс по длине и ширине семян.
    Если мм/px не задан — значения остаются в пикселях.
    """
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


def format_size_stats(stats):
    """Форматирует статистику размеров для вывода."""
    if stats is None:
        return "Нет данных"
    unit = stats["unit"]
    decimals = 2 if unit == "мм" else 1

    def line(title, values):
        mn, avg, mx = values
        return (
            f"{title}, {unit}:  "
            f"мин {mn:.{decimals}f} / ср {avg:.{decimals}f} / макс {mx:.{decimals}f}"
        )

    return "\n".join([
        f"Семян: {stats['count']}",
        line("Длина", stats["length"]),
        line("Ширина", stats["width"]),
    ])
    
    
# ============================================================
# Вынесенные функции из main.py
# ============================================================

def get_param_definitions() -> tuple[tuple[str, str, float, float, float, Any, str], ...]:
    """
    Единый источник параметров обработки.

    Формат:
        (
            ключ,
            подпись для GUI,
            минимальное значение,
            максимальное значение,
            шаг,
            значение по умолчанию,
            тип ("int", "float", "odd")
        )
    """
    return (
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


def get_checkbox_definitions() -> tuple[tuple[str, str, bool], ...]:
    """
    Единый источник чекбоксов.

    Формат:
        (
            подпись для GUI,
            ключ параметра,
            значение по умолчанию
        )
    """
    return (
        ("Оценивать ширину границы", "estimate_edge", True),
        ("Оставлять только сфокусированные", "filter_edge", False),
        ("Фильтровать выбросы по площади", "filter_outliers", True),
        ("Не сохранять красные", "skip_blurry", True),
        ("Сохранять только сфокусированные", "save_only_focused", False),
    )


def get_default_params() -> dict[str, Any]:
    """
    Возвращает параметры по умолчанию.

    Используется и в GUI, и в пакетной обработке,
    чтобы не дублировать значения вручную.
    """
    params = {}

    for key, _label, _low, _high, _step, default, _kind in get_param_definitions():
        params[key] = default

    for _text, key, default in get_checkbox_definitions():
        params[key] = default

    params["mm_per_px"] = None

    return params


def parse_mm_per_px_text(text: str) -> Optional[float]:
    """
    Парсит строку с мм/px.

    Допустимы точка и запятая.
    Пустая строка означает None (замеры в пикселях).
    """
    text = (text or "").strip().replace(",", ".")

    if not text:
        return None

    try:
        value = float(text)
    except ValueError:
        raise ValueError("Масштаб мм/px должен быть числом, например 0.025")

    if not math.isfinite(value) or value <= 0:
        raise ValueError("Масштаб мм/px должен быть положительным числом, например 0.025")

    return value


def round_param_value(value: Any, low: float, high: float, step: float, kind: str) -> Any:
    """
    Округляет значение параметра до допустимого шага и диапазона.

    kind:
        "int"   — целое число;
        "odd"   — нечётное целое число;
        "float" — дробное число.
    """
    value = float(value)
    low_f = float(low)
    high_f = float(high)
    step_f = float(step)

    if kind == "float":
        if step_f <= 0:
            return max(low_f, min(high_f, value))

        decimals = 2 if step_f < 0.1 else 1
        value = round(round(value / step_f) * step_f, decimals)
        return max(low_f, min(high_f, float(value)))

    low_i = int(low_f)
    high_i = int(high_f)

    if step_f <= 0:
        value = int(round(value))
    else:
        value = int(round(value / step_f) * step_f)

    value = max(low_i, min(high_i, value))

    if kind == "odd" and value % 2 == 0:
        if value + 1 <= high_i:
            value += 1
        else:
            value -= 1

        value = max(low_i, value)

    return value


def format_param_label(title: str, value: Any, kind: str, step: float) -> str:
    """
    Форматирует подпись параметра для GUI.
    """
    step_f = float(step)

    if kind == "float":
        decimals = 2 if step_f < 0.1 else 1
        return f"{title}: {float(value):.{decimals}f}"

    return f"{title}: {int(round(float(value)))}"


def normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    """
    Приводит параметры к согласованному виду.

    - делает ядро размытия нечётным;
    - исправляет min/max area;
    - исправляет min/max ratio;
    - исправляет пороги фокуса/размытия.
    """
    params = {**get_default_params(), **(params or {})}

    params["blur_kernel"] = ensure_odd(int(float(params["blur_kernel"])), 1)

    if int(params["max_area"]) <= int(params["min_area"]):
        params["max_area"] = int(params["min_area"]) + 100

    if float(params["max_ratio"]) <= float(params["min_ratio"]):
        params["max_ratio"] = float(params["min_ratio"]) + 0.5

    if float(params["blur_edge_width"]) <= float(params["max_edge_width"]):
        params["blur_edge_width"] = float(params["max_edge_width"]) + 0.1

    return params


def try_calibrate_ruler(image_bgr: Any) -> tuple[Optional[float], Optional[str]]:
    """
    Безопасная обёртка для калибровки линейки.

    Возвращает:
        (mm_per_px, None) при успехе;
        (None, текст ошибки) при ошибке.
    """
    try:
        mm_per_px = calibrate_ruler(image_bgr, debug=False)

        if mm_per_px is None:
            return None, "Калибровка вернула пустое значение."

        mm_per_px = float(mm_per_px)

        if not math.isfinite(mm_per_px) or mm_per_px <= 0:
            return None, "Калибровка вернула некорректный мм/px."

        return mm_per_px, None

    except Exception as exc:
        return None, str(exc)


def format_calibration_message(mm_per_px: float) -> str:
    """
    Единый формат сообщения о калибровке.
    """
    mm = float(mm_per_px)

    if mm <= 0:
        return "0.000000 мм/px"

    return f"{mm:.6f} мм/px ({1.0 / mm:.2f} px/mm)"


def count_seed_statuses(items: list[dict[str, Any]]) -> dict[str, int]:
    """
    Считает статусы семян.
    """
    items = list(items or [])

    focused = sum(1 for it in items if it.get("status") == "focused")
    blurry = sum(1 for it in items if it.get("status") == "blurry")
    unknown = sum(1 for it in items if it.get("status") == "unknown")

    return {
        "total": len(items),
        "focused": focused,
        "blurry": blurry,
        "unknown": unknown,
    }


def format_seed_status_message(counts: dict[str, int]) -> str:
    """
    Форматирует строку статуса по количеству семян.
    """
    return (
        f"Найдено: {counts.get('total', 0)}; "
        f"фокус: {counts.get('focused', 0)}; "
        f"размыто: {counts.get('blurry', 0)}; "
        f"неизвестно: {counts.get('unknown', 0)}."
    )


def select_items_for_export(
    items: list[dict[str, Any]],
    *,
    save_only_focused: bool = False,
    skip_blurry: bool = True,
) -> list[dict[str, Any]]:
    """
    Выбирает объекты для сохранения по флагам.

    Если включён save_only_focused, skip_blurry игнорируется.
    """
    items = list(items or [])

    if save_only_focused:
        return [it for it in items if it.get("status") == "focused"]

    if skip_blurry:
        return [it for it in items if it.get("status") != "blurry"]

    return items


def assign_unique_seed_indices(
    items: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], int]]:
    """
    Сортирует объекты по индексу и назначает каждому уникальный индекс.

    Возвращает список кортежей:
        (item, seed_index)
    """
    def sort_key(item: dict[str, Any]) -> int:
        try:
            return int(item.get("index", 0))
        except Exception:
            return 0

    used_indices = set()
    result = []

    for item in sorted(items, key=sort_key):
        try:
            seed_index = int(item.get("index", 0))
        except Exception:
            seed_index = 0

        if seed_index <= 0:
            seed_index = max(used_indices) + 1 if used_indices else 1

        while seed_index in used_indices:
            seed_index += 1

        used_indices.add(seed_index)
        result.append((item, seed_index))

    return result


def compute_crop_box(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int, Any],
    padding: int = 0,
) -> tuple[int, int, int, int]:
    """
    Вычисляет координаты кропа с отступом, не выходя за границы изображения.

    Возвращает:
        (x1, y1, x2, y2)
    """
    h, w = image_shape[:2]
    x, y, bw, bh = map(int, bbox)

    pad = max(0, int(padding))

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(int(w), x + bw + pad)
    y2 = min(int(h), y + bh + pad)

    return x1, y1, x2, y2


def _px_to_mm(value_px: Any, mm_per_px: Optional[float], decimals: int = 4) -> str:
    """
    Безопасно переводит пиксели в мм и форматирует строку.
    """
    if value_px is None or mm_per_px is None:
        return ""

    try:
        return fmt_float(float(value_px) * float(mm_per_px), decimals)
    except (TypeError, ValueError):
        return ""


def build_seed_csv_row(
    item: dict[str, Any],
    file_index: str,
    seed_index: int,
    mm_per_px: Optional[float],
) -> dict[str, Any]:
    """
    Собирает одну строку для seeds.csv.
    """
    length_px = item.get("length_px", None)
    width_px = item.get("width_px", None)

    return {
        "file_index": file_index,
        "index": seed_index,
        "status": str(item.get("status", "unknown")),
        "edge_width_px": fmt_float(item.get("edge_width"), 3),
        "length_px": fmt_float(length_px, 3),
        "width_px": fmt_float(width_px, 3),
        "angle_deg": fmt_float(item.get("angle_deg", item.get("angle", None)), 3),
        "area_px": fmt_float(item.get("area"), 1),
        "length_mm": _px_to_mm(length_px, mm_per_px, 4),
        "width_mm": _px_to_mm(width_px, mm_per_px, 4),
        "mm_per_px": fmt_float(mm_per_px, 8) if mm_per_px is not None else "",
    }


def export_seed_items(
    image_bgr: Any,
    items: list[dict[str, Any]],
    out_dir: str,
    source_name: str,
    *,
    padding: int = 0,
    mm_per_px: Optional[float] = None,
) -> dict[str, Any]:
    """
    Единая функция экспорта семян:

    - создаёт папку;
    - сохраняет кропы;
    - сохраняет контуры;
    - удаляет старые файлы группы;
    - обновляет seeds.csv.

    Возвращает словарь:
        {
            "saved": int,
            "selected": int,
            "contour_errors": int,
            "saved_file_indexes": list[str],
            "csv_error": Optional[str],
        }
    """
    items = list(items or [])

    summary = {
        "saved": 0,
        "selected": len(items),
        "contour_errors": 0,
        "saved_file_indexes": [],
        "csv_error": None,
    }

    if not items or image_bgr is None:
        return summary

    os.makedirs(out_dir, exist_ok=True)

    pad = max(0, int(padding))
    saved_rows = []

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
        filename = f"{file_index}.jpg"
        full_path = os.path.join(out_dir, filename)

        if not imwrite_unicode(
            full_path,
            crop,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        ):
            continue

        contour_path = os.path.join(out_dir, f"{file_index}_contour.json")

        try:
            save_contour_json(
                path=contour_path,
                file_index=file_index,
                seed_index=seed_index,
                contour=item.get("contour"),
                crop_origin=(x1, y1),
                crop_shape=crop.shape,
                bbox=bbox,
                padding=pad,
                source_name=source_name,
            )
        except Exception:
            summary["contour_errors"] += 1

        summary["saved"] += 1
        summary["saved_file_indexes"].append(file_index)

        saved_rows.append(
            build_seed_csv_row(
                item=item,
                file_index=file_index,
                seed_index=seed_index,
                mm_per_px=mm_per_px,
            )
        )

    try:
        remove_group_files_except(out_dir, source_name, summary["saved_file_indexes"])
    except Exception:
        pass

    csv_path = os.path.join(out_dir, "seeds.csv")

    try:
        update_seeds_csv(csv_path, saved_rows, source_name)
    except Exception as exc:
        summary["csv_error"] = str(exc)

    return summary


def format_export_message(
    saved: int,
    selected: int,
    source_name: str,
    out_dir: str,
    contour_errors: int = 0,
) -> str:
    """
    Форматирует сообщение об экспорте.
    """
    msg = (
        f"Сохранено {saved} из {selected} фрагментов "
        f"для '{source_name}' в: {out_dir}"
    )

    if contour_errors:
        msg += f", ошибки сохранения контуров: {contour_errors}"

    return msg


def clamp_display_scale(
    scale: float,
    width: int,
    height: int,
    min_scale: float,
    max_scale: float,
    max_display_dim: float,
) -> float:
    """
    Ограничивает масштаб отображения изображения.
    """
    width = max(1, int(width))
    height = max(1, int(height))

    max_scale_by_dim = float(max_display_dim) / float(max(width, height))
    max_allowed = min(float(max_scale), max_scale_by_dim)

    return max(float(min_scale), min(max_allowed, float(scale)))


def calculate_fit_scale(
    width: float,
    height: float,
    viewport_width: float,
    viewport_height: float,
    mode: str,
    current_scale: float = 1.0,
) -> float:
    """
    Считает масштаб для режимов "по ширине", "по окну" или ручной режим.
    """
    width = max(1.0, float(width))
    height = max(1.0, float(height))
    viewport_width = max(1.0, float(viewport_width))
    viewport_height = max(1.0, float(viewport_height))

    if mode == "width":
        return viewport_width / width

    if mode == "window":
        return min(viewport_width / width, viewport_height / height)

    return float(current_scale)