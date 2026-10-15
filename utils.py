"""
Обработчики изображения:
- чтение/запись с поддержкой кириллицы;
- сегментация;
- фильтрация объектов;
- оценка фокуса;
- замеры семян;
- статистика размеров;
- обновление параметров в CSV;
- сохранение контуров семян в JSON.
"""

import os
import re
import csv
import json

import cv2
import numpy as np


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


# ------------------------------------------------------------
# Ввод/вывод
# ------------------------------------------------------------

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


def to_uint8(img):
    if img.dtype == np.uint8:
        return img

    dst = np.empty_like(img)
    cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)

    return dst.astype(np.uint8)


def ensure_odd(value, min_value=1):
    value = int(round(float(value)))

    if value < min_value:
        value = min_value

    if value % 2 == 0:
        value += 1

    return value


def sanitize_filename(name):
    """
    Делает имя файла/префикс безопасным для Windows/Linux.
    Используется для формирования имён вида:
        2_1_001.jpg
        2_1_001_contour.json
    """
    name = os.path.basename(str(name))
    name = re.sub(r'[\\/:*?"<>|\s]+', "_", name)
    name = name.strip("._ ")

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

    Группа определяется по префиксу:
        <group_prefix>_<digits>.jpg
        <group_prefix>_<digits>.png
        <group_prefix>_<digits>_contour.json

    keep_file_indexes:
        список/множество file_index, которые нужно оставить,
        например ["2_1_001", "2_1_002"]
    """
    if not group_prefix or not os.path.isdir(dir_path):
        return

    keep = set(str(x) for x in keep_file_indexes)

    img_pattern = re.compile(
        r"^" + re.escape(group_prefix) + r"_(\d+)\.(jpe?g|png)$",
        re.IGNORECASE
    )

    json_pattern = re.compile(
        r"^" + re.escape(group_prefix) + r"_(\d+)_contour\.json$",
        re.IGNORECASE
    )

    for name in os.listdir(dir_path):
        file_index = None
        remove = False

        m = img_pattern.match(name)
        if m:
            digits = m.group(1)
            ext = m.group(2).lower()
            file_index = f"{group_prefix}_{digits}"

            # Новый формат хранения — JPG.
            # Старые PNG/JPEG той же группы удаляем всегда.
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

    Логика:
    - читает существующий CSV, если он есть;
    - удаляет строки, у которых file_index относится к текущему файлу:
      <group_prefix>_<digits>;
    - добавляет новые строки текущего файла;
    - сохраняет CSV в utf-8-sig.

    Так достигается:
    - добавление данных для новых фотографий;
    - замена данных при повторной обработке той же фотографии.
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

    # Удаляем старые строки текущего исходного файла.
    group_pattern = re.compile(r"^" + re.escape(group_prefix) + r"_\d+$")

    kept_rows = []

    for row in existing_rows:
        file_index = str(row.get("file_index", "")).strip()

        if file_index and group_pattern.match(file_index):
            continue

        kept_rows.append(row)

    # Собираем итоговые поля.
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


# ------------------------------------------------------------
# Сегментация
# ------------------------------------------------------------

def make_mask(gray, params):
    """
    Создаёт бинарную маску семян.
    Используется только глобальный порог Otsu.
    """
    k = ensure_odd(params["blur_kernel"], 1)
    blurred = cv2.GaussianBlur(gray, (k, k), 0)

    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Если маска получилась инвертированной, исправляем.
    # Белый класс должен быть светлее чёрного класса на исходном grayscale.
    white_count = int(np.count_nonzero(mask))

    if 0 < white_count < mask.size:
        white_mask = mask > 0

        mean_white = float(np.mean(gray[white_mask]))
        mean_black = float(np.mean(gray[~white_mask]))

        if mean_white < mean_black or white_count > mask.size * 0.6:
            mask = cv2.bitwise_not(mask)

    elif white_count == mask.size:
        # Если всё стало белым, маска почти наверняка испорчена.
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
    """
    Заполняет внешние контуры.
    Это помогает закрыть дырки внутри семян.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled = np.zeros_like(mask)

    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)

    return filled


def object_contrast(gray, contour, bbox):
    """
    Примитивная оценка контраста объекта относительно локального фона.
    """
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

    cv2.drawContours(
        mask,
        [shifted.astype(np.int32)],
        -1,
        255,
        thickness=cv2.FILLED
    )

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
    Полезно против одиночного мусора.
    """
    if len(candidates) < 6:
        return candidates

    areas = np.array([c["area"] for c in candidates], dtype=np.float32)
    dims = np.array([min(c["bbox"][2], c["bbox"][3]) for c in candidates], dtype=np.float32)

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
    """
    Ищет точку пересечения уровня при спадании значения.
    Используется для оценки границы 80% -> 20%.
    """
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
    """
    Примерная оценка ширины переходной границы семя/фон.
    """
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

    Допускается, что объекты в одной строке могут немного отличаться
    по вертикали. Для этого используется допуск, зависящий от
    медианной высоты объекта.
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
        # Если вертикальный разрыв больше допуска — начинается новая строка.
        if cy - prev_cy > row_tolerance:
            rows.append(current_row)
            current_row = []

        current_row.append((cy, cx, item))
        prev_cy = cy

    if current_row:
        rows.append(current_row)

    sorted_items = []

    for row in rows:
        # Внутри строки сортируем по горизонтали, слева направо.
        row.sort(key=lambda it: (it[1], it[0]))
        sorted_items.extend([item for _, _, item in row])

    return sorted_items


def process_image(img_bgr, params):
    """
    Основная обработка изображения.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = to_uint8(gray)
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

        # Замер: минимальный повёрнутый прямоугольник вокруг контура.
        # Большая сторона — длина, меньшая — ширина (в пикселях).
        rect = cv2.minAreaRect(cand["contour"])
        box = cv2.boxPoints(rect)

        side_a, side_b = rect[1]

        length_px = float(max(side_a, side_b))
        width_px = float(min(side_a, side_b))

        # Угол основной длинной стороны повёрнутого прямоугольника.
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

    # Объекты сначала группируются в строки с допуском по вертикали,
    # а затем сортируются слева направо внутри каждой строки.
    items = sort_items_reading_order(items)

    # Проставляем стабильный номер семечки на фото.
    for i, item in enumerate(items, 1):
        item["index"] = i

    annotated = img_bgr.copy()
    mm_per_px = params.get("mm_per_px")

    for item in items:
        i = item["index"]

        x, y, bw, bh = item["bbox"]
        color = STATUS_COLORS.get(item["status"], (255, 255, 255))

        cv2.drawContours(
            annotated,
            [item["contour"]],
            -1,
            color,
            1,
            lineType=cv2.LINE_AA
        )

        cv2.rectangle(
            annotated,
            (x, y),
            (x + bw, y + bh),
            color,
            2
        )

        # Повёрнутый прямоугольник, по которому считаются длина и ширина.
        box = cv2.boxPoints(item["rect"]).astype(np.int32)

        cv2.polylines(
            annotated,
            [box],
            True,
            (255, 0, 255),
            1,
            cv2.LINE_AA
        )

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
            annotated,
            label,
            (x, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return {
        "items": items,
        "annotated": annotated,
        "mask": mask,
    }


# ------------------------------------------------------------
# Статистика размеров
# ------------------------------------------------------------

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