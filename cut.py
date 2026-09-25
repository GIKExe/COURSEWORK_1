import os
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox

import numpy as np
import cv2
from PIL import Image, ImageTk

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.ANTIALIAS


STATUS_COLORS = {
    "focused": (0, 200, 0),       # зелёный
    "transition": (0, 220, 255),  # жёлтый
    "blurry": (0, 0, 255),        # красный
    "unknown": (80, 160, 255),    # оранжевый
}


TOOLTIPS = {
    "thresh_mode": (
        "Способ бинаризации.\n"
        "Otsu — глобальный порог, хорош при равномерном тёмном фоне.\n"
        "Adaptive — локальный порог, лучше при неравномерном освещении."
    ),
    "blur_kernel": (
        "Размер ядра гауссова размытия перед порогом.\n"
        "Убирает шум, но слишком большое значение ухудшает границы."
    ),
    "block_size": (
        "Размер локальной области для Adaptive.\n"
        "Должен быть нечётным.\n"
        "Больше значение — устойчивее к градиенту фона."
    ),
    "adaptive_c": (
        "Константа, вычитаемая из локального среднего в Adaptive.\n"
        "Положительное значение — строже, отрицательное — больше объектов."
    ),
    "close_kernel": (
        "Морфологическое закрытие.\n"
        "Соединяет разрывы и заполняет мелкие дырки внутри семян.\n"
        "Слишком большое значение может склеить соседние семена."
    ),
    "open_kernel": (
        "Морфологическое открытие.\n"
        "Удаляет мелкий шум.\n"
        "Слишком большое значение может удалить мелкие семена."
    ),
    "min_area": "Минимальная площадь объекта в пикселях. Всё меньше считается шумом.",
    "max_area": "Максимальная площадь объекта. Слишком крупные объекты отбрасываются.",
    "min_ratio": "Минимальная вытянутость: большая сторона бокса / меньшая сторона.",
    "max_ratio": "Максимальная вытянутость. Очень вытянутые объекты обычно не семена.",
    "min_solidity": (
        "Solidity = площадь контура / площадь выпуклой оболочки.\n"
        "Низкое значение — рваный/неполный объект."
    ),
    "min_extent": (
        "Extent = площадь контура / площадь прямоугольника.\n"
        "Низкое значение — объект занимает мало своего бокса."
    ),
    "min_contrast": (
        "Минимальная разница яркости между семнем и фоном.\n"
        "Помогает не брать слабый мусор.\n"
        "Если семена теряются — уменьшите значение."
    ),
    "max_edge_width": (
        "Если оценка ширины границы не больше этого значения,\n"
        "семя считается сфокусированным и помечается зелёным."
    ),
    "blur_edge_width": (
        "Если оценка ширины границы не меньше этого значения,\n"
        "семя считается размытым и помечается красным."
    ),
    "padding": "Отступ вокруг прямоугольника при сохранении фрагментов.",
    "estimate_edge": (
        "Вычислять ли ширину переходной границы.\n"
        "Если выключить, статус будет неизвестным, но обработка быстрее."
    ),
    "filter_edge": "Оставлять при обработке только зелёные сфокусированные семена.",
    "filter_outliers": (
        "Удалять объекты, которые сильно отличаются от медианной площади и размера остальных семян."
    ),
    "skip_blurry": (
        "При сохранении пропускать красные объекты.\n"
        "Помогает от мусора, но может пропустить сильно размытые семена."
    ),
    "save_only_focused": "При сохранении сохранять только зелёные объекты.",
}


def setup_dpi_awareness():
    """
    Пытаемся сделать приложение DPI-осведомлённым в Windows,
    чтобы при 125%/150% интерфейс не был мыльным.
    """
    if os.name != "nt":
        return

    try:
        import ctypes

        # Per-Monitor Aware (Windows 8.1+)
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            # Более старый вариант
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


class ToolTip:
    """Простой всплывающий tooltip для Tkinter/ttk."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None

        widget.bind("<Enter>", self.showtip, add="+")
        widget.bind("<Leave>", self.hidetip, add="+")
        widget.bind("<ButtonPress>", self.hidetip, add="+")

    def showtip(self, event=None):
        if self.tipwindow or not self.text:
            return

        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        try:
            tw.attributes("-topmost", True)
        except Exception:
            pass

        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#ffffe0",
            foreground="#000000",
            relief="solid",
            borderwidth=1,
            font=tkfont.nametofont("TkDefaultFont"),
            wraplength=320,
        )
        label.pack(ipadx=4, ipady=2)

    def hidetip(self, event=None):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


# ------------------------------------------------------------
# Ввод/вывод
# ------------------------------------------------------------

def imread_unicode(path):
    """Читает изображение, включая пути с кириллицей."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    """Сохраняет изображение, включая пути с кириллицей."""
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".png"
        path += ext

    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
        return True
    return False


def to_uint8(img):
    if img.dtype == np.uint8:
        return img
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)


def ensure_odd(value, min_value=1):
    value = int(round(value))
    if value < min_value:
        value = min_value
    if value % 2 == 0:
        value += 1
    return value


# ------------------------------------------------------------
# Сегментация
# ------------------------------------------------------------

def make_mask(gray, params):
    """
    Создаёт бинарную маску семян.
    """
    k = ensure_odd(params["blur_kernel"], 1)
    blurred = cv2.GaussianBlur(gray, (k, k), 0)

    if params["thresh_mode"] == "Adaptive":
        block = ensure_odd(params["block_size"], 3)
        c = int(params["adaptive_c"])

        # ПРАВИЛЬНЫЙ флаг: cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        mask = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            c
        )
    else:
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

        cand["edge_width"] = edge_width
        cand["status"] = status
        items.append(cand)

    items.sort(key=lambda it: (it["bbox"][1], it["bbox"][0]))

    annotated = img_bgr.copy()

    for i, item in enumerate(items, 1):
        x, y, bw, bh = item["bbox"]
        color = STATUS_COLORS.get(item["status"], (255, 255, 255))

        cv2.drawContours(annotated, [item["contour"]], -1, color, 1, lineType=cv2.LINE_AA)
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)

        label = f"{i}"
        if item["edge_width"] is not None:
            label += f" {item['edge_width']:.1f}px"

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
# GUI
# ------------------------------------------------------------

class FlaxSeedApp:
    MIN_SCALE = 0.02
    MAX_SCALE = 16.0
    MAX_DISPLAY_DIM = 12000

    def __init__(self, root):
        self.root = root
        self.original_bgr = None
        self.result_bgr = None
        self.seed_items = []
        self.image_path = None
        self.tk_image = None

        self.current_view = "original"  # original / result
        self.display_scale = 1.0
        self.fit_mode = "width"         # width / window / manual
        self._last_canvas_size = (0, 0)

        self.status_var = tk.StringVar(
            value="Откройте фото, подберите параметры и нажмите «Обработать»."
        )

        self._init_vars()
        self._build_ui()

    def _init_vars(self):
        self.vars = {}

        # key, label, low, high, step, default, kind
        self.param_defs = [
            ("blur_kernel", "Размытие, ядро", 1, 31, 2, 5, "odd"),
            ("block_size", "Adaptive block size", 3, 199, 2, 51, "odd"),
            ("adaptive_c", "Adaptive C", -30, 30, 1, 5, "int"),
            ("close_kernel", "Close, морфология", 1, 31, 1, 7, "int"),
            ("open_kernel", "Open, морфология", 1, 31, 1, 3, "int"),
            ("min_area", "Мин. площадь, px", 10, 100000, 10, 200, "int"),
            ("max_area", "Макс. площадь, px", 100, 300000, 100, 80000, "int"),
            ("min_ratio", "Мин. вытянутость", 1.0, 10.0, 0.1, 1.0, "float"),
            ("max_ratio", "Макс. вытянутость", 1.0, 15.0, 0.1, 4.5, "float"),
            ("min_solidity", "Мин. solidity", 0.0, 1.0, 0.01, 0.60, "float"),
            ("min_extent", "Мин. extent", 0.0, 1.0, 0.01, 0.30, "float"),
            ("min_contrast", "Мин. контраст", 0, 100, 1, 4, "int"),
            ("max_edge_width", "Фокус: граница <=, px", 0.5, 15.0, 0.1, 4.5, "float"),
            ("blur_edge_width", "Размыто: граница >=, px", 1.0, 25.0, 0.1, 7.0, "float"),
            ("padding", "Отступ при сохранении, px", 0, 50, 1, 4, "int"),
        ]

        self.param_kinds = {}
        self.param_titles = {}
        self.param_lows = {}
        self.param_highs = {}
        self.param_steps = {}
        self.scale_value_labels = {}

        for key, label, low, high, step, default, kind in self.param_defs:
            if kind == "float":
                self.vars[key] = tk.DoubleVar(value=default)
            else:
                self.vars[key] = tk.IntVar(value=default)

            self.param_kinds[key] = kind
            self.param_titles[key] = label
            self.param_lows[key] = low
            self.param_highs[key] = high
            self.param_steps[key] = step

        self.vars["thresh_mode"] = tk.StringVar(value="Otsu")
        self.vars["estimate_edge"] = tk.BooleanVar(value=True)
        self.vars["filter_edge"] = tk.BooleanVar(value=False)
        self.vars["filter_outliers"] = tk.BooleanVar(value=True)
        self.vars["skip_blurry"] = tk.BooleanVar(value=True)
        self.vars["save_only_focused"] = tk.BooleanVar(value=False)

    def _build_ui(self):
        self.root.title("Выделение и нарезка семян льна")
        self.root.geometry("1450x850")

        toolbar = ttk.Frame(self.root, padding=5)
        toolbar.pack(fill=tk.X)

        buttons = [
            ("Открыть фото", self.open_image),
            ("Обработать", self.process),
            ("Исходное", self.show_original),
            ("Результат", self.show_result),
            ("Сохранить нарезку", self.save_crops),
            ("Сохранить результат", self.save_result_image),
        ]

        for text, cmd in buttons:
            ttk.Button(toolbar, text=text, command=cmd).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient="vertical").pack(side=tk.LEFT, fill="y", padx=6)

        zoom_buttons = [
            ("+", self.zoom_in),
            ("-", self.zoom_out),
            ("1:1", self.zoom_actual),
            ("По ширине", self.zoom_fit_width),
            ("По окну", self.zoom_fit_window),
        ]

        for text, cmd in zoom_buttons:
            ttk.Button(toolbar, text=text, command=cmd).pack(side=tk.LEFT, padx=2)

        self.zoom_label = ttk.Label(toolbar, text="100%", width=6, anchor="center")
        self.zoom_label.pack(side=tk.LEFT, padx=4)

        main = ttk.Frame(self.root, padding=5)
        main.pack(fill=tk.BOTH, expand=True)

        left_outer = ttk.Frame(main)
        left_outer.pack(side=tk.LEFT, fill=tk.Y)

        left_canvas = tk.Canvas(left_outer, width=370, highlightthickness=0)
        left_scroll = ttk.Scrollbar(left_outer, orient=tk.VERTICAL, command=left_canvas.yview)

        self.param_frame = ttk.Frame(left_canvas)
        win_id = left_canvas.create_window((0, 0), window=self.param_frame, anchor="nw")

        self.param_frame.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )

        left_canvas.bind(
            "<Configure>",
            lambda e, wid=win_id: left_canvas.itemconfigure(wid, width=e.width)
        )

        left_canvas.configure(yscrollcommand=left_scroll.set)

        left_canvas.pack(side=tk.LEFT, fill=tk.Y)
        left_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._build_param_panel()

        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(right, bg="#202020", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        vbar = ttk.Scrollbar(right, orient="vertical", command=self.canvas.yview)
        vbar.grid(row=0, column=1, sticky="ns")

        hbar = ttk.Scrollbar(right, orient="horizontal", command=self.canvas.xview)
        hbar.grid(row=1, column=0, sticky="ew")

        self.canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Зум: Ctrl + колесо
        self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)
        self.canvas.bind("<Control-Button-4>", lambda e: self._zoom_factor(1.25, e))
        self.canvas.bind("<Control-Button-5>", lambda e: self._zoom_factor(0.8, e))

        # Обычное колесо — прокрутка
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)
        self.canvas.bind("<Button-5>", self._on_mousewheel)

        # Панорамирование левой кнопкой мыши
        self.canvas.bind("<ButtonPress-1>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B1-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))

        status = ttk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            padding=(5, 2)
        )
        status.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_param_panel(self):
        pf = self.param_frame

        ttk.Label(pf, text="Порог:").pack(anchor="w", padx=4, pady=(4, 0))

        combo = ttk.Combobox(
            pf,
            textvariable=self.vars["thresh_mode"],
            values=["Otsu", "Adaptive"],
            state="readonly",
            width=18
        )
        combo.pack(fill=tk.X, padx=4)
        ToolTip(combo, TOOLTIPS.get("thresh_mode", ""))

        checks = [
            ("Оценивать ширину границы", "estimate_edge"),
            ("Оставлять только сфокусированные", "filter_edge"),
            ("Фильтровать выбросы по площади", "filter_outliers"),
            ("Не сохранять красные", "skip_blurry"),
            ("Сохранять только сфокусированные", "save_only_focused"),
        ]

        for text, key in checks:
            cb = ttk.Checkbutton(pf, text=text, variable=self.vars[key])
            cb.pack(anchor="w", padx=4, pady=1)
            ToolTip(cb, TOOLTIPS.get(key, ""))

        ttk.Separator(pf, orient="horizontal").pack(fill="x", pady=4)

        for key, label, low, high, step, default, kind in self.param_defs:
            frame = ttk.Frame(pf)
            frame.pack(fill=tk.X, padx=4, pady=2)

            lbl = ttk.Label(frame, text=self._format_param(key))
            lbl.pack(anchor="w")
            self.scale_value_labels[key] = lbl

            scale = ttk.Scale(
                frame,
                from_=low,
                to=high,
                orient=tk.HORIZONTAL,
                command=lambda value, k=key: self._on_scale(k, value)
            )
            scale.pack(fill=tk.X)
            scale.set(self.vars[key].get())

            tooltip_text = TOOLTIPS.get(key, "")
            if tooltip_text:
                ToolTip(lbl, tooltip_text)
                ToolTip(scale, tooltip_text)

    def _format_param(self, key):
        val = self.vars[key].get()
        step = self.param_steps.get(key, 1)

        if self.param_kinds.get(key) == "float":
            decimals = 2 if step < 0.1 else 1
            return f"{self.param_titles[key]}: {float(val):.{decimals}f}"

        return f"{self.param_titles[key]}: {int(round(float(val)))}"

    def _round_param(self, key, raw_value):
        kind = self.param_kinds[key]
        low = self.param_lows[key]
        high = self.param_highs[key]
        step = self.param_steps[key]
        value = float(raw_value)

        if kind == "float":
            decimals = 2 if step < 0.1 else 1
            value = round(round(value / step) * step, decimals)
            return max(float(low), min(float(high), float(value)))

        value = int(round(value / step) * step)
        value = max(int(low), min(int(high), value))

        if kind == "odd" and value % 2 == 0:
            if value + 1 <= int(high):
                value += 1
            else:
                value -= 1

        if kind == "odd":
            value = max(int(low), value)

        return value

    def _on_scale(self, key, raw_value):
        value = self._round_param(key, raw_value)
        self.vars[key].set(value)

        if key in self.scale_value_labels:
            self.scale_value_labels[key].config(text=self._format_param(key))

    def get_params(self):
        params = {}

        for key, var in self.vars.items():
            params[key] = var.get()

        params["blur_kernel"] = ensure_odd(params["blur_kernel"], 1)
        params["block_size"] = ensure_odd(params["block_size"], 3)

        if int(params["max_area"]) <= int(params["min_area"]):
            params["max_area"] = int(params["min_area"]) + 100

        if float(params["max_ratio"]) <= float(params["min_ratio"]):
            params["max_ratio"] = float(params["min_ratio"]) + 0.5

        if float(params["blur_edge_width"]) <= float(params["max_edge_width"]):
            params["blur_edge_width"] = float(params["max_edge_width"]) + 0.1

        return params

    # ------------------------------------------------------------
    # View / zoom
    # ------------------------------------------------------------

    def _current_image(self):
        if self.current_view == "result":
            return self.result_bgr
        return self.original_bgr

    def _clamp_scale(self, scale, img):
        h, w = img.shape[:2]
        max_scale_by_dim = self.MAX_DISPLAY_DIM / float(max(w, h))
        max_scale = min(self.MAX_SCALE, max_scale_by_dim)
        return max(self.MIN_SCALE, min(max_scale, float(scale)))

    def _scroll_size(self):
        try:
            sr = self.canvas.cget("scrollregion")
            if sr:
                parts = sr.split()
                if len(parts) == 4:
                    return int(parts[2]), int(parts[3])
        except Exception:
            pass

        return self.canvas.winfo_width(), self.canvas.winfo_height()

    def _redraw_view(self):
        self.canvas.delete("all")

        img = self._current_image()
        if img is None:
            self.zoom_label.config(text="100%")
            return

        h, w = img.shape[:2]
        scale = self._clamp_scale(self.display_scale, img)
        self.display_scale = scale

        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        if new_w != w or new_h != h:
            pil_img = pil_img.resize((new_w, new_h), RESAMPLE)

        self.tk_image = ImageTk.PhotoImage(pil_img)

        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.canvas.configure(scrollregion=(0, 0, new_w, new_h))

        self.zoom_label.config(text=f"{int(round(scale * 100))}%")

    def _apply_fit(self):
        img = self._current_image()
        if img is None:
            return

        h, w = img.shape[:2]

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        if cw <= 2 or ch <= 2:
            self.display_scale = self._clamp_scale(1.0, img)
            self._redraw_view()
            return

        if self.fit_mode == "width":
            scale = cw / float(w)
        elif self.fit_mode == "window":
            scale = min(cw / float(w), ch / float(h))
        else:
            scale = self.display_scale

        self.display_scale = self._clamp_scale(scale, img)
        self._redraw_view()

    def _refresh_view_after_change(self):
        self.canvas.update_idletasks()

        if self.fit_mode in ("width", "window"):
            self._apply_fit()
        else:
            self._redraw_view()

    def _on_canvas_configure(self, event):
        if self._current_image() is None:
            return

        size = (event.width, event.height)
        if size == self._last_canvas_size:
            return

        self._last_canvas_size = size

        if self.fit_mode in ("width", "window"):
            self._apply_fit()

    def _zoom_factor(self, factor, event=None):
        img = self._current_image()
        if img is None:
            return

        old_scale = self.display_scale
        new_scale = old_scale * float(factor)
        new_scale = self._clamp_scale(new_scale, img)

        if abs(new_scale - old_scale) < 1e-6:
            return

        if event is not None:
            vx = event.x
            vy = event.y
        else:
            vx = max(1, self.canvas.winfo_width() // 2)
            vy = max(1, self.canvas.winfo_height() // 2)

        canvas_x = self.canvas.canvasx(vx)
        canvas_y = self.canvas.canvasy(vy)

        img_x = canvas_x / old_scale if old_scale > 0 else 0.0
        img_y = canvas_y / old_scale if old_scale > 0 else 0.0

        self.display_scale = new_scale
        self.fit_mode = "manual"

        self._redraw_view()
        self.canvas.update_idletasks()

        self._scroll_to_image_point(img_x, img_y, vx, vy)

    def _scroll_to_image_point(self, img_x, img_y, viewport_x, viewport_y):
        new_x = img_x * self.display_scale
        new_y = img_y * self.display_scale

        scroll_w, scroll_h = self._scroll_size()
        view_w = self.canvas.winfo_width()
        view_h = self.canvas.winfo_height()

        if scroll_w > view_w:
            frac = (new_x - viewport_x) / float(scroll_w)
            self.canvas.xview_moveto(max(0.0, min(1.0, frac)))
        else:
            self.canvas.xview_moveto(0.0)

        if scroll_h > view_h:
            frac = (new_y - viewport_y) / float(scroll_h)
            self.canvas.yview_moveto(max(0.0, min(1.0, frac)))
        else:
            self.canvas.yview_moveto(0.0)

    def zoom_in(self):
        self._zoom_factor(1.25)

    def zoom_out(self):
        self._zoom_factor(0.8)

    def zoom_actual(self):
        img = self._current_image()
        if img is None:
            return

        self.fit_mode = "manual"
        self.display_scale = self._clamp_scale(1.0, img)
        self._redraw_view()

    def zoom_fit_width(self):
        self.fit_mode = "width"
        self._apply_fit()

    def zoom_fit_window(self):
        self.fit_mode = "window"
        self._apply_fit()

    def _on_ctrl_mousewheel(self, event):
        factor = 1.25 if event.delta > 0 else 0.8
        self._zoom_factor(factor, event)

    def _on_mousewheel(self, event):
        # Если нажат Ctrl, зум обработается отдельным биндом.
        if getattr(event, "state", 0) & 0x0004:
            return

        delta = 0

        if getattr(event, "delta", 0) != 0:
            delta = -1 if event.delta > 0 else 1
        elif getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1

        if delta == 0:
            return

        # Shift — горизонтальная прокрутка, иначе вертикальная.
        if getattr(event, "state", 0) & 0x0001:
            self.canvas.xview_scroll(delta, "units")
        else:
            self.canvas.yview_scroll(delta, "units")

    # ------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Выберите фото семян",
            filetypes=[
                ("Изображения", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("Все файлы", "*.*"),
            ]
        )

        if not path:
            return

        img = imread_unicode(path)
        if img is None:
            messagebox.showerror("Ошибка", "Не удалось прочитать изображение.")
            return

        self.image_path = path
        self.original_bgr = img
        self.result_bgr = None
        self.seed_items = []
        self.current_view = "original"
        self.fit_mode = "width"

        h, w = img.shape[:2]
        self.status_var.set(f"Загружено: {os.path.basename(path)}, {w}x{h}.")

        self.canvas.after_idle(self._refresh_view_after_change)

    def process(self):
        if self.original_bgr is None:
            messagebox.showwarning("Нет изображения", "Сначала откройте фото.")
            return

        self.status_var.set("Обработка...")
        self.root.update_idletasks()

        try:
            params = self.get_params()
            result = process_image(self.original_bgr, params)

            self.result_bgr = result["annotated"]
            self.seed_items = result["items"]
            self.current_view = "result"

            self.canvas.after_idle(self._refresh_view_after_change)

            n_focus = sum(1 for it in self.seed_items if it["status"] == "focused")
            n_blur = sum(1 for it in self.seed_items if it["status"] == "blurry")
            n_unknown = sum(1 for it in self.seed_items if it["status"] == "unknown")

            self.status_var.set(
                f"Найдено: {len(self.seed_items)}; "
                f"фокус: {n_focus}; размыто: {n_blur}; неизвестно: {n_unknown}."
            )

        except Exception as exc:
            messagebox.showerror("Ошибка обработки", str(exc))
            self.status_var.set("Ошибка обработки.")

    def show_original(self):
        if self.original_bgr is None:
            messagebox.showwarning("Нет изображения", "Сначала откройте фото.")
            return

        self.current_view = "original"
        self.canvas.after_idle(self._refresh_view_after_change)

    def show_result(self):
        if self.result_bgr is None:
            messagebox.showwarning("Нет результата", "Сначала выполните обработку.")
            return

        self.current_view = "result"
        self.canvas.after_idle(self._refresh_view_after_change)

    def save_crops(self):
        if self.original_bgr is None:
            messagebox.showwarning("Нет изображения", "Сначала откройте фото.")
            return

        if not self.seed_items:
            messagebox.showwarning("Нет данных", "Нет обнаруженных семян для сохранения.")
            return

        items = self.seed_items

        if self.vars["save_only_focused"].get():
            items = [it for it in items if it["status"] == "focused"]
        elif self.vars["skip_blurry"].get():
            items = [it for it in items if it["status"] != "blurry"]

        if not items:
            messagebox.showwarning("Нет данных", "Нет объектов для сохранения по текущим фильтрам.")
            return

        dirpath = filedialog.askdirectory(title="Куда сохранить нарезанные семена")
        if not dirpath:
            return

        h, w = self.original_bgr.shape[:2]
        pad = max(0, int(self.vars["padding"].get()))
        saved = 0

        for i, item in enumerate(items, 1):
            x, y, bw, bh = item["bbox"]

            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)

            crop = self.original_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            if item["edge_width"] is None:
                edge_txt = "NA"
            else:
                edge_txt = f"{item['edge_width']:.1f}px".replace(".", "_")

            filename = f"seed_{i:03d}_{item['status']}_{edge_txt}.png"
            full_path = os.path.join(dirpath, filename)

            if imwrite_unicode(full_path, crop):
                saved += 1

        self.status_var.set(f"Сохранено {saved} из {len(items)} фрагментов в: {dirpath}")

    def save_result_image(self):
        if self.result_bgr is None:
            messagebox.showwarning("Нет результата", "Сначала выполните обработку.")
            return

        path = filedialog.asksaveasfilename(
            title="Сохранить результат",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
                ("Все файлы", "*.*"),
            ],
        )

        if not path:
            return

        if imwrite_unicode(path, self.result_bgr):
            self.status_var.set(f"Результат сохранён: {path}")
        else:
            messagebox.showerror("Ошибка", "Не удалось сохранить результат.")


def main():
    setup_dpi_awareness()

    root = tk.Tk()

    # Пытаемся сделать шрифты чёткими/масштабированными под DPI.
    try:
        dpi = root.winfo_fpixels("1i")
        root.tk.call("tk", "scaling", dpi / 72.0)
    except Exception:
        pass

    try:
        base_font = tkfont.nametofont("TkDefaultFont")
        base_font.configure(size=10)
    except Exception:
        pass

    app = FlaxSeedApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()