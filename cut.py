import os
import tkinter as tk
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


# ------------------------------------------------------------
# Вспомогательные функции ввода/вывода
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
    Предполагаем, что фон тёмный, семена обычно светлее фона.
    """
    k = ensure_odd(params["blur_kernel"], 1)
    blurred = cv2.GaussianBlur(gray, (k, k), 0)

    if params["thresh_mode"] == "Adaptive":
        block = ensure_odd(params["block_size"], 3)
        c = int(params["adaptive_c"])
        mask = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_GAUSSIAN_C,
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

        # Если белые пиксели темнее чёрных или их слишком много — инверсия.
        if mean_white < mean_black or white_count > mask.size * 0.6:
            mask = cv2.bitwise_not(mask)
    elif white_count == mask.size:
        # Если всё изображение стало белым, маска, скорее всего, испорчена.
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


def is_likely_ruler(bbox, img_w, img_h, params):
    """
    Эвристика для отсечения линейки внизу изображения.
    """
    if not params.get("remove_ruler", True):
        return False

    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return False

    ratio = float(max(w, h)) / float(max(1, min(w, h)))
    bottom_limit = int(img_h * 0.82)

    # Длинный широкий объект, похожий на линейку.
    if w > img_w * 0.35 and ratio > 4.0:
        return True

    # Объект у нижней части и при этом сильно вытянут.
    if y + h >= bottom_limit and ratio > 3.0 and w > img_w * 0.18:
        return True

    # Очень тонкая и длинная область.
    if h < max(4, int(img_h * 0.015)) and w > img_w * 0.25:
        return True

    return False


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

    # Запасной вариант: ближайшая точка к уровню, если переход есть.
    if np.any(y >= level) and np.any(y < level):
        idx = int(np.argmin(np.abs(y - level)))
        return float(x[idx])

    return None


def estimate_edge_width(gray, contour, bbox, max_points=160):
    """
    Примерная оценка ширины переходной границы семя/фон.

    Идея:
    - берём контур семени;
    - строим нормали;
    - усредняем интенсивность вдоль нормалей изнутри наружу;
    - нормируем профиль;
    - измеряем расстояние между уровнями 0.8 и 0.2.

    Для резких границ это обычно 1-4 пикселя.
    Для размытых/плохо освещённых — 7 и больше.
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

    # Касательные и нормали к контуру.
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

    # Направление наружу: фон обычно темнее семени.
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
    Возвращает список семян, аннотированное изображение и маску.
    """
    h, w = img_bgr.shape[:2]

    bottom_percent = max(0, min(70, int(params["bottom_percent"])))
    bottom_px = int(h * bottom_percent / 100.0)
    roi_h = h - bottom_px

    if roi_h < 10:
        roi_h = min(h, 10)

    bottom_px = h - roi_h
    roi = img_bgr[:roi_h, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
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

    if float(params["blur_edge_width"]) <= float(params["max_edge_width"]):
        params = dict(params)
        params["blur_edge_width"] = float(params["max_edge_width"]) + 0.1

    estimate_edge = bool(params["estimate_edge"]) or bool(params["filter_edge"])
    filter_edge = bool(params["filter_edge"])

    items = []
    roi_h_gt, roi_w_gt = gray.shape[:2]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)

        if bw < 4 or bh < 4:
            continue

        rect_area = bw * bh
        if area <= 0:
            area = rect_area

        if area < min_area or area > max_area:
            continue

        if bw > roi_w_gt * 0.85 or bh > roi_h_gt * 0.85:
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

        bbox = (x, y, bw, bh)

        if is_likely_ruler(bbox, roi_w_gt, roi_h_gt, params):
            continue

        edge_width = None
        if estimate_edge:
            edge_width = estimate_edge_width(gray, cnt, bbox)

        status = classify_edge(edge_width, params)

        if filter_edge and status != "focused":
            continue

        items.append({
            "bbox": bbox,
            "area": float(area),
            "edge_width": edge_width,
            "status": status,
            "contour": cnt,
        })

    # Сортировка по строкам и столбцам.
    items.sort(key=lambda it: (it["bbox"][1], it["bbox"][0]))

    annotated = img_bgr.copy()

    # Затемнить нижнюю область с линейкой, если она отсечена.
    if bottom_px > 0:
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, roi_h), (w - 1, h - 1), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0)

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
        "roi_h": roi_h,
    }


# ------------------------------------------------------------
# GUI
# ------------------------------------------------------------

class FlaxSeedApp:
    DISPLAY_MAX_W = 1000
    DISPLAY_MAX_H = 700

    def __init__(self, root):
        self.root = root
        self.original_bgr = None
        self.result_bgr = None
        self.seed_items = []
        self.image_path = None
        self.tk_image = None

        self.status_var = tk.StringVar(
            value="Откройте фото, подберите параметры и нажмите «Обработать»."
        )

        self._init_vars()
        self._build_ui()

    def _init_vars(self):
        self.vars = {}

        # key, label, low, high, step, default, kind
        self.param_defs = [
            ("bottom_percent", "Обрезать снизу, %", 0, 60, 1, 12, "int"),
            ("blur_kernel", "Размытие, ядро", 1, 31, 2, 5, "odd"),
            ("block_size", "Adaptive block size", 3, 199, 2, 51, "odd"),
            ("adaptive_c", "Adaptive C", -30, 30, 1, 5, "int"),
            ("close_kernel", "Close, морфология", 1, 31, 1, 7, "int"),
            ("open_kernel", "Open, морфология", 1, 31, 1, 3, "int"),
            ("min_area", "Мин. площадь, px", 10, 100000, 10, 300, "int"),
            ("max_area", "Макс. площадь, px", 100, 300000, 100, 60000, "int"),
            ("min_ratio", "Мин. вытянутость", 1.0, 10.0, 0.1, 1.0, "float"),
            ("max_ratio", "Макс. вытянутость", 1.0, 15.0, 0.1, 6.0, "float"),
            ("min_solidity", "Мин. solidity", 0.0, 1.0, 0.01, 0.55, "float"),
            ("min_extent", "Мин. extent", 0.0, 1.0, 0.01, 0.25, "float"),
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
        self.vars["remove_ruler"] = tk.BooleanVar(value=True)
        self.vars["filter_edge"] = tk.BooleanVar(value=False)
        self.vars["save_only_focused"] = tk.BooleanVar(value=False)

    def _build_ui(self):
        self.root.title("Выделение и нарезка семян льна")
        self.root.geometry("1400x820")

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

        main = ttk.Frame(self.root, padding=5)
        main.pack(fill=tk.BOTH, expand=True)

        left_outer = ttk.Frame(main)
        left_outer.pack(side=tk.LEFT, fill=tk.Y)

        left_canvas = tk.Canvas(left_outer, width=360, highlightthickness=0)
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

        self.canvas = tk.Canvas(
            right,
            bg="#202020",
            width=self.DISPLAY_MAX_W,
            height=self.DISPLAY_MAX_H
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

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

        checks = [
            ("Оценивать ширину границы", "estimate_edge"),
            ("Отсекать линейку", "remove_ruler"),
            ("Оставлять только сфокусированные", "filter_edge"),
            ("Сохранять только сфокусированные", "save_only_focused"),
        ]

        for text, key in checks:
            ttk.Checkbutton(pf, text=text, variable=self.vars[key]).pack(
                anchor="w", padx=4, pady=1
            )

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

        self.show_image(img)

        h, w = img.shape[:2]
        self.status_var.set(f"Загружено: {os.path.basename(path)}, {w}x{h}.")

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

            self.show_image(self.result_bgr)

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
        self.show_image(self.original_bgr)

    def show_result(self):
        if self.result_bgr is None:
            messagebox.showwarning("Нет результата", "Сначала выполните обработку.")
            return
        self.show_image(self.result_bgr)

    def show_image(self, bgr):
        if bgr is None:
            return

        h, w = bgr.shape[:2]
        scale = min(self.DISPLAY_MAX_W / float(w), self.DISPLAY_MAX_H / float(h), 1.0)

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        if scale < 1.0:
            pil_img = pil_img.resize((new_w, new_h), RESAMPLE)

        self.tk_image = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(new_w // 2, new_h // 2, image=self.tk_image, anchor="center")

    def save_crops(self):
        if not self.seed_items:
            messagebox.showwarning("Нет данных", "Нет обнаруженных семян для сохранения.")
            return

        items = self.seed_items

        if self.vars["save_only_focused"].get():
            items = [it for it in items if it["status"] == "focused"]
            if not items:
                messagebox.showwarning("Нет данных", "Нет сфокусированных семян для сохранения.")
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
    root = tk.Tk()
    app = FlaxSeedApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()