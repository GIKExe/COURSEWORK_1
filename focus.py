#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Фокус-детектор  (OpenCV + Pillow + Tkinter)
-------------------------------------------
1. Пользователь выбирает фотографию.
2. Программа строит карту резкости и находит ЦЕНТР и РАДИУС области фокуса.
3. Всё, что ВНЕ области фокуса, закрашивается полупрозрачным красным.

Зависимости:
    pip install opencv-python pillow numpy
Запуск:
    python focus_detector.py
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
except ImportError as exc:                                    # pragma: no cover
    sys.exit(f"Не хватает библиотек: {exc}\n\nУстановите:  pip install opencv-python pillow numpy")


# ==========================================================================
#  Константы
# ==========================================================================
MAX_VIEW_SIDE = 1400          # разрешение картинки для показа/наложения
MAX_CALC_SIDE = 820           # разрешение для вычисления карты резкости (скорость)

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

METHODS = {
    "Норм. Лапласиан (устойчив к текстуре)": "norm_lap",
    "Локальное СКО Лапласиана":              "lapvar",
    "Градиент (Tenengrad)":                  "tenengrad",
}


# ==========================================================================
#  Ввод-вывод (пути с кириллицей cv2.imread не понимает)
# ==========================================================================
def imread_unicode(path: str, flags=cv2.IMREAD_COLOR):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    except Exception:
        return cv2.imread(path, flags)


def imwrite_unicode(path: str, img) -> bool:
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    try:
        buf.tofile(path)
        return True
    except Exception:
        return False


def resize_max_side(img, max_side: int):
    """Уменьшает изображение так, чтобы большая сторона была <= max_side."""
    h, w = img.shape[:2]
    s = max_side / float(max(h, w))
    if s >= 1.0:
        return img.copy(), 1.0
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA), s


# ==========================================================================
#  АНАЛИЗ: карта резкости -> центр и радиус фокуса
# ==========================================================================
def build_focus_map(gray: np.ndarray, sigma: float = 6.0, method: str = "norm_lap") -> np.ndarray:
    """
    Строит карту «резкости» по полутоновому изображению.
    Чем выше значение пикселя — тем резче локальная окрестность.
    """
    g = gray.astype(np.float32) / 255.0
    k = max(3, int(2 * round(2 * sigma) + 1))          # размер скользящего окна (нечётный)

    if method == "lapvar":
        # локальное стандартное отклонение Лапласиана
        lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
        m1 = cv2.boxFilter(lap, cv2.CV_32F, (k, k))
        m2 = cv2.boxFilter(lap * lap, cv2.CV_32F, (k, k))
        fm = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))

    elif method == "tenengrad":
        # энергия градиента (Собель) в окне
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        fm = np.sqrt(cv2.boxFilter(gx * gx + gy * gy, cv2.CV_32F, (k, k)))

    else:  # "norm_lap" — |L| / |∇|, почти не зависит от «насыщенности» текстуры
        lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        eps = 0.02 * float(grad.mean()) + 1e-6
        fm = cv2.boxFilter(lap / (grad + eps), cv2.CV_32F, (k, k))

    fm = cv2.GaussianBlur(fm, (0, 0), sigmaX=max(0.8, sigma * 0.6))
    return fm.astype(np.float32)


def normalize01(m: np.ndarray, lo_p: float = 1.0, hi_p: float = 99.9) -> np.ndarray:
    lo, hi = np.percentile(m, (lo_p, hi_p))
    out = (m - lo) / max(float(hi - lo), 1e-6)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


@dataclass
class FocusResult:
    center: tuple[float, float]     # центр фокуса (в координатах расчётного кадра)
    radius: float                   # радиус фокуса
    area_ratio: float               # доля резких пикселей
    threshold: float                # использованный порог
    core_sharp: float               # средняя резкость внутри
    back_sharp: float               # средняя резкость снаружи
    fmap: np.ndarray                # нормированная карта резкости (0..1)
    mask: np.ndarray                # бинарная маска резкой области


def analyze_focus(gray: np.ndarray, method: str = "norm_lap",
                  sigma: float = 6.0, level: float = 0.45) -> FocusResult:
    """Главная функция поиска центра и радиуса области фокусировки."""
    h, w = gray.shape[:2]
    fmap = build_focus_map(gray, sigma=sigma, method=method)
    fn = normalize01(fmap)

    if float(fn.max()) < 1e-6:                       # абсолютно плоский кадр
        return FocusResult((w / 2.0, h / 2.0), min(w, h) / 2.0, 1.0, 0.0, 0.0, 0.0,
                           fn, np.ones((h, w), np.uint8))

    thr = float(level * fn.max())
    mask = (fn >= thr).astype(np.uint8)

    # убираем одиночные «горячие» пиксели и замыкаем мелкие дырки
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    ymax, xmax = np.unravel_index(int(np.argmax(fn)), fn.shape)

    lab = int(labels[ymax, xmax])
    if num <= 1 or lab == 0:                          # fallback: самая крупная компонента
        lab = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) if num > 1 else 0
    comp = (labels == lab).astype(np.uint8) if lab != 0 else mask

    # --- взвешенный центр (веса = значения карты резкости) ---
    ys, xs = np.nonzero(comp)
    if len(xs) == 0:
        cx, cy = float(xmax), float(ymax)
        dists = np.array([0.0])
    else:
        ws = fn[ys, xs] + 1e-6
        cx = float(np.sum(xs * ws) / np.sum(ws))
        cy = float(np.sum(ys * ws) / np.sum(ws))
        dists = np.hypot(xs - cx, ys - cy)

    area = float(comp.sum())
    radius = float(np.percentile(dists, 95)) if len(dists) else 0.0   # устойчиво к выбросам
    radius = max(radius, float(np.sqrt(area / np.pi)), 0.02 * min(h, w))
    radius = min(radius, float(np.hypot(w, h) / 2.0))

    inside = comp.astype(bool)
    return FocusResult(
        center=(cx, cy),
        radius=radius,
        area_ratio=area / float(h * w),
        threshold=thr,
        core_sharp=float(fn[inside].mean()) if inside.any() else 0.0,
        back_sharp=float(fn[~inside].mean()) if (~inside).any() else 0.0,
        fmap=fn,
        mask=comp,
    )


# ==========================================================================
#  РЕНДЕР: полупрозрачная красная маска + аннотации
# ==========================================================================
def soft_focus_mask(shape, center, radius: float, feather: float) -> np.ndarray:
    """Маска 1.0 внутри круга фокуса, 0.0 снаружи, с плавным переходом."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (int(round(center[0])), int(round(center[1]))),
               max(1, int(round(radius))), 255, -1, cv2.LINE_AA)
    feather = float(np.clip(feather, 0, 150))
    if feather > 0.5:
        k = int(2 * round(feather) + 1)
        m = cv2.GaussianBlur(m, (k, k), 0)
    return m.astype(np.float32) / 255.0


def apply_red_overlay(bgr: np.ndarray, soft: np.ndarray, alpha: float) -> np.ndarray:
    """Красит всё ВНЕ области фокуса полупрозрачным красным."""
    wgt = (float(np.clip(alpha, 0, 0.95)) * (1.0 - soft))[..., None]
    red = np.array([0, 0, 255], np.float32)                     # BGR
    out = bgr.astype(np.float32) * (1.0 - wgt) + red * wgt
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_annotations(img: np.ndarray, center, radius: float) -> None:
    """Окружность, перекрестье центра и подпись радиуса."""
    c = (int(round(center[0])), int(round(center[1])))
    r = max(1, int(round(radius)))
    col = (0, 255, 190)

    cv2.circle(img, c, r, col, max(2, r // 200), cv2.LINE_AA)
    s = max(12, r // 8)
    cv2.line(img, (c[0] - s, c[1]), (c[0] + s, c[1]), col, 2, cv2.LINE_AA)
    cv2.line(img, (c[0], c[1] - s), (c[0], c[1] + s), col, 2, cv2.LINE_AA)
    cv2.circle(img, c, max(3, r // 70), col, -1, cv2.LINE_AA)

    txt = f"R = {r} px"
    font, sc, th = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
    (tw, thh), base = cv2.getTextSize(txt, font, sc, th)
    tx = int(np.clip(c[0] + r * 0.71, 4, img.shape[1] - tw - 8))
    ty = int(np.clip(c[1] - r * 0.71, thh + 8, img.shape[0] - 6))
    cv2.rectangle(img, (tx - 4, ty - thh - 4), (tx + tw + 4, ty + base + 2), (18, 18, 18), -1)
    cv2.putText(img, txt, (tx, ty), font, sc, (255, 255, 255), th, cv2.LINE_AA)


# ==========================================================================
#  Демо-кадр (для проверки алгоритма без своей фотографии)
# ==========================================================================
def make_demo_image(w: int = 1200, h: int = 800) -> np.ndarray:
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 45 + 120 * (xx / w)
    img[..., 1] = 70 + 100 * (yy / h)
    img[..., 2] = 95
    img = np.clip(img + rng.integers(0, 80, (h, w, 3)).astype(np.float32), 0, 255)

    for _ in range(140):
        color = tuple(int(v) for v in rng.integers(50, 255, 3))
        pt = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.circle(img, pt, int(rng.integers(5, 55)), color, -1, cv2.LINE_AA)
    cv2.putText(img, "FOCUS TEST", (int(w * .18), int(h * .55)),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 6, cv2.LINE_AA)
    img = np.clip(img, 0, 255).astype(np.uint8)

    cx, cy = w * 0.5, h * 0.45
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (min(w, h) * 0.42)
    alpha = np.clip(dist, 0, 1)[..., None]
    blurred = cv2.GaussianBlur(img, (0, 0), 20)
    out = img.astype(np.float32) * (1 - alpha) + blurred.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# ==========================================================================
#  ПРИЛОЖЕНИЕ
# ==========================================================================
class FocusApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Фокус-детектор · OpenCV + Pillow + Tkinter")
        self.geometry("1340x860")
        self.minsize(1000, 640)

        # ---- изображения ----
        self.src_bgr = None          # оригинал
        self.view_bgr = None         # рабочий кадр (для показа)
        self.calc_bgr = None         # уменьшенный кадр (для расчётов)
        self.rendered_bgr = None     # кэш уже нарисованного результата
        self.calc2view = 1.0
        self.view2src = 1.0

        self.result: FocusResult | None = None
        self.path: str | None = None
        self.photo = None
        self._debounce = None
        self.manual_center = None    # ручная коррекция центра (в координатах view)
        self.ox = self.oy = 0
        self.disp_scale = 1.0

        # ---- переменные UI ----
        self.var_method  = tk.StringVar(value=list(METHODS)[0])
        self.var_level   = tk.DoubleVar(value=0.45)
        self.var_sigma   = tk.DoubleVar(value=6.0)
        self.var_feather = tk.DoubleVar(value=45.0)
        self.var_alpha   = tk.DoubleVar(value=0.45)
        self.var_rmult   = tk.DoubleVar(value=1.0)
        self.var_annot   = tk.BooleanVar(value=True)
        self.var_manual  = tk.BooleanVar(value=False)
        self.var_mode    = tk.StringVar(value="result")

        self._build_ui()
        self.bind("<Control-o>", lambda e: self.open_file())
        self.bind("<Control-s>", lambda e: self.save_result())

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        # ---- статус-бар ----
        self.status = ttk.Label(self, text="Откройте фотографию (Ctrl+O) или нажмите «Демо-кадр»",
                                anchor="w", relief="sunken", padding=(8, 4))
        self.status.pack(side="bottom", fill="x")

        # ---- левая панель ----
        bar = ttk.Frame(self, padding=10)
        bar.pack(side="left", fill="y")

        ttk.Button(bar, text="📂  Открыть фото…", command=self.open_file).pack(fill="x")
        ttk.Button(bar, text="🎛  Демо-кадр", command=self.load_demo).pack(fill="x", pady=(4, 0))
        ttk.Button(bar, text="💾  Сохранить результат", command=self.save_result).pack(fill="x", pady=(4, 0))
        ttk.Separator(bar).pack(fill="x", pady=10)

        ttk.Label(bar, text="Метод оценки резкости:").pack(anchor="w")
        cb = ttk.Combobox(bar, textvariable=self.var_method, state="readonly",
                          values=list(METHODS.keys()))
        cb.pack(fill="x", pady=(2, 8))
        cb.bind("<<ComboboxSelected>>", lambda e: self.schedule_recompute())

        self._slider(bar, "Порог резкости", self.var_level, 0.10, 0.90, "{:.2f}")
        self._slider(bar, "Окно анализа (σ)", self.var_sigma, 2.0, 18.0, "{:.1f}")
        self._slider(bar, "Растушёвка края, px", self.var_feather, 0.0, 150.0, "{:.0f}")
        self._slider(bar, "Плотность красного", self.var_alpha, 0.05, 0.90, "{:.2f}")
        self._slider(bar, "Радиус: множитель", self.var_rmult, 0.40, 2.00, "{:.2f}")

        ttk.Separator(bar).pack(fill="x", pady=10)
        ttk.Label(bar, text="Режим просмотра:").pack(anchor="w")
        for text, val in (("Результат (красная маска)", "result"),
                          ("Карта резкости", "heat"),
                          ("Оригинал", "orig")):
            ttk.Radiobutton(bar, text=text, value=val, variable=self.var_mode,
                            command=self.invalidate).pack(anchor="w")

        ttk.Checkbutton(bar, text="Показывать центр и радиус", variable=self.var_annot,
                        command=self.invalidate).pack(anchor="w", pady=(8, 0))
        ttk.Checkbutton(bar, text="Ручной центр (клик по фото,\nShift+клик — радиус)",
                        variable=self.var_manual).pack(anchor="w", pady=(4, 0))
        ttk.Button(bar, text="↺ Сбросить ручную коррекцию",
                   command=self.reset_manual).pack(fill="x", pady=(6, 0))

        ttk.Separator(bar).pack(fill="x", pady=10)
        ttk.Label(bar, text="Результат анализа:").pack(anchor="w")
        self.info = tk.Text(bar, width=38, height=10, wrap="word", relief="flat",
                            background=self["bg"], font=("Consolas", 9), state="disabled")
        self.info.pack(fill="x", pady=(2, 0))

        # ---- холст ----
        wrap = ttk.Frame(self)
        wrap.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(wrap, background="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self._placeholder()

    def _slider(self, parent, text, var, lo, hi, fmt):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(6, 0))
        ttk.Label(row, text=text).pack(side="left")
        lbl = ttk.Label(row, text=fmt.format(var.get()), width=6, anchor="e")
        lbl.pack(side="right")
        ttk.Scale(parent, from_=lo, to=hi, variable=var, orient="horizontal",
                  command=lambda _v, l=lbl, v=var, f=fmt: self._on_slider(l, v, f)).pack(fill="x")

    def _on_slider(self, label, var, fmt):
        label.config(text=fmt.format(var.get()))
        self.schedule_recompute()

    # -------------------------------------------------------------- загрузка
    def open_file(self):
        path = filedialog.askopenfilename(
            title="Выберите фотографию",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                       ("Все файлы", "*.*")])
        if not path:
            return
        img = imread_unicode(path)
        if img is None:
            messagebox.showerror("Ошибка", f"Не удалось прочитать файл:\n{path}")
            return
        self.set_image(img, path)

    def load_demo(self):
        self.set_image(make_demo_image(), "demo_focus_test.png")

    def set_image(self, bgr: np.ndarray, path: str | None):
        self.src_bgr = bgr
        self.path = path
        self.view_bgr, s1 = resize_max_side(bgr, MAX_VIEW_SIDE)
        self.calc_bgr, _ = resize_max_side(self.view_bgr, MAX_CALC_SIDE)
        self.view2src = bgr.shape[1] / self.view_bgr.shape[1]
        self.calc2view = self.view_bgr.shape[1] / self.calc_bgr.shape[1]
        self.manual_center = None
        self.var_rmult.set(1.0)
        self.status.config(text=f"Обработка: {os.path.basename(path or 'кадр')} …")
        self.update_idletasks()
        self._do_recompute()
        self.status.config(text=f"Готово: {os.path.basename(path or 'кадр')}  "
                                f"({bgr.shape[1]}×{bgr.shape[0]} px)")

    # --------------------------------------------------------------- расчёт
    def schedule_recompute(self, delay: int = 90):
        if self._debounce is not None:
            self.after_cancel(self._debounce)
        self._debounce = self.after(delay, self._do_recompute)

    def _do_recompute(self):
        self._debounce = None
        if self.calc_bgr is None:
            return
        gray = cv2.cvtColor(self.calc_bgr, cv2.COLOR_BGR2GRAY)
        res = analyze_focus(gray, METHODS[self.var_method.get()],
                            self.var_sigma.get(), self.var_level.get())
        k = self.calc2view
        res.center = (res.center[0] * k, res.center[1] * k)
        res.radius *= k
        vh, vw = self.view_bgr.shape[:2]
        res.fmap = cv2.resize(res.fmap, (vw, vh), interpolation=cv2.INTER_LINEAR)
        self.result = res
        self.invalidate()
        self.update_info()

    def invalidate(self):
        self.rendered_bgr = None
        self.redraw()

    # ------------------------------------------------------- центр / радиус
    @property
    def eff_center(self):
        if self.manual_center is not None:
            return self.manual_center
        return self.result.center if self.result else None

    @property
    def eff_radius(self):
        if not self.result:
            return 0.0
        return max(4.0, self.result.radius * float(self.var_rmult.get()))

    def reset_manual(self):
        self.manual_center = None
        self.var_rmult.set(1.0)
        self.var_manual.set(False)
        self.invalidate()
        self.update_info()

    # ------------------------------------------------------------- отрисовка
    def build_view_image(self) -> np.ndarray:
        if self.rendered_bgr is not None:
            return self.rendered_bgr

        base = self.view_bgr
        mode = self.var_mode.get()
        res = self.result
        c, r = self.eff_center, self.eff_radius

        if mode == "heat" and res is not None:
            heat = cv2.applyColorMap((res.fmap * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            out = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
        elif mode == "orig" or res is None or c is None:
            out = base.copy()
        else:
            soft = soft_focus_mask(base.shape, c, r, self.var_feather.get())
            out = apply_red_overlay(base, soft, self.var_alpha.get())
            if self.var_annot.get():
                draw_annotations(out, c, r)

        self.rendered_bgr = out
        return out

    def redraw(self):
        if self.view_bgr is None:
            self._placeholder()
            return
        cw = max(self.canvas.winfo_width(), 50)
        ch = max(self.canvas.winfo_height(), 50)
        img = self.build_view_image()

        scale = min(cw / img.shape[1], ch / img.shape[0], 1.0)
        nw, nh = max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))
        small = cv2.resize(img, (nw, nh),
                           interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))   # ссылка защищает от сборщика мусора
        self.canvas.delete("all")
        self.ox = (cw - nw) // 2
        self.oy = (ch - nh) // 2
        self.disp_scale = scale
        self.canvas.create_image(self.ox, self.oy, image=self.photo, anchor="nw")

    def _placeholder(self):
        self.canvas.delete("all")
        cw, ch = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        self.canvas.create_text(cw // 2, ch // 2, fill="#8a8a8a",
                                font=("Segoe UI", 15), justify="center",
                                text="📷\nОткройте фотографию (Ctrl+O)\nили нажмите «Демо-кадр»")

    # ---------------------------------------------------------- мышь / инфо
    def on_canvas_click(self, event):
        if not self.var_manual.get() or self.result is None or self.disp_scale <= 0:
            return
        vx = (event.x - self.ox) / self.disp_scale
        vy = (event.y - self.oy) / self.disp_scale
        h, w = self.view_bgr.shape[:2]
        if not (0 <= vx < w and 0 <= vy < h):
            return

        if event.state & 0x0001:                     # Shift: задаём радиус по точке клика
            auto_r = self.result.radius
            if auto_r > 1:
                d = float(np.hypot(vx - self.eff_center[0], vy - self.eff_center[1]))
                self.var_rmult.set(float(np.clip(d / auto_r, 0.4, 2.0)))
                self._do_recompute()
        else:                                        # обычный клик: центр
            self.manual_center = (vx, vy)
            self.invalidate()
            self.update_info()

    def update_info(self):
        res = self.result
        if res is None or self.view_bgr is None:
            return
        vh, vw = self.view_bgr.shape[:2]
        c, r = self.eff_center, self.eff_radius
        contrast = res.core_sharp / res.back_sharp if res.back_sharp > 1e-6 else float("inf")
        src_h, src_w = self.src_bgr.shape[:2]
        k = self.view2src

        lines = [
            f"Файл     : {os.path.basename(self.path or '—')}",
            f"Размер   : {src_w}×{src_h} px",
            f"Метод    : {METHODS[self.var_method.get()]}",
            "",
            f"ЦЕНТР ФОКУСА",
            f"  X = {c[0]*k:8.1f} px  ({c[0]/vw:.2f} от ширины)",
            f"  Y = {c[1]*k:8.1f} px  ({c[1]/vh:.2f} от высоты)",
            f"РАДИУС ФОКУСА",
            f"  R = {r*k:8.1f} px  (×{self.var_rmult.get():.2f})",
            "",
            f"Резкая область : {res.area_ratio*100:5.1f} % кадра",
            f"Резкость ядра  : {res.core_sharp:.3f}",
            f"Резкость фона  : {res.back_sharp:.3f}",
            f"Контраст фокуса: ×{contrast:.2f}",
        ]
        if self.manual_center is not None:
            lines.append("\n(центр задан вручную)")

        self.info.config(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", "\n".join(lines))
        self.info.config(state="disabled")

    # -------------------------------------------------------------- сохранение
    def save_result(self):
        if self.src_bgr is None or self.result is None:
            messagebox.showinfo("Сохранение", "Сначала откройте изображение.")
            return
        base = os.path.splitext(os.path.basename(self.path or "result"))[0]
        path = filedialog.asksaveasfilename(
            title="Сохранить результат", defaultextension=".png",
            initialfile=f"{base}_focus.png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("Все файлы", "*.*")])
        if not path:
            return

        k = self.view2src
        out = self.src_bgr.copy()
        if self.var_mode.get() != "orig":
            c = (self.eff_center[0] * k, self.eff_center[1] * k)
            r = self.eff_radius * k
            if self.var_mode.get() == "heat":
                heat = cv2.applyColorMap((self.result.fmap * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                heat = cv2.resize(heat, (out.shape[1], out.shape[0]))
                out = cv2.addWeighted(out, 0.55, heat, 0.45, 0)
            else:
                soft = soft_focus_mask(out.shape, c, r, self.var_feather.get() * k)
                out = apply_red_overlay(out, soft, self.var_alpha.get())
                if self.var_annot.get():
                    draw_annotations(out, c, r)

        ok = imwrite_unicode(path, out)
        self.status.config(text=("Сохранено: " + path) if ok else "Ошибка сохранения")
        if not ok:
            messagebox.showerror("Ошибка", "Не удалось записать файл.")


# ==========================================================================
if __name__ == "__main__":
    app = FocusApp()
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        app.after(120, lambda: app.set_image(imread_unicode(sys.argv[1]), sys.argv[1]))
    app.mainloop()