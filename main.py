import os
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox

import numpy as np
import cv2
from PIL import Image, ImageTk

from utils import (
    imread_unicode,
    imwrite_unicode,
    ensure_odd,
    process_image,
    compute_size_stats,
    format_size_stats,
)
from ruler import calibrate_ruler

RESAMPLE = Image.Resampling.LANCZOS
TOOLTIPS = {
    "blur_kernel": (
        "Размер ядра гауссова размытия перед порогом.\n"
        "Убирает шум, но слишком большое значение ухудшает границы."
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
        "Минимальная разница яркости между семенем и фоном.\n"
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
        "Удалять объекты, которые сильно отличаются от медианной площади "
        "и размера остальных семян."
    ),
    "skip_blurry": (
        "При сохранении пропускать красные объекты.\n"
        "Помогает от мусора, но может пропустить сильно размытые семена."
    ),
    "save_only_focused": "При сохранении сохранять только зелёные объекты.",
    "mm_per_px": (
        "Размер одного пикселя в миллиметрах (мм/px).\n"
        "Например, если 10 мм на фото = 400 px, то 10 / 400 = 0.025.\n"
        "Можно вводить с точкой или запятой.\n"
        "Пусто — замеры считаются в пикселях."
    ),
    "calibrate_ruler": (
        "Найти линейку на текущем фото и автоматически вычислить мм/px.\n"
        "Результат будет подставлен в поле масштаба."
    ),
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

        # Масштаб (мм на пиксель) и текст статистики размеров — отдельно от self.vars,
        # потому что мм/px вводится вручную и разбирается в get_params().
        self.mm_per_px_var = tk.StringVar(value="")
        self.stats_var = tk.StringVar(value="—")

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
            ("Калибровать линейку", self.calibrate_ruler_from_image),
            ("Исходное", self.show_original),
            ("Результат", self.show_result),
            ("Сохранить нарезку", self.save_crops),
            ("Сохранить результат", self.save_result_image),
        ]

        for text, cmd in buttons:
            btn = ttk.Button(toolbar, text=text, command=cmd)
            btn.pack(side=tk.LEFT, padx=2)

            if text == "Калибровать линейку":
                ToolTip(btn, TOOLTIPS.get("calibrate_ruler", ""))

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

        status = ttk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            padding=(5, 2)
        )
        status.pack(fill=tk.X, side=tk.BOTTOM)

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

    def _build_param_panel(self):
        pf = self.param_frame

        # --- Масштаб и статистика размеров ---
        ttk.Label(pf, text="Масштаб, мм/px:").pack(anchor="w", padx=4, pady=(4, 0))

        mm_entry = ttk.Entry(pf, textvariable=self.mm_per_px_var, width=18)
        mm_entry.pack(fill=tk.X, padx=4)
        ToolTip(mm_entry, TOOLTIPS.get("mm_per_px", ""))

        ttk.Label(pf, text="Статистика семян:").pack(anchor="w", padx=4, pady=(6, 0))
        ttk.Label(
            pf,
            textvariable=self.stats_var,
            justify="left"
        ).pack(anchor="w", padx=4)

        ttk.Separator(pf, orient="horizontal").pack(fill="x", pady=4)

        # Порог теперь фиксированный — только Otsu.
        ttk.Label(pf, text="Порог: Otsu").pack(anchor="w", padx=4, pady=(4, 0))

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

        if int(params["max_area"]) <= int(params["min_area"]):
            params["max_area"] = int(params["min_area"]) + 100

        if float(params["max_ratio"]) <= float(params["min_ratio"]):
            params["max_ratio"] = float(params["min_ratio"]) + 0.5

        if float(params["blur_edge_width"]) <= float(params["max_edge_width"]):
            params["blur_edge_width"] = float(params["max_edge_width"]) + 0.1

        params["mm_per_px"] = self._parse_mm_per_px()

        return params

    def _parse_mm_per_px(self):
        """
        Возвращает мм/px как float или None, если поле пустое.
        Принимает и точку, и запятую.
        """
        text = self.mm_per_px_var.get().strip().replace(",", ".")

        if not text:
            return None

        try:
            value = float(text)
        except ValueError:
            raise ValueError("Масштаб мм/px должен быть числом, например 0.025")

        if not np.isfinite(value) or value <= 0:
            raise ValueError("Масштаб мм/px должен быть положительным числом, например 0.025")

        return value

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

        self.stats_var.set("—")
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

            stats = compute_size_stats(self.seed_items, params["mm_per_px"])
            self.stats_var.set(format_size_stats(stats))

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

    def calibrate_ruler_from_image(self):
        if self.original_bgr is None:
            messagebox.showwarning("Нет изображения", "Сначала откройте фото.")
            return

        self.status_var.set("Калибровка линейки...")
        self.root.update_idletasks()

        try:
            mm_per_px = calibrate_ruler(self.original_bgr, debug=True)

            self.mm_per_px_var.set(f"{mm_per_px:.6f}")
            self.status_var.set(
                f"Калибровка выполнена: {mm_per_px:.6f} мм/px "
                f"({1.0 / mm_per_px:.2f} px/mm)."
            )

        except Exception as exc:
            messagebox.showerror("Ошибка калибровки", str(exc))
            self.status_var.set("Ошибка калибровки линейки.")

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
            messagebox.showwarning(
                "Нет данных",
                "Нет объектов для сохранения по текущим фильтрам."
            )
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