"""Segmentation GUI: parameter tuning, processing, calibration, export."""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

import cv2
from PIL import Image

from core.calibration import format_calibration_message, try_calibrate_ruler
from core.export import export_seed_items, format_export_message, make_source_name, select_items_for_export
from core.io_utils import imread_unicode, imwrite_unicode
from core.measurement import compute_size_stats, count_seed_statuses, format_seed_status_message, format_size_stats
from core.params import (
    format_param_label, get_checkbox_definitions, get_default_params,
    get_param_definitions, normalize_params, parse_mm_per_px_text, round_param_value,
)
from core.segmentation import process_image
from gui.common.widgets import (
    ToolTip,
    создать_верхнюю_панель,
    создать_панель_масштаба,
    создать_панель_модулей,
    создать_статус_строку,
)
from gui.common.zoomable import ZoomableCanvas

TOOLTIPS: dict[str, str] = {
    "blur_kernel": "Размер ядра гауссова размытия перед порогом.\nУбирает шум, но слишком большое значение ухудшает границы.",
    "close_kernel": "Морфологическое закрытие.\nСоединяет разрывы и заполняет мелкие дырки внутри семян.\nСлишком большое значение может склеить соседние семена.",
    "open_kernel": "Морфологическое открытие.\nУдаляет мелкий шум.\nСлишком большое значение может удалить мелкие семена.",
    "min_area": "Минимальная площадь объекта в пикселях. Всё меньше считается шумом.",
    "max_area": "Максимальная площадь объекта. Слишком крупные объекты отбрасываются.",
    "min_ratio": "Минимальная вытянутость: большая сторона бокса / меньшая сторона.",
    "max_ratio": "Максимальная вытянутость. Очень вытянутые объекты обычно не семена.",
    "min_solidity": "Solidity = площадь контура / площадь выпуклой оболочки.\nНизкое значение — рваный/неполный объект.",
    "min_extent": "Extent = площадь контура / площадь прямоугольника.\nНизкое значение — объект занимает мало своего бокса.",
    "min_contrast": "Минимальная разница яркости между семенем и фоном.\nПомогает не брать слабый мусор.\nЕсли семена теряются — уменьшите значение.",
    "max_edge_width": "Если оценка ширины границы не больше этого значения,\nсемя считается сфокусированным и помечается зелёным.",
    "blur_edge_width": "Если оценка ширины границы не меньше этого значения,\nсемя считается размытым и помечается красным.",
    "padding": "Отступ вокруг прямоугольника при сохранении фрагментов.",
    "estimate_edge": "Вычислять ли ширину переходной границы.\nЕсли выключить, статус будет неизвестным, но обработка быстрее.",
    "filter_edge": "Оставлять при обработке только зелёные сфокусированные семена.",
    "filter_outliers": "Удалять объекты, которые сильно отличаются от медианной площади и размера остальных семян.",
    "skip_blurry": "При сохранении пропускать красные объекты.\nПомогает от мусора, но может пропустить сильно размытые семена.",
    "save_only_focused": "При сохранении сохранять только зелёные объекты.",
    "mm_per_px": "Размер одного пикселя в миллиметрах (мм/px).\nНапример, если 10 мм на фото = 400 px, то 10 / 400 = 0.025.\nМожно вводить с точкой или запятой.\nПусто — замеры считаются в пикселях.",
    "calibrate_ruler": "Найти линейку на текущем фото и автоматически вычислить мм/px.\nРезультат будет подставлен в поле масштаба.",
}


class SegmentationApp:
    """Main segmentation GUI with parameter tuning and export."""

    def __init__(
        self,
        root: tk.Tk,
        out_dir: str,
        switch_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.root = root
        self.out_dir = out_dir
        self._switch = switch_callback
        self.original_bgr: Any = None
        self.result_bgr: Any = None
        self.seed_items: list[dict[str, Any]] = []
        self.image_path: str | None = None
        self.current_view = "original"

        self._init_vars()
        self._build_ui()

    def _init_vars(self) -> None:
        self.vars: dict[str, tk.Variable] = {}
        self.param_defs = list(get_param_definitions())
        self.checkbox_defs = list(get_checkbox_definitions())
        self.param_kinds: dict[str, str] = {}
        self.param_titles: dict[str, str] = {}
        self.param_lows: dict[str, float] = {}
        self.param_highs: dict[str, float] = {}
        self.param_steps: dict[str, float] = {}
        self.scale_labels: dict[str, ttk.Label] = {}

        for key, label, low, high, step, default, kind in self.param_defs:
            self.vars[key] = tk.DoubleVar(value=default) if kind == "float" else tk.IntVar(value=default)
            self.param_kinds[key] = kind
            self.param_titles[key] = label
            self.param_lows[key] = low
            self.param_highs[key] = high
            self.param_steps[key] = step
        for _, key, default in self.checkbox_defs:
            self.vars[key] = tk.BooleanVar(value=default)
        self.mm_per_px_var = tk.StringVar(value="")
        self.stats_var = tk.StringVar(value="—")
        self.status_var = tk.StringVar(value="Откройте фото, подберите параметры и нажмите «Обработать».")

    def _build_ui(self) -> None:
        self.root.title("Выделение и нарезка семян ячменя")
        self.root.geometry("1450x850")
        self.root.minsize(1050, 650)

        # ── Верхний тулбар ──
        toolbar = создать_верхнюю_панель(self.root)

        # 1. Единый блок переключения модулей
        создать_панель_модулей(toolbar, "segmentation", self._nav)

        ttk.Separator(toolbar, orient="vertical").pack(side=tk.LEFT, fill="y", padx=8)

        # 2. Кнопки операций
        buttons = [
            ("Открыть фото", self.open_image),
            ("Обработать", self.process),
            ("Калибровать линейку", self.calibrate_ruler_from_image),
            ("Исходное", lambda: self._set_view("original")),
            ("Результат", lambda: self._set_view("result")),
            ("Сохранить нарезку", self.save_crops),
            ("Сохранить результат", self.save_result_image),
        ]
        for text, cmd in buttons:
            btn = ttk.Button(toolbar, text=text, command=cmd)
            btn.pack(side=tk.LEFT, padx=2)
            if text == "Калибровать линейку":
                ToolTip(btn, TOOLTIPS.get("calibrate_ruler", ""))

        ttk.Separator(toolbar, orient="vertical").pack(side=tk.LEFT, fill="y", padx=8)

        # 3. Единый блок масштабирования
        _, self.zoom_label, _ = создать_панель_масштаба(toolbar, self)

        # ── Строка состояния снизу ──
        создать_статус_строку(self.root, переменная=self.status_var, отступ=(8, 4))

        # ── Основная область: слева параметры, справа холст ──
        main = ttk.Frame(self.root, padding=5)
        main.pack(fill=tk.BOTH, expand=True)

        left_outer = ttk.Frame(main)
        left_outer.pack(side=tk.LEFT, fill=tk.Y)
        left_canvas = tk.Canvas(left_outer, width=370, highlightthickness=0)
        left_scroll = ttk.Scrollbar(left_outer, orient=tk.VERTICAL, command=left_canvas.yview)
        self.param_frame = ttk.Frame(left_canvas)
        win_id = left_canvas.create_window((0, 0), window=self.param_frame, anchor="nw")
        self.param_frame.bind("<Configure>", lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.bind("<Configure>", lambda e, wid=win_id: left_canvas.itemconfigure(wid, width=e.width))
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_canvas.pack(side=tk.LEFT, fill=tk.Y)
        left_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Скролл колесиком мыши по панели параметров
        def _on_param_wheel(event: tk.Event) -> None:
            delta = -1 if (event.delta > 0 or getattr(event, "num", 0) == 4) else 1
            left_canvas.yview_scroll(delta, "units")

        left_canvas.bind("<MouseWheel>", _on_param_wheel)
        self.param_frame.bind("<MouseWheel>", _on_param_wheel)

        self._build_param_panel()

        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.zoom_widget = ZoomableCanvas(right)
        self.zoom_widget.set_zoom_callback(lambda pct: self.zoom_label.config(text=f"{pct}%"))

    def _build_param_panel(self) -> None:
        pf = self.param_frame
        ttk.Label(pf, text="Масштаб, мм/px:").pack(anchor="w", padx=4, pady=(4, 0))
        mm_entry = ttk.Entry(pf, textvariable=self.mm_per_px_var, width=18)
        mm_entry.pack(fill=tk.X, padx=4)
        ToolTip(mm_entry, TOOLTIPS.get("mm_per_px", ""))
        ttk.Label(pf, text="Статистика семян:").pack(anchor="w", padx=4, pady=(6, 0))
        ttk.Label(pf, textvariable=self.stats_var, justify="left").pack(anchor="w", padx=4)
        ttk.Separator(pf, orient="horizontal").pack(fill="x", pady=4)
        for text, key, _ in self.checkbox_defs:
            cb = ttk.Checkbutton(pf, text=text, variable=self.vars[key])
            cb.pack(anchor="w", padx=4, pady=1)
            ToolTip(cb, TOOLTIPS.get(key, ""))
        ttk.Separator(pf, orient="horizontal").pack(fill="x", pady=4)
        for key, label, low, high, step, default, kind in self.param_defs:
            frame = ttk.Frame(pf)
            frame.pack(fill=tk.X, padx=4, pady=2)
            lbl = ttk.Label(frame, text=self._format_param(key))
            lbl.pack(anchor="w")
            self.scale_labels[key] = lbl
            scale = ttk.Scale(frame, from_=low, to=high, orient=tk.HORIZONTAL,
                              command=lambda v, k=key: self._on_scale(k, v))
            scale.pack(fill=tk.X)
            scale.set(self.vars[key].get())
            tt = TOOLTIPS.get(key, "")
            if tt:
                ToolTip(lbl, tt)
                ToolTip(scale, tt)

    def _format_param(self, key: str) -> str:
        return format_param_label(self.param_titles[key], self.vars[key].get(), self.param_kinds[key], self.param_steps[key])

    def _on_scale(self, key: str, raw: str) -> None:
        value = round_param_value(raw, self.param_lows[key], self.param_highs[key], self.param_steps[key], self.param_kinds[key])
        self.vars[key].set(value)
        if key in self.scale_labels:
            self.scale_labels[key].config(text=self._format_param(key))

    def get_params(self) -> dict[str, Any]:
        params = {key: var.get() for key, var in self.vars.items()}
        params = normalize_params(params)
        params["mm_per_px"] = parse_mm_per_px_text(self.mm_per_px_var.get())
        return params

    def _set_view(self, view: str) -> None:
        if view == "result" and self.result_bgr is None:
            messagebox.showwarning("Нет результата", "Сначала выполните обработку.")
            return
        if view == "original" and self.original_bgr is None:
            messagebox.showwarning("Нет изображения", "Сначала откройте фото.")
            return
        self.current_view = view
        self._update_canvas()

    def _update_canvas(self) -> None:
        img = self.result_bgr if self.current_view == "result" else self.original_bgr
        if img is None:
            self.zoom_widget.clear()
            return
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.zoom_widget.set_image(Image.fromarray(rgb))

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите фото семян",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Все файлы", "*.*")],
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
        h, w = img.shape[:2]
        self.status_var.set(f"Загружено: {os.path.basename(path)}, {w}x{h}.")
        self.root.after_idle(self._update_canvas)

    def process(self) -> None:
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
            self.root.after_idle(self._update_canvas)
            self.status_var.set(format_seed_status_message(count_seed_statuses(self.seed_items)))
        except Exception as exc:
            messagebox.showerror("Ошибка обработки", str(exc))
            self.status_var.set("Ошибка обработки.")

    def calibrate_ruler_from_image(self) -> None:
        if self.original_bgr is None:
            messagebox.showwarning("Нет изображения", "Сначала откройте фото.")
            return
        self.status_var.set("Калибровка линейки...")
        self.root.update_idletasks()
        mm_per_px, error = try_calibrate_ruler(self.original_bgr)
        if mm_per_px is None:
            messagebox.showerror("Ошибка калибровки", error or "Неизвестная ошибка калибровки.")
            self.status_var.set("Ошибка калибровки линейки.")
            return
        self.mm_per_px_var.set(f"{mm_per_px:.6f}")
        self.status_var.set(f"Калибровка выполнена: {format_calibration_message(mm_per_px)}.")

    def save_crops(self) -> None:
        if self.original_bgr is None:
            messagebox.showwarning("Нет изображения", "Сначала откройте фото.")
            return
        if not self.seed_items:
            messagebox.showwarning("Нет данных", "Нет обнаруженных семян для сохранения.")
            return
        selected = select_items_for_export(
            self.seed_items,
            save_only_focused=self.vars["save_only_focused"].get(),
            skip_blurry=self.vars["skip_blurry"].get(),
        )
        if not selected:
            messagebox.showwarning("Нет данных", "Нет объектов для сохранения по текущим фильтрам.")
            return
        try:
            mm_per_px = parse_mm_per_px_text(self.mm_per_px_var.get())
        except ValueError as exc:
            messagebox.showerror("Ошибка масштаба", str(exc))
            return
        source_name = make_source_name(self.image_path or "")
        try:
            summary = export_seed_items(
                self.original_bgr, selected, self.out_dir, source_name,
                padding=int(self.vars["padding"].get()), mm_per_px=mm_per_px,
            )
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось сохранить фрагменты: {exc}")
            return
        if summary["csv_error"]:
            messagebox.showerror("Ошибка CSV", f"Не удалось сохранить CSV: {summary['csv_error']}")
            return
        self.status_var.set(format_export_message(summary["saved"], summary["selected"], source_name, self.out_dir, summary["contour_errors"]))

    def save_result_image(self) -> None:
        if self.result_bgr is None:
            messagebox.showwarning("Нет результата", "Сначала выполните обработку.")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить результат", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Все файлы", "*.*")],
        )
        if not path:
            return
        if imwrite_unicode(path, self.result_bgr):
            self.status_var.set(f"Результат сохранён: {path}")
        else:
            messagebox.showerror("Ошибка", "Не удалось сохранить результат.")

    def _nav(self, target: str) -> None:
        if self._switch:
            self._switch(target, self.out_dir, 0)
