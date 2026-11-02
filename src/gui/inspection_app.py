"""Inspection GUI: view/edit seed parameters and contours."""
from __future__ import annotations

import json
import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from PIL import Image, ImageDraw

from core.contour_io import parse_raw_points
from core.csv_utils import read_csv_robust, write_csv
import re
from core.io_utils import fmt_float
from core.models import AUG_EXTRA_FIELDS, DEFAULT_CSV_FIELDS, EXTRA_EDITABLE_FIELDS, IMAGE_EXTENSIONS, SKIP_CSV_FIELDS
from core.seed_geometry import compute_median_color_hex
from gui.common.file_scan import scan_directory
from gui.common.ruler import draw_canvas_ruler, draw_mm_ruler
from gui.common.zoomable import ZoomableCanvas, load_image_async, RESAMPLE

POINT_HANDLE_RADIUS = 5
POINT_HIT_RADIUS = 9

BASE_COLUMNS = DEFAULT_CSV_FIELDS
ALL_COLUMNS = BASE_COLUMNS + EXTRA_EDITABLE_FIELDS


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def format_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.10g}"


class InspectionApp:
    """Seed parameter inspection and contour editing."""

    def __init__(
        self,
        root: tk.Tk,
        data_dir: str,
        switch_callback: Callable[..., None] | None = None,
        initial_index: int = 0,
        initial_file_index: str | None = None,
    ) -> None:
        self.root = root
        self.data_dir = os.path.abspath(data_dir)
        self._switch = switch_callback
        self.initial_index = initial_index
        self.initial_file_index = initial_file_index
        self.records: list[dict[str, str]] = []
        self.aug_records: list[dict[str, str]] = []
        self.fieldnames: list[str] = []
        self.current = initial_index
        self.output_path: str = os.path.join(self.data_dir, "seeds.csv")
        self.dirty = False
        self.unsaved = False
        self.load_error: str | None = None
        self.shape_options: list[str] = []
        self.status_options: list[str] = []
        self.contour_points: list[tuple[float, float]] = []
        self.contour_path: str | None = None
        self.contour_meta: dict[str, Any] = {}
        self.contour_dirty = False
        self.drag_point_index: int | None = None
        self.contour_line_items: list[int] = []
        self.contour_point_items: list[int] = []
        # Guards against stale async image decodes (see _load_current_media).
        self._image_gen = 0
        self._last_render_size: tuple[int, int] = (0, 0)

        self.image_files: dict[str, str] = {}
        self.json_files: dict[str, str] = {}
        self._build_ui()
        self._bind_keys()
        # Build the window first, then do CSV/dir-scan work when idle: the
        # app used to feel slow to open because parsing happened before the
        # first frame was ever drawn.
        def _startup() -> None:
            self._load_data()
            if self.records:
                if self.initial_file_index:
                    for i, r in enumerate(self.records):
                        if r.get("file_index") == self.initial_file_index:
                            self.current = i
                            break
                    else:
                        self.current = min(max(0, self.initial_index), len(self.records) - 1)
                else:
                    self.current = min(max(0, self.initial_index), len(self.records) - 1)
            self._update_all()
        self.root.after_idle(_startup)

    def _load_data(self) -> None:
        self.image_files, self.json_files = scan_directory(self.data_dir)
        csv_path = os.path.join(self.data_dir, "seeds.csv")
        if not os.path.isfile(csv_path):
            csv_path = os.path.join(self.data_dir, "seeds_edited.csv")
        if not os.path.isfile(csv_path):
            self.load_error = f"Файл не найден: {os.path.join(self.data_dir, 'seeds.csv')}"
            return
        self.output_path = csv_path
        try:
            fieldnames, rows = read_csv_robust(csv_path)
        except Exception as exc:
            self.load_error = f"Ошибка чтения CSV: {exc}"
            return
        self.fieldnames = list(fieldnames)
        for col in ALL_COLUMNS:
            if col not in self.fieldnames:
                self.fieldnames.append(col)
        self.records = []
        self.aug_records = []
        for row in rows:
            clean = {k: ("" if v is None else str(v).strip()) for k, v in row.items() if k is not None}
            for col in self.fieldnames:
                clean.setdefault(col, "")
            fi = clean.get("file_index", "")
            is_aug = (
                str(clean.get("is_aug", "0")).strip() in ("1", "True", "true")
                or bool(clean.get("aug_of", "").strip())
                or bool(re.search(r"_aug\d+$", fi))
            )
            if is_aug:
                self.aug_records.append(clean)
            else:
                self.records.append(clean)
        shape_set: set[str] = set()
        status_set: set[str] = set()
        for r in self.records:
            if r.get("shape", "").strip():
                shape_set.add(r["shape"].strip())
            if r.get("status", "").strip():
                status_set.add(r["status"].strip())
        self.shape_options = sorted(shape_set)
        self.status_options = sorted(status_set)

    def _build_ui(self) -> None:
        self.root.title("Контроль параметров семечек")
        self.root.geometry("1360x840")
        self.root.minsize(1050, 650)

        self.statusbar = ttk.Label(self.root, text="Загрузка...", anchor="w", padding=(8, 3), relief="sunken")
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        top = ttk.Frame(self.root, padding=(6, 5))
        top.pack(side=tk.TOP, fill=tk.X)

        # 1. Единый блок переключения модулей
        mod_frame = ttk.Frame(top)
        mod_frame.pack(side=tk.LEFT)
        ttk.Button(mod_frame, text="✂ Сегментация", command=lambda: self._nav("segmentation")).pack(side=tk.LEFT, padx=1)
        ttk.Button(mod_frame, text="🔍 Инспектор", state="disabled").pack(side=tk.LEFT, padx=1)
        ttk.Button(mod_frame, text="🔥 Аугментации", command=lambda: self._nav("aug_inspection")).pack(side=tk.LEFT, padx=1)

        ttk.Separator(top, orient="vertical").pack(side=tk.LEFT, fill="y", padx=8)

        # 2. Выбор папки
        ttk.Button(top, text="📁 Папка...", command=self._choose_folder).pack(side=tk.LEFT, padx=2)

        ttk.Separator(top, orient="vertical").pack(side=tk.LEFT, fill="y", padx=6)

        # 3. Навигация по семенам (единый стиль)
        self.btn_prev = ttk.Button(top, text="◀ Назад", width=10, command=self.prev_seed)
        self.btn_prev.pack(side=tk.LEFT, padx=2)
        nav = ttk.Frame(top)
        nav.pack(side=tk.LEFT, padx=4)
        ttk.Label(nav, text="Семечка:").pack(side=tk.LEFT)
        self.pos_var = tk.StringVar(value="0")
        self.pos_entry = ttk.Entry(nav, textvariable=self.pos_var, width=6, justify="center")
        self.pos_entry.pack(side=tk.LEFT, padx=4)
        self.pos_entry.bind("<Return>", self._on_goto)
        self.total_lbl = ttk.Label(nav, text="из 0")
        self.total_lbl.pack(side=tk.LEFT)
        self.btn_next = ttk.Button(top, text="Вперёд ▶", width=10, command=self.next_seed)
        self.btn_next.pack(side=tk.LEFT, padx=2)

        ttk.Separator(top, orient="vertical").pack(side=tk.LEFT, fill="y", padx=6)

        # 4. Единый блок масштабирования
        zoom_frame = ttk.Frame(top)
        zoom_frame.pack(side=tk.LEFT)
        ttk.Button(zoom_frame, text="+", width=3, command=lambda: self.zoom_widget.zoom_in()).pack(side=tk.LEFT, padx=1)
        ttk.Button(zoom_frame, text="-", width=3, command=lambda: self.zoom_widget.zoom_out()).pack(side=tk.LEFT, padx=1)
        ttk.Button(zoom_frame, text="1:1", width=4, command=lambda: self.zoom_widget.zoom_actual()).pack(side=tk.LEFT, padx=1)
        ttk.Button(zoom_frame, text="По окну", command=lambda: self.zoom_widget.zoom_fit()).pack(side=tk.LEFT, padx=1)
        self.zoom_lbl = ttk.Label(zoom_frame, text="100%", width=6, anchor="center")
        self.zoom_lbl.pack(side=tk.LEFT, padx=3)

        # 5. Кнопки сохранения справа
        ttk.Button(top, text="Сохранить контур", command=lambda: self._save_contour(show_message=True)).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(top, text="Сохранить как...", command=self.save_as).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(top, text="Сохранить", command=self.save).pack(side=tk.RIGHT, padx=(4, 0))

        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill='both', expand=True, padx=8, pady=(0, 8))

        img_frame = ttk.LabelFrame(main, text="Изображение", padding=6)
        main.add(img_frame, weight=3)
        self.zoom_widget = ZoomableCanvas(img_frame)
        self.zoom_widget.set_render_callback(self._draw_contour_overlay)
        self.zoom_widget.set_zoom_callback(lambda pct: self.zoom_lbl.config(text=f"{pct}%"))
        # Replace (not add to) the pan bindings that ZoomableCanvas sets up in
        # its constructor:  while a contour point is grabbed our handlers
        # return "break", so LMB drags move *only the point*, never the image;
        # when no point is grabbed they fall back to scan_mark/scan_dragto, so
        # plain panning of the picture still works.
        self.zoom_widget.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.zoom_widget.canvas.bind("<B1-Motion>", self._on_left_motion)
        self.zoom_widget.canvas.bind("<ButtonRelease-1>", self._on_left_release)

        def _pan_mark(event: tk.Event) -> None:
            """Start pan only when no contour point was grabbed."""
            if self.drag_point_index is None:
                self.zoom_widget.canvas.scan_mark(event.x, event.y)

        def _pan_dragto(event: tk.Event) -> None:
            if self.drag_point_index is None:
                self.zoom_widget.canvas.scan_dragto(event.x, event.y, gain=1)

        # add="+" keeps our handlers above; they return "break" whenever a
        # point is being dragged, so these pan handlers never run then.
        self.zoom_widget.canvas.bind("<ButtonPress-1>", _pan_mark, add="+")
        self.zoom_widget.canvas.bind("<B1-Motion>", _pan_dragto, add="+")
        self.zoom_widget.canvas.bind("<Double-Button-1>", self._on_double_click)
        self.zoom_widget.canvas.bind("<Button-3>", self._on_right_click)

        params_frame = ttk.LabelFrame(main, text="Параметры", padding=10)
        main.add(params_frame, weight=1)
        self._build_form(params_frame)

    def _nav(self, target: str) -> None:
        if self.dirty:
            self._save_current(strict=False)
        if self.unsaved:
            ans = messagebox.askyesnocancel(
                "Несохраненные изменения",
                "В таблице есть несохраненные изменения.\nСохранить перед переключением?",
            )
            if ans is None:
                return
            if ans and not self.save():
                return
        if self.contour_dirty:
            ans = messagebox.askyesnocancel(
                "Несохраненный контур",
                "Контур был изменен, но не сохранен кнопкой.\nСохранить контур перед переключением?",
            )
            if ans is None:
                return
            if ans and not self._save_contour(show_message=False):
                return
        if self._switch:
            cur_fi = self.records[self.current].get("file_index") if self.records and 0 <= self.current < len(self.records) else None
            self._switch(target, self.data_dir, self.current, cur_fi)

    def _choose_folder(self) -> None:
        d = filedialog.askdirectory(initialdir=self.data_dir, title="Выберите папку с семенами (out)")
        if d:
            self.data_dir = os.path.abspath(d)
            self._load_data()
            self.current = 0
            self._update_all()

    def _build_form(self, parent: ttk.Frame) -> None:
        form = ttk.Frame(parent)
        form.pack(fill=tk.BOTH, expand=True)

        def add_readonly(text: str, r: int) -> tk.StringVar:
            ttk.Label(form, text=text).grid(row=r, column=0, sticky="w", padx=(0, 10), pady=3)
            var = tk.StringVar()
            entry = ttk.Entry(form, textvariable=var, state="readonly", width=20)
            entry.grid(row=r, column=1, sticky="ew", pady=3)
            return var

        def add_entry(text: str, r: int) -> tk.StringVar:
            ttk.Label(form, text=text).grid(row=r, column=0, sticky="w", padx=(0, 10), pady=3)
            var = tk.StringVar()
            entry = ttk.Entry(form, textvariable=var, width=20)
            entry.grid(row=r, column=1, sticky="ew", pady=3)
            entry.bind("<KeyRelease>", self._mark_dirty_and_render)
            entry.bind("<FocusOut>", self._mark_dirty_and_render)
            return var

        row = 0
        self.file_index_var = add_readonly("Файл (file_index):", row); row += 1
        self.index_var = add_readonly("Индекс семечки:", row); row += 1

        ttk.Label(form, text="Статус:").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)
        self.status_var = tk.StringVar()
        self.status_box = ttk.Combobox(form, textvariable=self.status_var, values=self.status_options, width=17)
        self.status_box.grid(row=row, column=1, sticky="ew", pady=3)
        self.status_box.bind("<<ComboboxSelected>>", self._mark_dirty)
        self.status_box.bind("<KeyRelease>", self._mark_dirty)
        row += 1

        self.length_var = add_entry("Длина, мм:", row); row += 1
        self.width_var = add_entry("Ширина, мм:", row); row += 1
        self.thickness_var = add_entry("Толщина, мм:", row); row += 1
        self.mass_var = add_entry("Масса, г:", row); row += 1

        ttk.Label(form, text="Форма:").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)
        self.shape_var = tk.StringVar()
        self.shape_box = ttk.Combobox(form, textvariable=self.shape_var, values=self.shape_options, width=17)
        self.shape_box.grid(row=row, column=1, sticky="ew", pady=3)
        self.shape_box.bind("<<ComboboxSelected>>", self._mark_dirty)
        self.shape_box.bind("<KeyRelease>", self._mark_dirty)
        row += 1

        ttk.Separator(form, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=8); row += 1

        self.length_px_var = add_readonly("Длина, px:", row); row += 1
        self.width_px_var = add_readonly("Ширина, px:", row); row += 1
        self.angle_var = add_readonly("Угол, °:", row); row += 1
        self.area_var = add_readonly("Площадь, px:", row); row += 1
        self.edge_width_var = add_readonly("Ширина края, px:", row); row += 1
        self.mm_per_px_var = add_readonly("мм/пиксель:", row); row += 1

        ttk.Label(form, text="Цвет:").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)
        color_frame = ttk.Frame(form)
        color_frame.grid(row=row, column=1, sticky="ew", pady=3)
        self.color_hex_var = tk.StringVar()
        self.color_entry = ttk.Entry(color_frame, textvariable=self.color_hex_var, state="readonly", width=10)
        self.color_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.color_badge = tk.Frame(color_frame, width=30, height=20, relief="solid", bd=1, bg="#FFFFFF")
        self.color_badge.pack(side=tk.LEFT, padx=(6, 0))
        self.color_badge.pack_propagate(False)
        row += 1

        self._readonly_vars = {
            "file_index": self.file_index_var, "index": self.index_var,
            "length_px": self.length_px_var, "width_px": self.width_px_var,
            "angle_deg": self.angle_var, "area_px": self.area_var,
            "edge_width_px": self.edge_width_var, "mm_per_px": self.mm_per_px_var,
            "color_hex": self.color_hex_var,
        }

        self._entry_vars = {
            "length_mm": self.length_var, "width_mm": self.width_var,
            "thickness_mm": self.thickness_var, "mass_g": self.mass_var,
        }

        form.columnconfigure(1, weight=1)

    def _bind_keys(self) -> None:
        self.root.bind("<Left>", self._on_left_key)
        self.root.bind("<Right>", self._on_right_key)
        self.root.bind("<Control-s>", lambda e: self.save())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _nav_allowed(self) -> bool:
        try:
            widget = self.root.focus_get()
        except Exception:
            return True
        if widget is None:
            return True
        try:
            return widget.winfo_class() not in ("Entry", "TEntry", "TCombobox", "Combobox")
        except Exception:
            return True

    def _on_left_key(self, event: tk.Event) -> None:
        if self._nav_allowed():
            self.prev_seed()

    def _on_right_key(self, event: tk.Event) -> None:
        if self._nav_allowed():
            self.next_seed()

    def prev_seed(self) -> None:
        self._move(-1)

    def next_seed(self) -> None:
        self._move(1)

    def _move(self, delta: int) -> None:
        if not self.records:
            return
        if self.dirty and not self._save_current(strict=True):
            return
        new_index = self.current + delta
        if 0 <= new_index < len(self.records):
            self.current = new_index
            self._update_all()

    def _on_goto(self, event: tk.Event | None = None) -> None:
        if not self.records:
            return
        if self.dirty and not self._save_current(strict=True):
            return
        try:
            pos = int(self.pos_var.get().strip())
        except ValueError:
            messagebox.showerror("Ошибка перехода", "Введите номер семечки числом.")
            self.pos_var.set(str(self.current + 1))
            return
        if 1 <= pos <= len(self.records):
            self.current = pos - 1
            self._update_all()
        else:
            messagebox.showerror("Ошибка перехода", f"Допустимый диапазон: от 1 до {len(self.records)}.")
            self.pos_var.set(str(self.current + 1))

    def _update_all(self) -> None:
        self._update_nav()
        self._update_form()
        self._load_current_media()
        self._update_statusbar()

    def _update_nav(self) -> None:
        total = len(self.records)
        self.total_lbl.config(text=f"из {total}")
        if total:
            self.pos_var.set(str(self.current + 1))
            self.btn_prev.config(state="normal" if self.current > 0 else "disabled")
            self.btn_next.config(state="normal" if self.current < total - 1 else "disabled")
        else:
            self.pos_var.set("0")
            self.btn_prev.config(state="disabled")
            self.btn_next.config(state="disabled")

    def _update_color_badge(self, hex_code: str) -> None:
        if not hasattr(self, "color_badge"):
            return
        code = str(hex_code).strip()
        if re.match(r"^#[0-9A-Fa-f]{6}$", code):
            try:
                self.color_badge.configure(bg=code)
                return
            except Exception:
                pass
        self.color_badge.configure(bg="#F0F0F0")

    def _recompute_current_color(self) -> str:
        """Compute median color from current image and contour points."""
        src_img = getattr(self.zoom_widget, "_source_image", None) or self.zoom_widget.image
        if src_img is None or not self.contour_points or len(self.contour_points) < 3:
            return ""
        return compute_median_color_hex(src_img, self.contour_points)

    def _on_contour_modified(self) -> None:
        """Update live color display when contour is changed."""
        src_img = getattr(self.zoom_widget, "_source_image", None) or self.zoom_widget.image
        if not self.records or src_img is None or len(self.contour_points) < 3:
            return
        hex_code = self._recompute_current_color()
        if hex_code:
            self.color_hex_var.set(hex_code)
            self._update_color_badge(hex_code)

    def _update_form(self) -> None:
        if not self.records:
            for var in [*self._readonly_vars.values(), *self._entry_vars.values(), self.status_var, self.shape_var]:
                var.set("")
            self._update_color_badge("")
            return
        record = self.records[self.current]
        for key, var in self._readonly_vars.items():
            var.set(record.get(key, ""))
        self._update_color_badge(self.color_hex_var.get())
        for key, var in self._entry_vars.items():
            var.set(record.get(key, ""))
        if not self.length_var.get() and record.get("length_px") and record.get("mm_per_px"):
            l_px = parse_float(record.get("length_px"))
            m_px = parse_float(record.get("mm_per_px"))
            if l_px is not None and m_px is not None and m_px > 0:
                self.length_var.set(f"{l_px * m_px:.4f}")
        if not self.width_var.get() and record.get("width_px") and record.get("mm_per_px"):
            w_px = parse_float(record.get("width_px"))
            m_px = parse_float(record.get("mm_per_px"))
            if w_px is not None and m_px is not None and m_px > 0:
                self.width_var.set(f"{w_px * m_px:.4f}")
        self.status_var.set(record.get("status", ""))
        self.shape_var.set(record.get("shape", ""))

    def _load_current_media(self) -> None:
        self.contour_path = None
        self.contour_meta = {}
        self.contour_points = []
        self.contour_dirty = False
        self.drag_point_index = None
        self.contour_line_items = []
        self.contour_point_items = []
        if not self.records:
            self.zoom_widget.set_image(None)
            return
        record = self.records[self.current]
        image_path = self._find_image_path(record)

        # 1. Load contour data FIRST so self.contour_points is ready
        # before any render callback fires (synchronous cache-hit or async).
        file_index = str(record.get("file_index", "")).strip()
        if not file_index and image_path:
            file_index = os.path.splitext(os.path.basename(image_path))[0]
        file_index = self._base_name(file_index)
        loaded = False
        if file_index:
            # scan_directory keys JSONs by base name without extension.
            actual = self.json_files.get(f"{file_index}_contour".lower())
            if actual:
                loaded = self._load_contour_file(os.path.join(self.data_dir, actual))
            if not loaded:
                # The file may not have been in the scan yet (e.g. just created).
                self.contour_path = os.path.join(self.data_dir, f"{file_index}_contour.json")
                if not os.path.isfile(self.contour_path):
                    self.contour_path = None
        # If the exact "<file_index>_contour.json" was missing, try to locate a
        # contour JSON that matches the image actually found on disk.
        if self.contour_path is None and not loaded and image_path:
            img_base = self._base_name(image_path)
            actual = self.json_files.get(f"{img_base}_contour".lower())
            if actual:
                loaded = self._load_contour_file(os.path.join(self.data_dir, actual))

        # 2. Decode/set the image.
        self._image_gen += 1
        gen = self._image_gen
        if image_path and os.path.isfile(image_path):
            def on_img(img: Image.Image | None, gen=gen, path=image_path) -> None:
                if gen != self._image_gen or not self.root.winfo_exists():
                    return
                self.zoom_widget.set_image(img if img is not None else self._make_placeholder(record))
                self._draw_contour_overlay()
                if not record.get("color_hex") and self.contour_points:
                    c_hex = self._recompute_current_color()
                    if c_hex:
                        record["color_hex"] = c_hex
                        self.color_hex_var.set(c_hex)
                        self._update_color_badge(c_hex)
            load_image_async(image_path, on_img, self.zoom_widget.canvas)
        else:
            self.zoom_widget.set_image(self._make_placeholder(record))

        # 3. Explicitly trigger overlay draw (for synchronous cached loads or immediate display)
        self._draw_contour_overlay()
        src_img = getattr(self.zoom_widget, "_source_image", None) or self.zoom_widget.image
        if not record.get("color_hex") and self.contour_points and src_img is not None:
            c_hex = self._recompute_current_color()
            if c_hex:
                record["color_hex"] = c_hex
                self.color_hex_var.set(c_hex)
                self._update_color_badge(c_hex)

    def _load_contour_file(self, path: str) -> bool:
        """Read contour points from *path*; return True on success."""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return False
        if isinstance(data, dict):
            self.contour_meta = data
            self.contour_points = parse_raw_points(data.get("points", []))
        elif isinstance(data, list):
            self.contour_meta = {"points": data}
            self.contour_points = parse_raw_points(data)
        else:
            return False
        self.contour_path = path
        return True

    @staticmethod
    def _base_name(name: str) -> str:
        """Return *name* without directory part and without a known extension."""
        base = os.path.basename(str(name).strip())
        lowered = base.lower()
        if lowered.endswith(IMAGE_EXTENSIONS) or lowered.endswith(".json"):
            base = os.path.splitext(base)[0]
        return base

    def _image_by_base(self, base: str) -> str | None:
        """Find an image file by its base name (case-insensitive, any extension)."""
        if not base:
            return None
        key = base.lower()
        # Fast path: exact base-name match from the directory scan
        # (scan_directory keys images by name without extension).
        actual = self.image_files.get(key)
        if actual:
            return os.path.join(self.data_dir, actual)
        # Fallback: tolerate stray whitespace / partial names in the CSV.
        for stem, actual in self.image_files.items():
            stem = stem.strip()
            if stem == key or stem.startswith(key + "_") or key.startswith(stem + "_"):
                return os.path.join(self.data_dir, actual)
        return None

    def _find_image_path(self, record: dict[str, str]) -> str | None:
        bases: list[str] = []
        for field in ("file_index", "index"):
            base = self._base_name(record.get(field, ""))
            if base and base not in bases:
                bases.append(base)
        for base in bases:
            found = self._image_by_base(base)
            if found:
                return found
        # Last resort: "<file_index>_<seed index>" naming used by the exporter
        # (e.g. file_index="1", index="3" -> "1_3.jpg").
        fi = self._base_name(record.get("file_index", ""))
        idx = self._base_name(record.get("index", ""))
        if fi and idx and fi != idx:
            return self._image_by_base(f"{fi}_{idx}") or self._image_by_base(f"{idx}_{fi}")
        return None

    def _make_placeholder(self, record: dict[str, str]) -> Image.Image:
        img = Image.new("RGB", (760, 500), (245, 245, 245))
        draw = ImageDraw.Draw(img)
        fi = record.get("file_index", "") or record.get("index", "") or str(self.current + 1)
        draw.text((30, 190), "Изображение не найдено", fill=(140, 0, 0))
        draw.text((30, 218), f"Ожидался файл: {fi}.jpg / {fi}.png", fill=(140, 0, 0))
        return img

    def _render_image(self) -> None:
        """Repaint overlays after a zoom/pan/edit without re-decoding photos.

        Full resize + PhotoImage conversion happens only when the geometry
        actually changed (window resize / zoom buttons); plain edits such as
        typing length/mm_per_px just redraw the cheap overlay layer.
        """
        if self.zoom_widget.image is not None:
            cw, ch = self.zoom_widget.canvas.winfo_width(), self.zoom_widget.canvas.winfo_height()
            if (cw, ch) != self._last_render_size:
                # Geometry changed: rebuild the display bitmap at the new fit.
                self._last_render_size = (cw, ch)
                self.zoom_widget.render()
            self._draw_contour_overlay()
        self.zoom_lbl.config(text=f"{self.zoom_widget.get_zoom_percent()}%")

    def _draw_contour_overlay(self) -> None:
        canvas = self.zoom_widget.canvas
        # Remove items of previous frames first: without this every mouse-move
        # during a drag would leave the old points/lines on the canvas and the
        # widget would quickly freeze under thousands of stacked items.
        canvas.delete("contour")
        self.contour_line_items = []
        self.contour_point_items = []
        if self.zoom_widget.display_scale <= 0:
            return
        scale = self.zoom_widget.display_scale
        if len(self.contour_points) >= 2:
            # Connect every point with its two neighbours along the polyline,
            # and close the figure back to the first point (a segment between
            # the last and the first point is drawn as well).
            coords: list[float] = []
            for x, y in self.contour_points:
                coords.extend([x * scale, y * scale])
            coords.extend([self.contour_points[0][0] * scale, self.contour_points[0][1] * scale])
            self.contour_line_items = [
                canvas.create_line(coords, fill="#000000", width=4, tags=("contour", "contour_line")),
                canvas.create_line(coords, fill="#00ff00", width=2, tags=("contour", "contour_line")),
            ]
        r = POINT_HANDLE_RADIUS
        for i, (x, y) in enumerate(self.contour_points):
            sx, sy = x * scale, y * scale
            item = canvas.create_oval(sx - r, sy - r, sx + r, sy + r, outline="#000000", fill="#ffff00", width=1,
                                      tags=("contour", "contour_pt", f"pt:{i}"))
            self.contour_point_items.append(item)

        # Draw vector ruler
        record = self.records[self.current] if self.records and 0 <= self.current < len(self.records) else {}
        mm_per_px = parse_float(record.get("mm_per_px"))
        length_mm = parse_float(self.length_var.get()) or parse_float(record.get("length_mm"))
        width_mm = parse_float(self.width_var.get()) or parse_float(record.get("width_mm"))
        if not length_mm and mm_per_px and record.get("length_px"):
            length_mm = (parse_float(record.get("length_px")) or 0.0) * mm_per_px
        if not width_mm and mm_per_px and record.get("width_px"):
            width_mm = (parse_float(record.get("width_px")) or 0.0) * mm_per_px
        if mm_per_px and mm_per_px > 0 and self.zoom_widget.image:
            draw_canvas_ruler(
                canvas, mm_per_px, scale, length_mm, width_mm,
                canvas_h=self.zoom_widget.image.size[1], tag="contour"
            )

    def _event_to_image(self, event: tk.Event) -> tuple[float, float] | None:
        if self.zoom_widget.image is None or self.zoom_widget.display_scale <= 0:
            return None
        cx = self.zoom_widget.canvas.canvasx(event.x)
        cy = self.zoom_widget.canvas.canvasy(event.y)
        x, y = cx / self.zoom_widget.display_scale, cy / self.zoom_widget.display_scale
        w, h = self.zoom_widget.image.size
        return max(0.0, min(float(w - 1), x)), max(0.0, min(float(h - 1), y))

    def _find_point_at(self, event: tk.Event) -> int | None:
        canvas = self.zoom_widget.canvas
        cx = canvas.canvasx(event.x)
        cy = canvas.canvasy(event.y)
        # Match against our own point list instead of find_overlapping():
        # the base image item covers the whole canvas and would otherwise be
        # hit by every click (and stale items confuse the index lookup).
        if not self.contour_points or self.zoom_widget.display_scale <= 0:
            return None
        scale = self.zoom_widget.display_scale
        best_index: int | None = None
        best_dist = float(POINT_HIT_RADIUS) + 1.0
        for i, (x, y) in enumerate(self.contour_points):
            dist = math.hypot(x * scale - cx, y * scale - cy)
            if dist <= POINT_HIT_RADIUS and dist < best_dist:
                best_dist, best_index = dist, i
        del canvas  # no canvas items are needed for the lookup
        return best_index

    def _on_left_press(self, event: tk.Event) -> None:
        idx = self._find_point_at(event)
        if idx is not None and 0 <= idx < len(self.contour_points):
            # Grab a point: start dragging it.  Returning "break" prevents the
            # pan bindings (scan_mark/scan_dragto) from also moving the whole
            # image while we drag the point — only the point moves.
            self.drag_point_index = idx
            self._update_point_handles()
            return "break"
        # No point under the cursor: let the image be panned instead.  The
        # companion _pan_mark/_pan_dragto handlers bound with add="+" do the
        # actual scan_mark/scan_dragto; this handler just does nothing.
        return None

    def _update_point_handles(self) -> None:
        """Move only the dragged handle and the contour lines (fast drag path)."""
        if self.drag_point_index is None:
            return
        scale = self.zoom_widget.display_scale
        canvas = self.zoom_widget.canvas
        x, y = self.contour_points[self.drag_point_index]
        sx, sy = x * scale, y * scale
        r = POINT_HANDLE_RADIUS
        item = self.contour_point_items[self.drag_point_index]
        canvas.coords(item, sx - r, sy - r, sx + r, sy + r)
        coords: list[float] = []
        for px, py in self.contour_points:
            coords.extend([px * scale, py * scale])
        # Keep the closing segment (last point -> first point) in sync with
        # the dragged point, exactly like _draw_contour_overlay builds it.
        coords.extend([self.contour_points[0][0] * scale, self.contour_points[0][1] * scale])
        for line_item in self.contour_line_items:
            canvas.coords(line_item, *coords)

    def _on_left_motion(self, event: tk.Event) -> None:
        if self.drag_point_index is not None:
            pt = self._event_to_image(event)
            if pt and 0 <= self.drag_point_index < len(self.contour_points):
                self.contour_points[self.drag_point_index] = pt
                self.contour_dirty = True
                self._update_point_handles()
            # Never let the pan handler move the image during a point drag.
            return "break"

    def _on_left_release(self, event: tk.Event) -> None:
        if self.drag_point_index is not None:
            self.drag_point_index = None
            self._on_contour_modified()
            return "break"

    @staticmethod
    def _dist_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
        """Distance from point P to segment AB (2D)."""
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        len2 = vx * vx + vy * vy
        if len2 <= 0.0:
            return math.hypot(wx, wy)
        t = max(0.0, min(1.0, (wx * vx + wy * vy) / len2))
        return math.hypot(wx - t * vx, wy - t * vy)

    def _insertion_index(self, pt: tuple[float, float]) -> int:
        """Index at which a new point must be inserted so that it connects to
        its two nearest neighbours along the contour (the endpoints of the
        closest existing segment), not to the first/last points of the list.

        With fewer than 2 points there is no segment to attach to, so the
        point is simply appended.
        """
        pts = self.contour_points
        n = len(pts)
        if n < 2:
            return n
        px, py = pt
        best_seg, best_d = 0, float("inf")
        for i in range(n):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % n]  # closing segment last->first included
            d = self._dist_to_segment(px, py, ax, ay, bx, by)
            if d < best_d:
                best_d, best_seg = d, i
        return best_seg + 1

    def _on_double_click(self, event: tk.Event) -> None:
        if self.zoom_widget.image is None:
            return
        if self._find_point_at(event) is not None:
            return
        pt = self._event_to_image(event)
        if pt is None:
            return
        if self.contour_path is None:
            # No existing JSON yet: create one next to the image so that the
            # new contour actually has somewhere to be saved.
            record = self.records[self.current] if self.records else {}
            file_index = str(record.get("file_index", "")).strip()
            if not file_index:
                file_index = os.path.splitext(os.path.basename(self._find_image_path(record) or ""))[0]
            file_index = self._base_name(file_index)
            if not file_index:
                messagebox.showwarning("Контур", "Не удалось определить имя файла для сохранения контура.")
                return
            self.contour_path = os.path.join(self.data_dir, f"{file_index}_contour.json")
            self.contour_meta = {}
        # Insert the point after its two nearest neighbours (the endpoints of
        # the closest existing contour segment) instead of appending it to the
        # end of the list, which used to wire the new point straight to the
        # first and last points of the contour.
        self.contour_points.insert(self._insertion_index(pt), pt)
        self.contour_dirty = True
        self._draw_contour_overlay()
        self._on_contour_modified()
        return "break"

    def _on_right_click(self, event: tk.Event) -> None:
        idx = self._find_point_at(event)
        if idx is not None and 0 <= idx < len(self.contour_points):
            # Actually delete the point (the old code kept the same canvas
            # items with stale "pt:<i>" tags, which only re-wired the lines
            # between the remaining points).
            del self.contour_points[idx]
            self.contour_dirty = True
            self._draw_contour_overlay()
            self._on_contour_modified()
        return "break"

    def _save_contour(self, show_message: bool = False) -> bool:
        src_img = getattr(self.zoom_widget, "_source_image", None) or self.zoom_widget.image
        if self.contour_path is None or src_img is None:
            if show_message:
                messagebox.showwarning("Сохранение контура", "Нет данных для сохранения.")
            return False
        try:
            payload = dict(self.contour_meta) if isinstance(self.contour_meta, dict) else {}
            record = self.records[self.current] if self.records else {}
            fi = str(record.get("file_index", "")).strip()
            if not fi:
                base = os.path.basename(self.contour_path)
                fi = base[:-len("_contour.json")] if base.lower().endswith("_contour.json") else os.path.splitext(base)[0]
            payload.update({"file_index": fi, "points_space": payload.get("points_space", "crop")})
            if str(record.get("index", "")).strip():
                payload["index"] = record["index"]
            w, h = src_img.size
            payload["image_width"], payload["image_height"] = int(w), int(h)
            payload["points"] = [[int(round(x)), int(round(y))] for x, y in self.contour_points]
            out_dir = os.path.dirname(self.contour_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(self.contour_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            self.contour_dirty = False

            # Recalculate color from modified contour and update record + UI
            hex_code = self._recompute_current_color()
            if hex_code:
                record["color_hex"] = hex_code
                self.color_hex_var.set(hex_code)
                self._update_color_badge(hex_code)

            # Persist updated color to seeds.csv
            if self.output_path:
                self._write_csv(self.output_path, show_message=False)

            if show_message:
                messagebox.showinfo("Сохранение контура", f"Контур сохранён:\n{self.contour_path}")
            return True
        except Exception as exc:
            if show_message:
                messagebox.showerror("Ошибка сохранения контура", str(exc))
            return False

    def _mark_dirty(self, event: tk.Event | None = None) -> None:
        self.dirty = True
        self.unsaved = True
        self._update_statusbar()

    def _mark_dirty_and_render(self, event: tk.Event | None = None) -> None:
        self._mark_dirty(event)
        # Coalesce rapid keystrokes into one overlay repaint per idle cycle.
        if getattr(self, "_render_after_id", None):
            try:
                self.root.after_cancel(self._render_after_id)
            except Exception:
                pass
        self._render_after_id = self.root.after(80, self._render_image_debounced)

    def _render_image_debounced(self) -> None:
        self._render_after_id = None
        self._render_image()

    def _save_current(self, strict: bool = True) -> bool:
        if not self.records:
            return True
        record = self.records[self.current]

        def optional_float(var: tk.StringVar, label: str) -> tuple[str, bool]:
            text = var.get().strip()
            if not text:
                return "", True
            val = parse_float(text)
            if val is None:
                if strict:
                    messagebox.showerror("Ошибка ввода", f"Поле «{label}» должно быть числом.\nТекущее значение: '{text}'")
                return "", False
            return format_number(val), True

        for var, key, label in [
            (self.length_var, "length_mm", "Длина, мм"),
            (self.width_var, "width_mm", "Ширина, мм"),
            (self.thickness_var, "thickness_mm", "Толщина, мм"),
            (self.mass_var, "mass_g", "Масса, г"),
        ]:
            val, ok = optional_float(var, label)
            if not ok:
                return False
            record[key] = val

        record["shape"] = self.shape_var.get().strip()
        record["status"] = self.status_var.get().strip()
        if record["shape"] and record["shape"] not in self.shape_options:
            self.shape_options.append(record["shape"])
            self.shape_options.sort()
            self.shape_box.configure(values=self.shape_options)
        if record["status"] and record["status"] not in self.status_options:
            self.status_options.append(record["status"])
            self.status_options.sort()
            self.status_box.configure(values=self.status_options)
        if self.dirty:
            self.dirty = False
            self.unsaved = True
        return True

    def save(self) -> bool:
        if self.dirty and not self._save_current(strict=True):
            return False
        if self.contour_dirty:
            self._save_contour(show_message=False)
        if not self.records:
            messagebox.showwarning("Сохранение", "Нет записей для сохранения.")
            return False
        if not self.output_path:
            self.output_path = os.path.join(self.data_dir, "seeds.csv")
        return self._write_csv(self.output_path)

    def save_as(self) -> bool:
        if self.dirty and not self._save_current(strict=True):
            return False
        if self.contour_dirty:
            self._save_contour(show_message=False)
        if not self.records:
            messagebox.showwarning("Сохранение", "Нет записей для сохранения.")
            return False
        initial = os.path.basename(self.output_path) if self.output_path else "seeds.csv"
        path = filedialog.asksaveasfilename(
            title="Сохранить новый CSV", defaultextension=".csv",
            initialdir=self.data_dir, initialfile=initial,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return False
        self.output_path = path
        return self._write_csv(path)

    def _get_fieldnames(self) -> list[str]:
        names: list[str] = []
        for name in ALL_COLUMNS:
            if name not in SKIP_CSV_FIELDS and name not in names:
                names.append(name)
        for name in self.fieldnames:
            if name not in SKIP_CSV_FIELDS and name not in names:
                names.append(name)
        if getattr(self, "aug_records", None):
            for name in AUG_EXTRA_FIELDS:
                if name not in SKIP_CSV_FIELDS and name not in names:
                    names.append(name)
        for record in self.records:
            for key in record:
                if key not in SKIP_CSV_FIELDS and key not in names:
                    names.append(key)
        return [n for n in names if n not in SKIP_CSV_FIELDS and n not in ("length_mm", "width_mm")]

    def _write_csv(self, path: str, show_message: bool = True) -> bool:
        try:
            fieldnames = self._get_fieldnames()
            orig_map = {r.get("file_index"): r for r in self.records if r.get("file_index")}
            for aug in getattr(self, "aug_records", []):
                parent_fi = aug.get("aug_of")
                if parent_fi in orig_map:
                    parent = orig_map[parent_fi]
                    for prop in ("status", "shape", "thickness_mm", "mass_g"):
                        if prop in parent:
                            aug[prop] = parent[prop]
                    if not aug.get("color_hex") and parent.get("color_hex"):
                        aug["color_hex"] = parent["color_hex"]

            rows_to_save = list(self.records) + getattr(self, "aug_records", [])
            rows = [{key: record.get(key, "") for key in fieldnames} for record in rows_to_save]
            write_csv(path, fieldnames, rows)
            self.output_path = path
            self.unsaved = False
            self.dirty = False
            self._update_statusbar()
            if show_message:
                messagebox.showinfo("Сохранение", f"CSV сохранён:\n{path}")
            return True
        except Exception as exc:
            if show_message:
                messagebox.showerror("Ошибка сохранения", str(exc))
            return False

    def _update_statusbar(self) -> None:
        if self.load_error:
            text = self.load_error
        elif not self.records:
            text = "Нет записей. Проверьте папку и файл seeds.csv."
        else:
            record = self.records[self.current]
            text = f"Запись {self.current + 1}/{len(self.records)} | file_index={record.get('file_index', '')}"
            if self.unsaved:
                text += " | есть несохранённые изменения"
        self.statusbar.config(text=text)

    def _on_close(self) -> None:
        if self.dirty:
            self._save_current(strict=False)
        if self.unsaved:
            answer = messagebox.askyesnocancel("Выход", "Сохранить изменения в таблице перед выходом?")
            if answer is None:
                return
            if answer and not self.save():
                return
        if self.contour_dirty:
            answer = messagebox.askyesnocancel(
                "Несохраненный контур",
                "Контур был изменен, но не сохранен кнопкой.\nСохранить контур перед выходом?",
            )
            if answer is None:
                return
            if answer and not self._save_contour(show_message=False):
                return
        self.root.destroy()