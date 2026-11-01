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
from core.io_utils import fmt_float
from core.models import DEFAULT_CSV_FIELDS, EXTRA_EDITABLE_FIELDS, IMAGE_EXTENSIONS, SKIP_CSV_FIELDS
from gui.common.file_scan import scan_directory
from gui.common.ruler import draw_mm_ruler
from gui.common.zoomable import ZoomableCanvas, RESAMPLE

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

    def __init__(self, root: tk.Tk, data_dir: str, switch_callback: Callable[[str], None] | None = None) -> None:
        self.root = root
        self.data_dir = data_dir
        self._switch = switch_callback
        self.records: list[dict[str, str]] = []
        self.fieldnames: list[str] = []
        self.current = 0
        self.output_path: str | None = None
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

        self.image_files: dict[str, str] = {}
        self.json_files: dict[str, str] = {}
        self._load_data()
        self._build_ui()
        self._bind_keys()
        self._update_all()

    def _load_data(self) -> None:
        self.image_files, self.json_files = scan_directory(self.data_dir)
        csv_path = os.path.join(self.data_dir, "seeds.csv")
        if not os.path.isfile(csv_path):
            self.load_error = f"Файл не найден: {csv_path}"
            return
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
        for row in rows:
            clean = {k: ("" if v is None else str(v).strip()) for k, v in row.items() if k is not None}
            for col in self.fieldnames:
                clean.setdefault(col, "")
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

        top = ttk.Frame(self.root, padding=(8, 7))
        top.pack(side=tk.TOP, fill=tk.X)
        self.btn_prev = ttk.Button(top, text="← Предыдущая", width=16, command=self.prev_seed)
        self.btn_prev.pack(side=tk.LEFT)
        nav = ttk.Frame(top)
        nav.pack(side=tk.LEFT, padx=10)
        ttk.Label(nav, text="Семечка:").pack(side=tk.LEFT)
        self.pos_var = tk.StringVar(value="0")
        self.pos_entry = ttk.Entry(nav, textvariable=self.pos_var, width=6, justify="center")
        self.pos_entry.pack(side=tk.LEFT, padx=4)
        self.pos_entry.bind("<Return>", self._on_goto)
        self.total_lbl = ttk.Label(nav, text="из 0")
        self.total_lbl.pack(side=tk.LEFT)
        self.btn_next = ttk.Button(top, text="Следующая →", width=16, command=self.next_seed)
        self.btn_next.pack(side=tk.LEFT)

        ttk.Separator(top, orient="vertical").pack(side=tk.LEFT, fill="y", padx=6)
        for text, cmd in [("+", lambda: self.zoom_widget.zoom_in()), ("-", lambda: self.zoom_widget.zoom_out()),
                          ("1:1", lambda: self.zoom_widget.zoom_actual()), ("По окну", lambda: self.zoom_widget.zoom_fit())]:
            ttk.Button(top, text=text, width=4, command=cmd).pack(side=tk.LEFT, padx=1)
        self.zoom_lbl = ttk.Label(top, text="100%", width=6, anchor="center")
        self.zoom_lbl.pack(side=tk.LEFT)

        ttk.Button(top, text="Сохранить контур", command=lambda: self._save_contour(show_message=True)).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(top, text="Сохранить как...", command=self.save_as).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(top, text="Сохранить", command=self.save).pack(side=tk.RIGHT)

        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill='both', expand=True, padx=8, pady=(0, 8))

        img_frame = ttk.LabelFrame(main, text="Изображение", padding=6)
        main.add(img_frame, weight=3)
        self.zoom_widget = ZoomableCanvas(img_frame)
        self.zoom_widget.set_render_callback(self._draw_contour_overlay)
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

    def _build_form(self, parent: ttk.Frame) -> None:
        form = ttk.Frame(parent)
        form.pack(fill=tk.BOTH, expand=True)

        def add_row(row: int, text: str) -> tk.StringVar:
            ttk.Label(form, text=text).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)
            var = tk.StringVar()
            return var

        row = 0
        self.file_index_var = add_row(row, "Файл (file_index):"); row += 1
        self.index_var = add_row(row, "Индекс семечки:"); row += 1

        self.status_var = tk.StringVar()
        ttk.Label(form, text="Статус:").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)
        self.status_box = ttk.Combobox(form, textvariable=self.status_var, values=self.status_options, width=17)
        self.status_box.grid(row=row, column=1, sticky="ew", pady=3)
        self.status_box.bind("<<ComboboxSelected>>", self._mark_dirty)
        self.status_box.bind("<KeyRelease>", self._mark_dirty)
        row += 1

        self.length_var = add_row(row, "Длина, мм:"); row += 1
        self.width_var = add_row(row, "Ширина, мм:"); row += 1
        self.thickness_var = add_row(row, "Толщина, мм:"); row += 1
        self.mass_var = add_row(row, "Масса, г:"); row += 1

        self.shape_var = tk.StringVar()
        ttk.Label(form, text="Форма:").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)
        self.shape_box = ttk.Combobox(form, textvariable=self.shape_var, values=self.shape_options, width=17)
        self.shape_box.grid(row=row, column=1, sticky="ew", pady=3)
        self.shape_box.bind("<<ComboboxSelected>>", self._mark_dirty)
        self.shape_box.bind("<KeyRelease>", self._mark_dirty)
        row += 1

        ttk.Separator(form, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=8); row += 1
        self.length_px_var = add_row(row, "Длина, px:"); row += 1
        self.width_px_var = add_row(row, "Ширина, px:"); row += 1
        self.angle_var = add_row(row, "Угол, °:"); row += 1
        self.area_var = add_row(row, "Площадь, px:"); row += 1
        self.edge_width_var = add_row(row, "Ширина края, px:"); row += 1
        self.mm_per_px_var = add_row(row, "мм/пиксель:"); row += 1

        self._readonly_vars = {
            "file_index": self.file_index_var, "index": self.index_var,
            "length_px": self.length_px_var, "width_px": self.width_px_var,
            "angle_deg": self.angle_var, "area_px": self.area_var,
            "edge_width_px": self.edge_width_var, "mm_per_px": self.mm_per_px_var,
        }
        for var in self._readonly_vars.values():
            pass

        self._entry_vars = {
            "length_mm": self.length_var, "width_mm": self.width_var,
            "thickness_mm": self.thickness_var, "mass_g": self.mass_var,
        }
        for r_i, (key, var) in enumerate(self._entry_vars.items()):
            entry = ttk.Entry(form, textvariable=var, width=20)
            entry.grid(row=list(self._entry_vars.keys()).index(key) + 3, column=1, sticky="ew", pady=3)
            entry.bind("<KeyRelease>", self._mark_dirty_and_render)
            entry.bind("<FocusOut>", self._mark_dirty_and_render)

        for r_i, (key, var) in enumerate(self._readonly_vars.items()):
            entry = ttk.Entry(form, textvariable=var, state="readonly", width=20)
            base_row = list(self._readonly_vars.keys()).index(key)
            entry.grid(row=base_row if base_row < 2 else base_row + 6, column=1, sticky="ew", pady=3)

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
        if self.contour_dirty:
            self._save_contour(show_message=False)
        if self.dirty and not self._save_current(strict=True):
            return
        new_index = self.current + delta
        if 0 <= new_index < len(self.records):
            self.current = new_index
            self._update_all()

    def _on_goto(self, event: tk.Event | None = None) -> None:
        if not self.records:
            return
        if self.contour_dirty:
            self._save_contour(show_message=False)
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
        self.root.after_idle(self._render_image)

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

    def _update_form(self) -> None:
        if not self.records:
            for var in [*self._readonly_vars.values(), *self._entry_vars.values(), self.status_var, self.shape_var]:
                var.set("")
            return
        record = self.records[self.current]
        for key, var in self._readonly_vars.items():
            var.set(record.get(key, ""))
        for key, var in self._entry_vars.items():
            var.set(record.get(key, ""))
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
        if image_path and os.path.isfile(image_path):
            try:
                img = Image.open(image_path).convert("RGB")
            except Exception:
                img = self._make_placeholder(record)
        else:
            img = self._make_placeholder(record)

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
        self.zoom_widget.set_image(img)

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
        if self.zoom_widget.image is not None:
            record = self.records[self.current] if self.records else {}
            mm_per_px = parse_float(record.get("mm_per_px"))
            length_mm = parse_float(self.length_var.get()) or parse_float(record.get("length_mm"))
            width_mm = parse_float(self.width_var.get()) or parse_float(record.get("width_mm"))
            draw_mm_ruler(self.zoom_widget.image, mm_per_px, self.zoom_widget.display_scale, length_mm, width_mm)
            self.zoom_widget.render()
        self.zoom_lbl.config(text=f"{self.zoom_widget.get_zoom_percent()}%")

    def _draw_contour_overlay(self) -> None:
        canvas = self.zoom_widget.canvas
        # Remove items of previous frames first: without this every mouse-move
        # during a drag would leave the old points/lines on the canvas and the
        # widget would quickly freeze under thousands of stacked items.
        canvas.delete("contour")
        self.contour_line_items = []
        self.contour_point_items = []
        if not self.contour_points or self.zoom_widget.display_scale <= 0:
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
            if self.contour_dirty:
                self._save_contour(show_message=False)
            return "break"

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
        self.contour_points.append(pt)
        self.contour_dirty = True
        self._draw_contour_overlay()
        self._save_contour(show_message=False)
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
            self._save_contour(show_message=False)
        return "break"

    def _save_contour(self, show_message: bool = False) -> bool:
        if self.contour_path is None or self.zoom_widget.image is None:
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
            w, h = self.zoom_widget.image.size
            payload["image_width"], payload["image_height"] = int(w), int(h)
            payload["points"] = [[int(round(x)), int(round(y))] for x, y in self.contour_points]
            out_dir = os.path.dirname(self.contour_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(self.contour_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            self.contour_dirty = False
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
        if not self.records:
            messagebox.showwarning("Сохранение", "Нет записей для сохранения.")
            return False
        if not self.output_path:
            return self.save_as()
        return self._write_csv(self.output_path)

    def save_as(self) -> bool:
        if self.dirty and not self._save_current(strict=True):
            return False
        if not self.records:
            messagebox.showwarning("Сохранение", "Нет записей для сохранения.")
            return False
        initial = os.path.basename(self.output_path) if self.output_path else "seeds_edited.csv"
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
            if name not in names:
                names.append(name)
        for name in self.fieldnames:
            if name not in SKIP_CSV_FIELDS and name not in names:
                names.append(name)
        for record in self.records:
            for key in record:
                if key not in SKIP_CSV_FIELDS and key not in names:
                    names.append(key)
        return names

    def _write_csv(self, path: str) -> bool:
        try:
            fieldnames = self._get_fieldnames()
            rows = [{key: record.get(key, "") for key in fieldnames} for record in self.records]
            write_csv(path, fieldnames, rows)
            self.unsaved = False
            self.dirty = False
            self._update_statusbar()
            messagebox.showinfo("Сохранение", f"CSV сохранён:\n{path}")
            return True
        except Exception as exc:
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
        if self.contour_dirty:
            self._save_contour(show_message=False)
        if self.unsaved:
            answer = messagebox.askyesnocancel("Выход", "Сохранить изменения перед выходом?")
            if answer is None:
                return
            if answer and not self.save():
                return
        self.root.destroy()