"""Augmented dataset inspection GUI."""
from __future__ import annotations

import json
import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

from core.csv_utils import read_csv_robust
from core.models import IMAGE_EXTENSIONS
from gui.common.file_scan import scan_directory
from gui.common.zoomable import ZoomableCanvas

CONTOUR_COLORS: dict[str, str] = {
    "Зелёный": "#00FF66", "Жёлтый": "#FFE600", "Бирюзовый": "#00F0FF",
    "Оранжевый": "#FF8800", "Красный": "#FF2244", "Белый": "#FFFFFF",
}


class AugInspectionApp:
    """Visual verification of augmented dataset with filtering."""

    def __init__(self, root: tk.Tk, data_dir: str, switch_callback: Callable[[str], None] | None = None) -> None:
        self.root = root
        self.data_dir = os.path.abspath(data_dir)
        self._switch = switch_callback
        self.records: list[dict[str, str]] = []
        self.filtered_indices: list[int] = []
        self.current_f_idx = 0
        self.image_files: dict[str, str] = {}
        self.json_files: dict[str, str] = {}
        self.current_contour_pts: np.ndarray | None = None
        self.current_json_meta: dict[str, Any] = {}

        self._build_ui()
        self._load_dataset(self.data_dir)
        self._bind_events()

    def _load_dataset(self, folder: str) -> None:
        self.data_dir = folder
        self.image_files, self.json_files = scan_directory(folder)
        self.records.clear()
        if not os.path.isdir(folder):
            self.status_var.set(f"Папка не найдена: {folder}")
            return
        csv_path = None
        for name in ("seeds.csv", "seeds_edited.csv"):
            p = os.path.join(folder, name)
            if os.path.isfile(p):
                csv_path = p
                break
        if not csv_path:
            self.status_var.set(f"CSV не найден в {folder}")
            return
        try:
            _, rows = read_csv_robust(csv_path)
            self.records = rows
        except Exception as exc:
            messagebox.showerror("Ошибка парсинга CSV", str(exc))
            return
        self._apply_filter()
        self.status_var.set(f"Загружено {len(self.records)} записей. Папка: {folder}")

    def _apply_filter(self, preserve_file_index: str | None = None) -> None:
        f_mode = self.filter_mode.get()
        search = self.search_text.get().strip().lower()
        matched: list[int] = []
        for i, r in enumerate(self.records):
            is_aug = str(r.get("is_aug", "0")).strip() in ("1", "True", "true")
            split = r.get("split", "").strip().lower()
            fi = r.get("file_index", "").lower()
            aug_of = r.get("aug_of", "").lower()
            if f_mode == "Только аугментации" and not is_aug:
                continue
            if f_mode == "Только оригиналы" and is_aug:
                continue
            if f_mode == "Train (обучение)" and split != "train":
                continue
            if f_mode == "Val (валидация)" and split != "val":
                continue
            if search and search not in fi and search not in aug_of:
                continue
            matched.append(i)
        self.filtered_indices = matched
        new_pos = 0
        if preserve_file_index:
            for idx, r_idx in enumerate(self.filtered_indices):
                if self.records[r_idx].get("file_index") == preserve_file_index:
                    new_pos = idx
                    break
        self.current_f_idx = min(new_pos, max(0, len(self.filtered_indices) - 1))
        self._update_view()

    def _build_ui(self) -> None:
        self.root.title("Аугментационный инспектор семян")
        self.root.geometry("1400x880")
        self.root.minsize(1080, 680)

        top = ttk.Frame(self.root, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(top, text="📁 Открыть папку...", command=self._choose_folder).pack(side=tk.LEFT, padx=3)
        ttk.Separator(top, orient="vertical").pack(side=tk.LEFT, fill="y", padx=6)

        self.btn_prev = ttk.Button(top, text="◀ Назад", width=9, command=self.prev_item)
        self.btn_prev.pack(side=tk.LEFT, padx=2)
        self.pos_var = tk.StringVar(value="0")
        self.pos_entry = ttk.Entry(top, textvariable=self.pos_var, width=6, justify="center")
        self.pos_entry.pack(side=tk.LEFT, padx=3)
        self.pos_entry.bind("<Return>", self._on_goto_pos)
        self.total_lbl = ttk.Label(top, text="из 0")
        self.total_lbl.pack(side=tk.LEFT, padx=(0, 4))
        self.btn_next = ttk.Button(top, text="Вперёд ▶", width=9, command=self.next_item)
        self.btn_next.pack(side=tk.LEFT, padx=2)

        ttk.Separator(top, orient="vertical").pack(side=tk.LEFT, fill="y", padx=6)
        ttk.Label(top, text="Фильтр:").pack(side=tk.LEFT, padx=(2, 2))
        self.filter_mode = tk.StringVar(value="Все")
        filter_combo = ttk.Combobox(top, textvariable=self.filter_mode,
                                    values=["Все", "Только аугментации", "Только оригиналы", "Train (обучение)", "Val (валидация)"],
                                    width=18, state="readonly")
        filter_combo.pack(side=tk.LEFT, padx=3)
        filter_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_filter())
        ttk.Label(top, text="Поиск:").pack(side=tk.LEFT, padx=(6, 2))
        self.search_text = tk.StringVar(value="")
        search_entry = ttk.Entry(top, textvariable=self.search_text, width=14)
        search_entry.pack(side=tk.LEFT, padx=3)
        search_entry.bind("<KeyRelease>", lambda e: self._apply_filter())
        ttk.Button(top, text="Сброс", width=6, command=self._reset_filters).pack(side=tk.LEFT, padx=2)

        ttk.Separator(top, orient="vertical").pack(side=tk.LEFT, fill="y", padx=6)
        for text, cmd in [("+", lambda: self.zoom_widget.zoom_in()), ("-", lambda: self.zoom_widget.zoom_out()),
                          ("1:1", lambda: self.zoom_widget.zoom_actual()), ("По окну", lambda: self.zoom_widget.zoom_fit())]:
            ttk.Button(top, text=text, width=4, command=cmd).pack(side=tk.LEFT, padx=1)
        self.zoom_lbl = ttk.Label(top, text="100%", width=6, anchor="center")
        self.zoom_lbl.pack(side=tk.LEFT)

        self.btn_family = ttk.Button(top, text="👨‍👧‍👦 Семья", command=self._filter_family)
        self.btn_family.pack(side=tk.RIGHT, padx=4)
        self.btn_orig = ttk.Button(top, text="🔍 К оригиналу", command=self._goto_original)
        self.btn_orig.pack(side=tk.RIGHT, padx=4)

        self.status_var = tk.StringVar(value="Готов к работе")
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w", padding=(6, 3)).pack(side=tk.BOTTOM, fill=tk.X)

        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        canvas_frame = ttk.LabelFrame(paned, text="Изображение", padding=4)
        paned.add(canvas_frame, weight=3)
        self.zoom_widget = ZoomableCanvas(canvas_frame, bg="#1a1a1a")
        self.zoom_widget.set_render_callback(self._draw_overlays)

        info_frame = ttk.Frame(paned, padding=8)
        paned.add(info_frame, weight=1)
        self._build_info_panel(info_frame)

    def _build_info_panel(self, parent: ttk.Frame) -> None:
        vis_group = ttk.LabelFrame(parent, text="Отображение контура", padding=6)
        vis_group.pack(fill=tk.X, pady=(0, 8))
        self.show_contour_var = tk.BooleanVar(value=True)
        self.show_points_var = tk.BooleanVar(value=False)
        self.show_axis_var = tk.BooleanVar(value=True)
        self.show_ruler_var = tk.BooleanVar(value=True)
        self.contour_color_name = tk.StringVar(value="Зелёный")
        for text, var in [("Показывать контур", self.show_contour_var), ("Точки контура", self.show_points_var),
                          ("Ось угла", self.show_axis_var), ("Мм линейка", self.show_ruler_var)]:
            ttk.Checkbutton(vis_group, text=text, variable=var, command=self._redraw).pack(anchor="w", pady=2)
        color_row = ttk.Frame(vis_group)
        color_row.pack(fill=tk.X, pady=4)
        ttk.Label(color_row, text="Цвет:").pack(side=tk.LEFT)
        color_combo = ttk.Combobox(color_row, textvariable=self.contour_color_name, values=list(CONTOUR_COLORS.keys()), width=12, state="readonly")
        color_combo.pack(side=tk.RIGHT)
        color_combo.bind("<<ComboboxSelected>>", lambda e: self._redraw())

        aug_group = ttk.LabelFrame(parent, text="Параметры аугментации", padding=6)
        aug_group.pack(fill=tk.X, pady=(0, 8))
        self.aug_type_lbl = ttk.Label(aug_group, text="Тип: —")
        self.aug_type_lbl.pack(anchor="w", pady=2)
        self.aug_parent_lbl = ttk.Label(aug_group, text="Родитель: —")
        self.aug_parent_lbl.pack(anchor="w", pady=2)
        self.aug_split_lbl = ttk.Label(aug_group, text="Split: —")
        self.aug_split_lbl.pack(anchor="w", pady=2)
        self.aug_params_lbl = ttk.Label(aug_group, text="—", justify="left")
        self.aug_params_lbl.pack(anchor="w", pady=2)

        data_group = ttk.LabelFrame(parent, text="Физические замеры", padding=6)
        data_group.pack(fill=tk.BOTH, expand=True)
        rows = [("file_index", "Файл:"), ("index", "Номер:"), ("status", "Статус:"), ("angle_deg", "Угол, °:"),
                ("length_mm", "Длина, мм:"), ("width_mm", "Ширина, мм:"), ("thickness_mm", "Толщина, мм:"),
                ("mass_g", "Масса, г:"), ("shape", "Форма:"), ("length_px", "Длина, px:"), ("width_px", "Ширина, px:"),
                ("area_px", "Площадь, px:"), ("edge_width_px", "Край, px:"), ("mm_per_px", "мм/px:")]
        self.prop_vars: dict[str, tk.StringVar] = {}
        for r_i, (k, title) in enumerate(rows):
            ttk.Label(data_group, text=title).grid(row=r_i, column=0, sticky="w", pady=1, padx=(2, 6))
            v = tk.StringVar(value="—")
            self.prop_vars[k] = v
            ttk.Label(data_group, textvariable=v).grid(row=r_i, column=1, sticky="w", pady=1)
        data_group.columnconfigure(1, weight=1)

    def _bind_events(self) -> None:
        self.root.bind("<Left>", lambda e: self.prev_item())
        self.root.bind("<Right>", lambda e: self.next_item())

    def prev_item(self) -> None:
        if self.filtered_indices and self.current_f_idx > 0:
            self.current_f_idx -= 1
            self._update_view()

    def next_item(self) -> None:
        if self.filtered_indices and self.current_f_idx < len(self.filtered_indices) - 1:
            self.current_f_idx += 1
            self._update_view()

    def _on_goto_pos(self, event: tk.Event | None = None) -> None:
        try:
            val = int(self.pos_var.get().strip())
            if 1 <= val <= len(self.filtered_indices):
                self.current_f_idx = val - 1
                self._update_view()
        except ValueError:
            pass

    def _reset_filters(self) -> None:
        self.filter_mode.set("Все")
        self.search_text.set("")
        self._apply_filter()

    def _goto_original(self) -> None:
        if not self.filtered_indices:
            return
        r = self.records[self.filtered_indices[self.current_f_idx]]
        target = r.get("aug_of", "").strip() or r.get("file_index", "").strip()
        for f_i, r_idx in enumerate(self.filtered_indices):
            if self.records[r_idx].get("file_index") == target:
                self.current_f_idx = f_i
                self._update_view()
                return
        self.filter_mode.set("Все")
        self.search_text.set("")
        self._apply_filter(preserve_file_index=target)

    def _filter_family(self) -> None:
        if not self.filtered_indices:
            return
        r = self.records[self.filtered_indices[self.current_f_idx]]
        parent = r.get("aug_of", "").strip() or r.get("file_index", "").strip()
        self.filter_mode.set("Все")
        self.search_text.set(parent)
        self._apply_filter(preserve_file_index=r.get("file_index"))

    def _choose_folder(self) -> None:
        d = filedialog.askdirectory(initialdir=self.data_dir, title="Выберите папку с датасетом")
        if d:
            self._load_dataset(d)

    def _update_view(self) -> None:
        n = len(self.filtered_indices)
        if n == 0:
            self.pos_var.set("0")
            self.total_lbl.config(text="из 0")
            self.zoom_widget.clear()
            return
        self.pos_var.set(str(self.current_f_idx + 1))
        self.total_lbl.config(text=f"из {n}")
        record = self.records[self.filtered_indices[self.current_f_idx]]
        fi = record.get("file_index", "").strip()

        for k, v in self.prop_vars.items():
            v.set(record.get(k, "") or "—")

        is_aug = str(record.get("is_aug", "0")).strip() in ("1", "True", "true")
        aug_of = record.get("aug_of", "").strip()
        split = record.get("split", "train").strip()
        self.aug_type_lbl.config(text="🔥 АУГМЕНТАЦИЯ" if is_aug else "⭐ ОРИГИНАЛ")
        self.btn_orig.config(state="normal" if is_aug else "disabled")
        self.aug_parent_lbl.config(text=f"Родитель: {aug_of or '—'}")
        self.aug_split_lbl.config(text=f"Split: {split.upper()}")

        self.current_contour_pts = None
        self.current_json_meta = {}
        # scan_directory keys both dicts by file name *without extension*:
        # the contour JSON for "1_2.jpg" is stored under "1_2_contour".
        json_file = self.json_files.get(f"{fi}_contour".lower()) or self.json_files.get(fi.lower())
        if json_file:
            try:
                with open(os.path.join(self.data_dir, json_file), "r", encoding="utf-8") as f:
                    self.current_json_meta = json.load(f)
                pts = self.current_json_meta.get("points", [])
                if pts:
                    self.current_contour_pts = np.array(pts, dtype=np.float32)
            except:
                pass

        aug_info = self.current_json_meta.get("aug", {})
        if aug_info:
            deg = aug_info.get("rot_angle_deg", "—")
            flip = "Да" if aug_info.get("flip_h") else "Нет"
            q = aug_info.get("jpeg_q", "—")
            self.aug_params_lbl.config(text=f"Поворот: {deg}°\nFlip: {flip}\nJPEG Q: {q}")
        else:
            self.aug_params_lbl.config(text="Оригинальный снимок")

        img_name = self.image_files.get(fi.lower())
        if img_name:
            try:
                self.zoom_widget.set_image(Image.open(os.path.join(self.data_dir, img_name)).convert("RGB"))
            except:
                self.zoom_widget.clear()
        else:
            self.zoom_widget.clear()
        self.zoom_lbl.config(text=f"{self.zoom_widget.get_zoom_percent()}%")

    def _redraw(self) -> None:
        self.zoom_widget.render()

    def _draw_overlays(self) -> None:
        if self.zoom_widget.image is None:
            return
        img = self.zoom_widget.image
        draw = ImageDraw.Draw(img)
        scale = self.zoom_widget.display_scale
        w_orig, h_orig = img.size

        if self.current_contour_pts is not None and self.show_contour_var.get():
            pts = self.current_contour_pts
            color = CONTOUR_COLORS.get(self.contour_color_name.get(), "#00FF66")
            if len(pts) >= 3:
                poly = [(float(p[0]), float(p[1])) for p in pts]
                draw.line(poly + [poly[0]], fill=color, width=max(1, int(round(1.5 / scale))))
                if self.show_points_var.get():
                    r = max(2, int(round(2.5 / scale)))
                    for px, py in poly:
                        draw.ellipse((px - r, py - r, px + r, py + r), fill="#FF0055", outline="#FFFFFF")

            if self.show_axis_var.get():
                rec = self.records[self.filtered_indices[self.current_f_idx]]
                try:
                    ang = float(rec.get("angle_deg", "").replace(",", "."))
                    cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
                    rad = math.radians(ang)
                    axis_len = max(w_orig, h_orig) * 0.4
                    dx, dy = math.cos(rad) * axis_len, math.sin(rad) * axis_len
                    draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill="#00FFFF", width=max(1, int(round(1.5 / scale))))
                    cr = max(3, int(round(3.5 / scale)))
                    draw.ellipse((cx - cr, cy - cr, cx + cr, cy + cr), fill="#00FFFF")
                except (ValueError, TypeError):
                    pass

        if self.show_ruler_var.get():
            rec = self.records[self.filtered_indices[self.current_f_idx]]
            try:
                mm_per_px = float(rec.get("mm_per_px", "").replace(",", "."))
                if mm_per_px > 0:
                    px_per_mm = 1.0 / mm_per_px
                    ruler_x = 10
                    draw.line([(ruler_x, 10), (ruler_x, h_orig - 10)], fill="#FFFFFF", width=2)
                    y, mm = 10.0, 0
                    while y < h_orig - 10:
                        tick_len = 10 if mm % 5 == 0 else 5
                        draw.line([(ruler_x, y), (ruler_x + tick_len, y)], fill="#FFFFFF", width=1)
                        if mm % 5 == 0 and mm > 0:
                            draw.text((ruler_x + 12, y - 5), str(mm), fill="#FFFFFF")
                        y += px_per_mm
                        mm += 1
            except (ValueError, TypeError):
                pass