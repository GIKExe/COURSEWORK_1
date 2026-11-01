#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspec_aug.py — просмотр и визуальная верификация аугментированного датасета семян.

Особенности:
- Специально ориентирован на папку аугментации (по умолчанию out_aug, либо аргумент командной строки);
- Отображает аугментированные и оригинальные изображения с наложенными контурами;
- Показывает параметры аугментации (угол поворота, отражение, качество JPEG, родительская семечка aug_of);
- Визуализирует ось ориентации семечки (angle_deg) и миллиметровую линейку;
- Фильтрация: Все / Только аугментации / Только оригиналы / Train / Val / Поиск по родительской семечке;
- Быстрый переход «К оригиналу» и просмотр всей группы («Семейство семечки»);
- Плавный зум (Ctrl+колёсико, +, -, 1:1, по окну) и панорамирование (ЛКМ).
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Any

import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont

RESAMPLE = Image.Resampling.LANCZOS

# =========================
# Настройки по умолчанию
# =========================

APP_TITLE = "Аугментационный инспектор семян (inspec_aug)"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) > 1:
    DEFAULT_DIR = os.path.abspath(sys.argv[1])
else:
    # Если есть out_aug — берём его, иначе out
    aug_candidate = os.path.join(BASE_DIR, "out_aug")
    if os.path.isdir(aug_candidate):
        DEFAULT_DIR = aug_candidate
    else:
        DEFAULT_DIR = os.path.join(BASE_DIR, "out")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

CONTOUR_COLORS = {
    "Зелёный": "#00FF66",
    "Жёлтый": "#FFE600",
    "Бирюзовый": "#00F0FF",
    "Оранжевый": "#FF8800",
    "Красный": "#FF2244",
    "Белый": "#FFFFFF",
}


def setup_dpi_awareness():
    """Настройка DPI awareness для чёткого отображения на Windows."""
    if os.name != "nt":
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def read_csv_robust(path: str) -> tuple[list[str], list[dict[str, str]]]:
    """Читает CSV с автоопределением кодировки и диалекта."""
    encodings = ("utf-8-sig", "utf-8", "cp1251", "latin-1")
    last_err = None

    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                sample = f.read(8192)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                except Exception:
                    dialect = csv.excel

                reader = csv.DictReader(f, dialect=dialect)
                fields = list(reader.fieldnames or [])
                rows = []
                for r in reader:
                    rows.append({k: (v or "").strip() for k, v in r.items() if k is not None})
                return fields, rows
        except Exception as e:
            last_err = e
            continue

    if last_err:
        raise last_err
    raise FileNotFoundError(f"Не удалось открыть CSV: {path}")


class AugInspectorApp:
    def __init__(self, root: tk.Tk, data_dir: str):
        self.root = root
        self.data_dir = os.path.abspath(data_dir)

        self.records: list[dict[str, str]] = []
        self.filtered_indices: list[int] = []
        self.current_f_idx: int = 0  # индекс внутри filtered_indices

        self.image_files: dict[str, str] = {}
        self.json_files: dict[str, str] = {}

        # Кэш текущего изображения
        self.current_img_raw: Image.Image | None = None
        self.current_tk_img: ImageTk.PhotoImage | None = None
        self.current_contour_pts: np.ndarray | None = None
        self.current_json_meta: dict[str, Any] = {}

        # Масштабирование
        self.display_scale: float = 1.0
        self.fit_mode: str = "fit"  # "fit" или "manual"

        # Настройки отображения контура
        self.show_contour_var = tk.BooleanVar(value=True)
        self.show_points_var = tk.BooleanVar(value=False)
        self.show_axis_var = tk.BooleanVar(value=True)
        self.show_ruler_var = tk.BooleanVar(value=True)
        self.contour_color_name = tk.StringVar(value="Зелёный")

        # Фильтры
        self.filter_mode = tk.StringVar(value="Все")
        self.search_text = tk.StringVar(value="")

        self._build_ui()
        self._load_dataset(self.data_dir)
        self._bind_events()

    # ==========================================
    # Загрузка датасета
    # ==========================================

    def _load_dataset(self, folder: str):
        self.data_dir = folder
        self.image_files.clear()
        self.json_files.clear()
        self.records.clear()

        if not os.path.isdir(self.data_dir):
            messagebox.showwarning("Папка не найдена", f"Директория не найдена:\n{self.data_dir}")
            self.status_var.set(f"Папка не найдена: {self.data_dir}")
            return

        # Сканируем файлы
        try:
            for name in os.listdir(self.data_dir):
                low = name.lower()
                base, ext = os.path.splitext(low)
                if ext in IMAGE_EXTENSIONS:
                    self.image_files[base] = name
                elif low.endswith("_contour.json"):
                    key = low[:-len("_contour.json")]
                    self.json_files[key] = name
        except Exception as e:
            messagebox.showerror("Ошибка чтения папки", str(e))
            return

        # Ищем CSV
        csv_candidates = ["seeds.csv", "seeds_edited.csv"]
        csv_path = None
        for c in csv_candidates:
            p = os.path.join(self.data_dir, c)
            if os.path.isfile(p):
                csv_path = p
                break

        if not csv_path:
            messagebox.showwarning(
                "CSV не найден",
                f"В папке {self.data_dir} не найден файл seeds.csv.\n"
                "Выполните аугментацию (python augment.py) или укажите другую папку.",
            )
            self.status_var.set(f"CSV не найден в {self.data_dir}")
            return

        try:
            _, rows = read_csv_robust(csv_path)
            self.records = rows
        except Exception as e:
            messagebox.showerror("Ошибка парсинга CSV", str(e))
            return

        self._apply_filter()
        self.status_var.set(
            f"Загружено {len(self.records)} записей из {os.path.basename(csv_path)}. "
            f"Папка: {self.data_dir}"
        )

    # ==========================================
    # Фильтрация
    # ==========================================

    def _apply_filter(self, preserve_file_index: str | None = None):
        f_mode = self.filter_mode.get()
        search = self.search_text.get().strip().lower()

        matched = []
        for i, r in enumerate(self.records):
            is_aug = str(r.get("is_aug", "0")).strip() in ("1", "True", "true")
            split = r.get("split", "").strip().lower()
            fi = r.get("file_index", "").lower()
            aug_of = r.get("aug_of", "").lower()

            # Режим фильтра
            if f_mode == "Только аугментации" and not is_aug:
                continue
            if f_mode == "Только оригиналы" and is_aug:
                continue
            if f_mode == "Train (обучение)" and split != "train":
                continue
            if f_mode == "Val (валидация)" and split != "val":
                continue

            # Текстовый поиск
            if search:
                if search not in fi and search not in aug_of:
                    continue

            matched.append(i)

        self.filtered_indices = matched

        # Пытаемся восстановить позицию
        new_pos = 0
        if preserve_file_index:
            for idx, r_idx in enumerate(self.filtered_indices):
                if self.records[r_idx].get("file_index") == preserve_file_index:
                    new_pos = idx
                    break

        self.current_f_idx = min(new_pos, max(0, len(self.filtered_indices) - 1))
        self._update_view()

    # ==========================================
    # Построение UI
    # ==========================================

    def _build_ui(self):
        self.root.title(APP_TITLE)
        self.root.geometry("1400x880")
        self.root.minsize(1080, 680)

        # Верхняя панель (тулбар)
        top = ttk.Frame(self.root, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)

        # Выбор папки
        ttk.Button(top, text="📁 Открыть папку...", command=self._choose_folder).pack(
            side=tk.LEFT, padx=3
        )

        ttk.Separator(top, orient="vertical").pack(side=tk.LEFT, fill="y", padx=6)

        # Навигация
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

        # Фильтры
        ttk.Label(top, text="Фильтр:").pack(side=tk.LEFT, padx=(2, 2))
        filter_combo = ttk.Combobox(
            top,
            textvariable=self.filter_mode,
            values=[
                "Все",
                "Только аугментации",
                "Только оригиналы",
                "Train (обучение)",
                "Val (валидация)",
            ],
            width=18,
            state="readonly",
        )
        filter_combo.pack(side=tk.LEFT, padx=3)
        filter_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_filter())

        ttk.Label(top, text="Поиск:").pack(side=tk.LEFT, padx=(6, 2))
        search_entry = ttk.Entry(top, textvariable=self.search_text, width=14)
        search_entry.pack(side=tk.LEFT, padx=3)
        search_entry.bind("<KeyRelease>", lambda e: self._apply_filter())

        ttk.Button(top, text="Сброс", width=6, command=self._reset_filters).pack(
            side=tk.LEFT, padx=2
        )

        ttk.Separator(top, orient="vertical").pack(side=tk.LEFT, fill="y", padx=6)

        # Зум
        ttk.Button(top, text="+", width=3, command=self.zoom_in).pack(side=tk.LEFT)
        ttk.Button(top, text="-", width=3, command=self.zoom_out).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="1:1", width=4, command=self.zoom_100).pack(side=tk.LEFT)
        ttk.Button(top, text="По окну", width=8, command=self.zoom_fit).pack(
            side=tk.LEFT, padx=(2, 4)
        )
        self.zoom_lbl = ttk.Label(top, text="100%", width=6, anchor="center")
        self.zoom_lbl.pack(side=tk.LEFT)

        # Кнопка перехода к оригиналу / семье справа
        self.btn_family = ttk.Button(
            top, text="👨‍👧‍👦 Показать всю семью", command=self._filter_family
        )
        self.btn_family.pack(side=tk.RIGHT, padx=4)

        self.btn_orig = ttk.Button(
            top, text="🔍 К оригиналу", command=self._goto_original
        )
        self.btn_orig.pack(side=tk.RIGHT, padx=4)

        # Строка состояния внизу
        self.status_var = tk.StringVar(value="Готов к работе")
        statusbar = ttk.Label(
            self.root, textvariable=self.status_var, relief="sunken", anchor="w", padding=(6, 3)
        )
        statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Главный сплиттер (слева изображение, справа инфо-панель)
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # Левая часть — Canvas с холстом
        canvas_frame = ttk.LabelFrame(
            paned,
            text="Изображение (Ctrl+колёсико — зум, ЛКМ — перемещение, стрелки ◀ ▶ — навигация)",
            padding=4,
        )
        paned.add(canvas_frame, weight=3)

        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg="#1a1a1a", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        vbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        vbar.grid(row=0, column=1, sticky="ns")
        hbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        hbar.grid(row=1, column=0, sticky="ew")

        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Правая часть — Панель параметров и переключателей
        info_frame = ttk.Frame(paned, padding=8)
        paned.add(info_frame, weight=1)

        self._build_info_panel(info_frame)

    def _build_info_panel(self, parent: ttk.Frame):
        # Группа управления визуализацией контура
        vis_group = ttk.LabelFrame(parent, text="Отображение контура", padding=6)
        vis_group.pack(fill=tk.X, pady=(0, 8))

        cb1 = ttk.Checkbutton(
            vis_group,
            text="Показывать контур",
            variable=self.show_contour_var,
            command=self._redraw_image,
        )
        cb1.pack(anchor="w", pady=2)

        cb2 = ttk.Checkbutton(
            vis_group,
            text="Точки контура",
            variable=self.show_points_var,
            command=self._redraw_image,
        )
        cb2.pack(anchor="w", pady=2)

        cb3 = ttk.Checkbutton(
            vis_group,
            text="Ось угла (angle_deg)",
            variable=self.show_axis_var,
            command=self._redraw_image,
        )
        cb3.pack(anchor="w", pady=2)

        cb4 = ttk.Checkbutton(
            vis_group,
            text="Миллиметровая линейка",
            variable=self.show_ruler_var,
            command=self._redraw_image,
        )
        cb4.pack(anchor="w", pady=2)

        color_row = ttk.Frame(vis_group)
        color_row.pack(fill=tk.X, pady=4)
        ttk.Label(color_row, text="Цвет контура:").pack(side=tk.LEFT)
        color_combo = ttk.Combobox(
            color_row,
            textvariable=self.contour_color_name,
            values=list(CONTOUR_COLORS.keys()),
            width=12,
            state="readonly",
        )
        color_combo.pack(side=tk.RIGHT)
        color_combo.bind("<<ComboboxSelected>>", lambda e: self._redraw_image())

        # Группа метаданных аугментации
        aug_group = ttk.LabelFrame(parent, text="Параметры аугментации", padding=6)
        aug_group.pack(fill=tk.X, pady=(0, 8))

        self.aug_type_lbl = ttk.Label(
            aug_group, text="Тип: —", font=("Segoe UI", 10, "bold")
        )
        self.aug_type_lbl.pack(anchor="w", pady=2)

        self.aug_parent_lbl = ttk.Label(aug_group, text="Родитель (aug_of): —")
        self.aug_parent_lbl.pack(anchor="w", pady=2)

        self.aug_split_lbl = ttk.Label(aug_group, text="Split: —")
        self.aug_split_lbl.pack(anchor="w", pady=2)

        self.aug_params_lbl = ttk.Label(
            aug_group, text="Поворот / Отражение / JPEG:\n—", justify="left"
        )
        self.aug_params_lbl.pack(anchor="w", pady=2)

        # Группа физических параметров
        data_group = ttk.LabelFrame(parent, text="Физические замеры семени", padding=6)
        data_group.pack(fill=tk.BOTH, expand=True)

        rows = [
            ("file_index", "Файл (file_index):"),
            ("index", "Номер семени:"),
            ("status", "Фокус-статус:"),
            ("angle_deg", "Угол оси, °:"),
            ("length_mm", "Длина, мм:"),
            ("width_mm", "Ширина, мм:"),
            ("thickness_mm", "Толщина, мм:"),
            ("mass_g", "Масса, г:"),
            ("shape", "Форма:"),
            ("length_px", "Длина, px:"),
            ("width_px", "Ширина, px:"),
            ("area_px", "Площадь, px:"),
            ("edge_width_px", "Ширина края, px:"),
            ("mm_per_px", "Масштаб мм/px:"),
        ]

        self.prop_vars: dict[str, tk.StringVar] = {}
        for r_i, (k, title) in enumerate(rows):
            ttk.Label(data_group, text=title, font=("Segoe UI", 8)).grid(
                row=r_i, column=0, sticky="w", pady=1, padx=(2, 6)
            )
            v = tk.StringVar(value="—")
            self.prop_vars[k] = v
            lbl = ttk.Label(
                data_group,
                textvariable=v,
                font=("Segoe UI", 8, "bold" if k in ("angle_deg", "status") else "normal"),
            )
            lbl.grid(row=r_i, column=1, sticky="w", pady=1)

        data_group.columnconfigure(1, weight=1)

    # ==========================================
    # Привязка горячих клавиш и событий мыши
    # ==========================================

    def _bind_events(self):
        self.root.bind("<Left>", lambda e: self.prev_item())
        self.root.bind("<Right>", lambda e: self.next_item())

        # Зум по Ctrl + колесо мыши
        self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_wheel)
        self.canvas.bind("<Control-Button-4>", lambda e: self.zoom_in())
        self.canvas.bind("<Control-Button-5>", lambda e: self.zoom_out())

        # Обычный скролл
        self.canvas.bind("<MouseWheel>", self._on_scroll_wheel)
        self.canvas.bind("<Button-4>", self._on_scroll_wheel)
        self.canvas.bind("<Button-5>", self._on_scroll_wheel)

        # Панорамирование перетаскиванием (ЛКМ)
        self.canvas.bind("<ButtonPress-1>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B1-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))

    # ==========================================
    # Навигация
    # ==========================================

    def prev_item(self):
        if not self.filtered_indices:
            return
        if self.current_f_idx > 0:
            self.current_f_idx -= 1
            self._update_view()

    def next_item(self):
        if not self.filtered_indices:
            return
        if self.current_f_idx < len(self.filtered_indices) - 1:
            self.current_f_idx += 1
            self._update_view()

    def _on_goto_pos(self, event=None):
        try:
            val = int(self.pos_var.get().strip())
            if 1 <= val <= len(self.filtered_indices):
                self.current_f_idx = val - 1
                self._update_view()
        except ValueError:
            pass

    def _reset_filters(self):
        self.filter_mode.set("Все")
        self.search_text.set("")
        self._apply_filter()

    def _goto_original(self):
        """Переход к оригинальной семечке для текущей аугментации."""
        if not self.filtered_indices:
            return
        r = self.records[self.filtered_indices[self.current_f_idx]]
        aug_of = r.get("aug_of", "").strip()
        target = aug_of if aug_of else r.get("file_index", "").strip()

        # Ищем запись target в полном списке
        for f_i, r_idx in enumerate(self.filtered_indices):
            if self.records[r_idx].get("file_index") == target:
                self.current_f_idx = f_i
                self._update_view()
                return

        # Если в текущем фильтре её нет — сбрасываем фильтр на "Все" и находим
        self.filter_mode.set("Все")
        self.search_text.set("")
        self._apply_filter(preserve_file_index=target)

    def _filter_family(self):
        """Показывает всю группу одной семечки (оригинал + все аугментации)."""
        if not self.filtered_indices:
            return
        r = self.records[self.filtered_indices[self.current_f_idx]]
        aug_of = r.get("aug_of", "").strip()
        parent = aug_of if aug_of else r.get("file_index", "").strip()

        self.filter_mode.set("Все")
        self.search_text.set(parent)
        self._apply_filter(preserve_file_index=r.get("file_index"))

    def _choose_folder(self):
        d = filedialog.askdirectory(initialdir=self.data_dir, title="Выберите папку с датасетом")
        if d:
            self._load_dataset(d)

    # ==========================================
    # Отрисовка и масштабирование
    # ==========================================

    def _update_view(self):
        n_total = len(self.filtered_indices)
        if n_total == 0:
            self.pos_var.set("0")
            self.total_lbl.config(text="из 0")
            self.canvas.delete("all")
            self.aug_type_lbl.config(text="Нет данных", foreground="#888888")
            return

        self.pos_var.set(str(self.current_f_idx + 1))
        self.total_lbl.config(text=f"из {n_total}")

        rec_idx = self.filtered_indices[self.current_f_idx]
        record = self.records[rec_idx]
        fi = record.get("file_index", "").strip()

        # Обновляем свойства в правой панели
        for k, v in self.prop_vars.items():
            val = record.get(k, "")
            v.set(val if val else "—")

        # Метаданные аугментации
        is_aug = str(record.get("is_aug", "0")).strip() in ("1", "True", "true")
        aug_of = record.get("aug_of", "").strip()
        split = record.get("split", "train").strip()

        if is_aug:
            self.aug_type_lbl.config(text="🔥 АУГМЕНТАЦИЯ", foreground="#FF9900")
            self.btn_orig.config(state="normal")
        else:
            self.aug_type_lbl.config(text="⭐ ОРИГИНАЛ", foreground="#00CC44")
            self.btn_orig.config(state="disabled")

        self.aug_parent_lbl.config(text=f"Родитель (aug_of): {aug_of if aug_of else '—'}")
        self.aug_split_lbl.config(
            text=f"Split: {split.upper()} ({'обучение' if split == 'train' else 'валидация'})"
        )

        # Загружаем JSON контура
        self.current_contour_pts = None
        self.current_json_meta = {}
        json_file = self.json_files.get(fi.lower())
        if json_file:
            jp = os.path.join(self.data_dir, json_file)
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    self.current_json_meta = json.load(f)
                    pts = self.current_json_meta.get("points", [])
                    if pts:
                        self.current_contour_pts = np.array(pts, dtype=np.float32)
            except Exception as e:
                print(f"Ошибка загрузки контура {jp}: {e}")

        # Отображаем инфо об аугментации из JSON
        aug_info = self.current_json_meta.get("aug", {})
        if aug_info:
            deg = aug_info.get("rot_angle_deg", aug_info.get("rot90", "—"))
            flip = "Да (H)" if aug_info.get("flip_h") else "Нет"
            q = aug_info.get("jpeg_q", "—")
            self.aug_params_lbl.config(
                text=f"Угол поворота: {deg}°\nОтражение (flip): {flip}\nJPEG Quality: {q}"
            )
        else:
            self.aug_params_lbl.config(text="Параметры аугментации:\nОригинальный снимок")

        # Загружаем картинку
        self.current_img_raw = None
        img_name = self.image_files.get(fi.lower())
        if img_name:
            ip = os.path.join(self.data_dir, img_name)
            try:
                self.current_img_raw = Image.open(ip).convert("RGB")
            except Exception as e:
                print(f"Ошибка открытия изображения {ip}: {e}")

        if self.fit_mode == "fit":
            self.zoom_fit()
        else:
            self._redraw_image()

    def _redraw_image(self):
        self.canvas.delete("all")
        if self.current_img_raw is None:
            self.canvas.create_text(
                300, 200, text="Изображение не найдено", fill="#FF4444", font=("Segoe UI", 14)
            )
            return

        w_orig, h_orig = self.current_img_raw.size
        scale = max(0.05, min(32.0, self.display_scale))
        self.display_scale = scale

        new_w = max(1, int(round(w_orig * scale)))
        new_h = max(1, int(round(h_orig * scale)))

        # Рендерим изображение с наложенным контуром и линейкой
        img_draw = self.current_img_raw.copy()
        draw = ImageDraw.Draw(img_draw)

        # Рисуем контур на полноразмерном изображении
        color_hex = CONTOUR_COLORS.get(self.contour_color_name.get(), "#00FF66")
        if self.show_contour_var.get() and self.current_contour_pts is not None:
            pts = self.current_contour_pts
            if len(pts) >= 3:
                # Полигональная линия контура
                poly_pts = [(float(p[0]), float(p[1])) for p in pts]
                # Замыкаем
                draw.line(poly_pts + [poly_pts[0]], fill=color_hex, width=max(1, int(round(1.5 / scale))))

                # Точки
                if self.show_points_var.get():
                    r = max(2, int(round(2.5 / scale)))
                    for px, py in poly_pts:
                        draw.ellipse((px - r, py - r, px + r, py + r), fill="#FF0055", outline="#FFFFFF")

                # Ось ориентации (angle_deg)
                if self.show_axis_var.get():
                    rec_idx = self.filtered_indices[self.current_f_idx]
                    angle_str = self.records[rec_idx].get("angle_deg", "").replace(",", ".")
                    try:
                        ang_deg = float(angle_str)
                        cx = float(np.mean(pts[:, 0]))
                        cy = float(np.mean(pts[:, 1]))
                        rad = math.radians(ang_deg)
                        axis_len = max(w_orig, h_orig) * 0.4
                        dx = math.cos(rad) * axis_len
                        dy = math.sin(rad) * axis_len

                        # Линия оси
                        draw.line(
                            [(cx - dx, cy - dy), (cx + dx, cy + dy)],
                            fill="#00FFFF",
                            width=max(1, int(round(1.5 / scale))),
                        )
                        # Центр тяжести
                        cr = max(3, int(round(3.5 / scale)))
                        draw.ellipse((cx - cr, cy - cr, cx + cr, cy + cr), fill="#00FFFF")
                    except ValueError:
                        pass

        # Миллиметровая линейка
        if self.show_ruler_var.get():
            rec_idx = self.filtered_indices[self.current_f_idx]
            mm_px_str = self.records[rec_idx].get("mm_per_px", "").replace(",", ".")
            try:
                mm_per_px = float(mm_px_str)
                if mm_per_px > 0:
                    px_per_mm = 1.0 / mm_per_px
                    # Рисуем шкалу слева
                    ruler_x = 10
                    draw.line([(ruler_x, 10), (ruler_x, h_orig - 10)], fill="#FFFFFF", width=2)
                    y = 10.0
                    mm = 0
                    while y < h_orig - 10:
                        tick_len = 10 if (mm % 5 == 0) else 5
                        draw.line([(ruler_x, y), (ruler_x + tick_len, y)], fill="#FFFFFF", width=1)
                        if mm % 5 == 0 and mm > 0:
                            draw.text((ruler_x + 12, y - 5), f"{mm}", fill="#FFFFFF")
                        y += px_per_mm
                        mm += 1
            except ValueError:
                pass

        # Масштабируем для отображения на Canvas
        if new_w != w_orig or new_h != h_orig:
            img_resized = img_draw.resize((new_w, new_h), RESAMPLE)
        else:
            img_resized = img_draw

        self.current_tk_img = ImageTk.PhotoImage(img_resized)
        self.canvas.create_image(0, 0, anchor="nw", image=self.current_tk_img)
        self.canvas.configure(scrollregion=(0, 0, new_w, new_h))
        self.zoom_lbl.config(text=f"{int(round(scale * 100))}%")

    def zoom_in(self):
        self.fit_mode = "manual"
        self.display_scale *= 1.25
        self._redraw_image()

    def zoom_out(self):
        self.fit_mode = "manual"
        self.display_scale *= 0.8
        self._redraw_image()

    def zoom_100(self):
        self.fit_mode = "manual"
        self.display_scale = 1.0
        self._redraw_image()

    def zoom_fit(self):
        self.fit_mode = "fit"
        if self.current_img_raw is None:
            return
        cw = max(10, self.canvas.winfo_width())
        ch = max(10, self.canvas.winfo_height())
        iw, ih = self.current_img_raw.size
        scale = min(cw / float(iw), ch / float(ih)) * 0.95
        self.display_scale = max(0.1, min(10.0, scale))
        self._redraw_image()

    def _on_canvas_resize(self, event):
        if self.fit_mode == "fit":
            self.zoom_fit()

    def _on_ctrl_wheel(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def _on_scroll_wheel(self, event):
        delta = -1 if (event.delta > 0 or getattr(event, "num", 0) == 4) else 1
        self.canvas.yview_scroll(delta, "units")


def main():
    setup_dpi_awareness()
    root = tk.Tk()
    app = AugInspectorApp(root, DEFAULT_DIR)
    root.mainloop()


if __name__ == "__main__":
    main()
