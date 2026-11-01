#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Seed Inspector — приложение для просмотра и ручной проверки параметров семечек.

Поддерживает формат данных:
    out/
        2_1_001.jpg
        2_1_001_contour.json
        2_1_002.jpg
        2_1_002_contour.json
        seeds.csv

В CSV ожидаются поля:
    file_index,index,status,edge_width_px,length_px,width_px,angle_deg,
    area_px,length_mm,width_mm,mm_per_px

Дополнительно редактируются и сохраняются:
    thickness_mm, mass_g, shape

Возможности:
- изображение семечки можно увеличивать/уменьшать вручную;
- слева рисуется миллиметровая линейка по mm_per_px;
- контур отображается поверх изображения;
- точки контура можно перетаскивать, добавлять и удалять;
- исправленный контур сохраняется обратно в *_contour.json.
"""

import os
import sys
import csv
import math
import json

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk, ImageDraw, ImageFont

RESAMPLE = Image.Resampling.LANCZOS

# =========================
# Пути и константы
# =========================

APP_TITLE = "Контроль параметров семечек"

# Если передать папку аргументом командной строки, используется она.
# Иначе ищется папка out рядом со скриптом.
if len(sys.argv) > 1:
    OUT_DIR = os.path.abspath(sys.argv[1])
else:
    candidates = [
        "out",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out"),
    ]
    OUT_DIR = os.path.abspath("out")
    for c in candidates:
        if os.path.isdir(c):
            OUT_DIR = os.path.abspath(c)
            break

INPUT_CSV = os.path.join(OUT_DIR, "seeds.csv")
DEFAULT_OUTPUT_CSV = os.path.join(OUT_DIR, "seeds_edited.csv")

BASE_COLUMNS = [
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

EXTRA_COLUMNS = [
    "thickness_mm",
    "mass_g",
    "shape",
]

DEFAULT_CSV_FIELDS = BASE_COLUMNS + EXTRA_COLUMNS

# Поля, которые не нужно вытаскивать в итоговый CSV,
# чтобы случайно не сохранить старые пути/имена файлов.
SKIP_CSV_FIELDS = {
    "filename",
    "path",
    "source_path",
}

# При желании сюда можно сразу добавить варианты формы, например:
# SHAPE_OPTIONS = ["овальная", "округлая", "удлиненная", "плоская"]
SHAPE_OPTIONS = []

IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
)

MAX_FALLBACK_WIDTH = 1100
MAX_FALLBACK_HEIGHT = 800

MIN_ZOOM = 0.02
MAX_ZOOM = 32.0

# Ограничение авто-увеличения очень маленьких картинок,
# чтобы не было чрезмерного "мыла".
MAX_FIT_SCALE = 12.0
FIT_PADDING = 26

POINT_HANDLE_RADIUS = 5
POINT_HIT_RADIUS = 9


# =========================
# Вспомогательные функции
# =========================

def parse_float(value):
    """
    Преобразует строку в float.
    Понимает пустую строку, None и запятую как десятичный разделитель.
    """
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


def format_number(value):
    """
    Форматирует число для записи в CSV.
    """
    if value is None:
        return ""
    return f"{float(value):.10g}"


def read_csv_robust(path):
    """
    Читает CSV с несколькими возможными кодировками и пытается сам определить
    разделитель: запятая, точка с запятой или табуляция.
    """
    encodings = ("utf-8-sig", "utf-8", "cp1251", "latin-1")
    last_error = None

    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding, newline="") as fh:
                sample = fh.read(8192)
                fh.seek(0)

                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                except Exception:
                    dialect = csv.excel

                reader = csv.DictReader(fh, dialect=dialect)
                fieldnames = list(reader.fieldnames or [])

                rows = []
                for row in reader:
                    clean_row = {}
                    for key, value in row.items():
                        if key is None:
                            continue
                        clean_row[key] = "" if value is None else str(value).strip()
                    rows.append(clean_row)

                return fieldnames, rows

        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except csv.Error as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error

    raise ValueError("Не удалось прочитать CSV-файл.")


# =========================
# Основное приложение
# =========================

class SeedInspectorApp:
    def __init__(self, root):
        self.root = root

        self.records = []
        self.fieldnames = []

        self.image_files = {}
        self.json_files = {}

        self.current = 0
        self.photo = None
        self.output_path = None

        # dirty — изменения текущей формы ещё не перенесены в records
        # unsaved — есть изменения CSV, которые ещё не сохранены в файл
        self.dirty = False
        self.unsaved = False

        self.load_error = None

        self.shape_options = []
        self.status_options = []

        self._font_cache = None

        # Состояние отображения изображения
        self.base_image = None
        self.display_scale = 1.0

        # zoom_mode:
        #   fit    — подогнать по окну
        #   manual — ручной масштаб
        self.zoom_mode = "fit"
        self.zoom_factor = 1.0

        self._last_canvas_size = (0, 0)

        # Контур и его редактирование
        self.contour_points = []
        self.contour_path = None
        self.contour_meta = {}
        self.contour_dirty = False

        self.drag_point_index = None
        self.pan_active = False

        self.contour_line_items = []
        self.contour_point_items = []

        self._load_data()
        self._build_ui()
        self._bind_keys()
        self._update_all()

    # =========================
    # Загрузка данных
    # =========================

    def _load_data(self):
        self.image_files = {}
        self.json_files = {}

        if not os.path.isdir(OUT_DIR):
            self.load_error = f"Папка не найдена: {OUT_DIR}"
            messagebox.showwarning(
                "Папка не найдена",
                self.load_error + "\n\n"
                "Положите скрипт рядом с папкой out или запустите так:\n"
                "python inspec.py /путь/к/папке/out"
            )
            return

        try:
            for name in os.listdir(OUT_DIR):
                low = name.lower()

                if low.endswith(IMAGE_EXTENSIONS):
                    self.image_files[low] = name
                elif low.endswith(".json"):
                    self.json_files[low] = name
        except Exception:
            self.image_files = {}
            self.json_files = {}

        if not os.path.isfile(INPUT_CSV):
            self.load_error = f"Файл не найден: {INPUT_CSV}"
            messagebox.showwarning("CSV не найден", self.load_error)
            return

        try:
            fieldnames, rows = read_csv_robust(INPUT_CSV)
        except Exception as exc:
            self.load_error = f"Ошибка чтения CSV: {exc}"
            messagebox.showerror("Ошибка CSV", self.load_error)
            return

        self.fieldnames = list(fieldnames)

        # Гарантируем наличие базовых и новых колонок.
        for column in BASE_COLUMNS + EXTRA_COLUMNS:
            if column not in self.fieldnames:
                self.fieldnames.append(column)

        self.records = []

        for row in rows:
            clean_row = {}

            for key, value in row.items():
                if key is None:
                    continue
                clean_row[key] = "" if value is None else str(value).strip()

            for column in self.fieldnames:
                clean_row.setdefault(column, "")

            self.records.append(clean_row)

        # Собираем уже существующие варианты формы и статуса из CSV, если они есть.
        shape_set = set(SHAPE_OPTIONS)
        status_set = set()

        for record in self.records:
            shape = record.get("shape", "").strip()
            if shape:
                shape_set.add(shape)

            status = record.get("status", "").strip()
            if status:
                status_set.add(status)

        self.shape_options = sorted(shape_set)
        self.status_options = sorted(status_set)

    # =========================
    # Интерфейс
    # =========================

    def _build_ui(self):
        self.root.title(APP_TITLE)
        self.root.geometry("1360x840")
        self.root.minsize(1050, 650)

        # Строка состояния снизу.
        self.statusbar = ttk.Label(
            self.root,
            text="Загрузка...",
            anchor="w",
            padding=(8, 3),
            relief="sunken"
        )
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Верхняя панель навигации, зума и сохранения.
        top = ttk.Frame(self.root, padding=(8, 7))
        top.pack(side=tk.TOP, fill=tk.X)

        self.btn_prev = ttk.Button(
            top,
            text="← Предыдущая",
            width=16,
            command=self.prev_seed
        )
        self.btn_prev.pack(side=tk.LEFT)

        nav = ttk.Frame(top)
        nav.pack(side=tk.LEFT, padx=10)

        ttk.Label(nav, text="Семечка:").pack(side=tk.LEFT)

        self.pos_var = tk.StringVar(value="0")
        self.pos_entry = ttk.Entry(
            nav,
            textvariable=self.pos_var,
            width=6,
            justify="center"
        )
        self.pos_entry.pack(side=tk.LEFT, padx=4)
        self.pos_entry.bind("<Return>", self._on_goto)

        self.total_lbl = ttk.Label(nav, text="из 0")
        self.total_lbl.pack(side=tk.LEFT)

        self.btn_next = ttk.Button(
            top,
            text="Следующая →",
            width=16,
            command=self.next_seed
        )
        self.btn_next.pack(side=tk.LEFT)

        ttk.Separator(top, orient="vertical").pack(side=tk.LEFT, fill="y", padx=6)

        # Управление масштабом.
        ttk.Button(top, text="+", width=3, command=self.zoom_in).pack(side=tk.LEFT)
        ttk.Button(top, text="-", width=3, command=self.zoom_out).pack(side=tk.LEFT, padx=(2, 2))
        ttk.Button(top, text="1:1", width=4, command=self.zoom_actual).pack(side=tk.LEFT)
        ttk.Button(top, text="По окну", command=self.zoom_fit).pack(side=tk.LEFT, padx=(4, 6))

        self.zoom_lbl = ttk.Label(top, text="100%", width=6, anchor="center")
        self.zoom_lbl.pack(side=tk.LEFT)

        # Кнопки сохранения справа.
        ttk.Button(
            top,
            text="Сохранить контур",
            command=lambda: self._save_contour(show_message=True)
        ).pack(side=tk.RIGHT, padx=(6, 0))

        ttk.Button(
            top,
            text="Сохранить как...",
            command=self.save_as
        ).pack(side=tk.RIGHT, padx=(6, 0))

        ttk.Button(
            top,
            text="Сохранить",
            command=self.save
        ).pack(side=tk.RIGHT)

        # Основная область: слева изображение, справа параметры.
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        img_frame = ttk.LabelFrame(
            main,
            text="Изображение (Ctrl + колесо — масштаб, ЛКМ по пустому месту — перемещение)",
            padding=6
        )
        main.add(img_frame, weight=3)

        img_frame.columnconfigure(0, weight=1)
        img_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            img_frame,
            bg="#202020",
            highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        vbar = ttk.Scrollbar(img_frame, orient="vertical", command=self.canvas.yview)
        vbar.grid(row=0, column=1, sticky="ns")

        hbar = ttk.Scrollbar(img_frame, orient="horizontal", command=self.canvas.xview)
        hbar.grid(row=1, column=0, sticky="ew")

        self.canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Зум: Ctrl + колесо
        self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)
        self.canvas.bind("<Control-Button-4>", lambda event: self._zoom_by(1.25))
        self.canvas.bind("<Control-Button-5>", lambda event: self._zoom_by(0.8))

        # Обычное колесо — прокрутка
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)
        self.canvas.bind("<Button-5>", self._on_mousewheel)

        # Интерактивное редактирование контура + панорамирование
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_left_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_left_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_left_release)

        self.canvas.bind("<Double-Button-1>", self._on_canvas_double_click)
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)

        params_frame = ttk.LabelFrame(
            main,
            text="Параметры",
            padding=10
        )
        main.add(params_frame, weight=1)

        hint_text = (
            "Новые поля: thickness_mm, mass_g, shape.\n"
            "Стрелки ←/→ — переключение, Ctrl+S — сохранение CSV.\n"
            "Контур: ЛКМ по точке — перетащить; двойной клик — добавить точку;\n"
            "ПКМ по точке — удалить точку. Изменения сохраняются в *_contour.json."
        )

        hint = ttk.Label(
            params_frame,
            text=hint_text,
            foreground="#555555",
            justify="left"
        )
        hint.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

        form = ttk.Frame(params_frame)
        form.pack(fill=tk.BOTH, expand=True)

        def add_caption(text, row):
            ttk.Label(
                form,
                text=text
            ).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)

        def add_readonly(row, width=20):
            var = tk.StringVar()
            entry = ttk.Entry(
                form,
                textvariable=var,
                state="readonly",
                width=width
            )
            entry.grid(row=row, column=1, sticky="ew", pady=3)
            return var

        def add_entry(row, width=20, render_on_change=False):
            var = tk.StringVar()
            entry = ttk.Entry(
                form,
                textvariable=var,
                width=width
            )
            entry.grid(row=row, column=1, sticky="ew", pady=3)

            if render_on_change:
                entry.bind("<KeyRelease>", self._mark_dirty_and_render)
                entry.bind("<FocusOut>", self._mark_dirty_and_render)
            else:
                entry.bind("<KeyRelease>", self._mark_dirty)
                entry.bind("<FocusOut>", self._mark_dirty)

            return entry, var

        row = 0

        add_caption("Файл (file_index):", row)
        self.file_index_var = add_readonly(row)
        row += 1

        add_caption("Индекс семечки:", row)
        self.index_var = add_readonly(row)
        row += 1

        add_caption("Статус:", row)
        self.status_var = tk.StringVar()
        self.status_box = ttk.Combobox(
            form,
            textvariable=self.status_var,
            values=self.status_options,
            width=17
        )
        self.status_box.grid(row=row, column=1, sticky="ew", pady=3)
        self.status_box.bind("<<ComboboxSelected>>", self._mark_dirty)
        self.status_box.bind("<KeyRelease>", self._mark_dirty)
        row += 1

        add_caption("Длина, мм:", row)
        _, self.length_var = add_entry(row, render_on_change=True)
        row += 1

        add_caption("Ширина, мм:", row)
        _, self.width_var = add_entry(row, render_on_change=True)
        row += 1

        add_caption("Толщина, мм:", row)
        _, self.thickness_var = add_entry(row)
        row += 1

        add_caption("Масса, г:", row)
        _, self.mass_var = add_entry(row)
        row += 1

        add_caption("Форма:", row)
        self.shape_var = tk.StringVar()
        self.shape_box = ttk.Combobox(
            form,
            textvariable=self.shape_var,
            values=self.shape_options,
            width=17
        )
        self.shape_box.grid(row=row, column=1, sticky="ew", pady=3)
        self.shape_box.bind("<<ComboboxSelected>>", self._mark_dirty)
        self.shape_box.bind("<KeyRelease>", self._mark_dirty)
        row += 1

        ttk.Separator(
            form,
            orient="horizontal"
        ).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        add_caption("Длина, px:", row)
        self.length_px_var = add_readonly(row)
        row += 1

        add_caption("Ширина, px:", row)
        self.width_px_var = add_readonly(row)
        row += 1

        add_caption("Угол, °:", row)
        self.angle_var = add_readonly(row)
        row += 1

        add_caption("Площадь, px:", row)
        self.area_var = add_readonly(row)
        row += 1

        add_caption("Ширина края, px:", row)
        self.edge_width_var = add_readonly(row)
        row += 1

        add_caption("мм/пиксель:", row)
        self.mm_per_px_var = add_readonly(row)
        row += 1

        form.columnconfigure(1, weight=1)

    def _bind_keys(self):
        self.root.bind("<Left>", self._on_left_key)
        self.root.bind("<Right>", self._on_right_key)
        self.root.bind("<Control-s>", lambda event: self.save())

        self.root.bind("<Control-equal>", lambda event: self.zoom_in())
        self.root.bind("<Control-plus>", lambda event: self.zoom_in())
        self.root.bind("<Control-minus>", lambda event: self.zoom_out())

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _navigation_key_allowed(self):
        """
        Не переключаем семечки стрелками, когда фокус находится
        в поле ввода или в выпадающем списке.
        """
        try:
            widget = self.root.focus_get()
        except Exception:
            return True

        if widget is None:
            return True

        try:
            class_name = widget.winfo_class()
        except Exception:
            return True

        return class_name not in ("Entry", "TEntry", "TCombobox", "Combobox")

    def _on_left_key(self, event):
        if not self._navigation_key_allowed():
            return
        self.prev_seed()

    def _on_right_key(self, event):
        if not self._navigation_key_allowed():
            return
        self.next_seed()

    # =========================
    # Навигация
    # =========================

    def prev_seed(self):
        self._move(-1)

    def next_seed(self):
        self._move(1)

    def _move(self, delta):
        if not self.records:
            return

        if self.contour_dirty:
            self._save_contour(show_message=False)

        # Если пользователь менял текущую запись, сначала переносим изменения
        # в память и проверяем числовые поля.
        if self.dirty and not self._save_current(strict=True):
            return

        new_index = self.current + delta

        if 0 <= new_index < len(self.records):
            self.current = new_index
            self._update_all()

    def _on_goto(self, event=None):
        if not self.records:
            return

        if self.contour_dirty:
            self._save_contour(show_message=False)

        if self.dirty and not self._save_current(strict=True):
            return

        try:
            pos = int(self.pos_var.get().strip())
        except ValueError:
            messagebox.showerror(
                "Ошибка перехода",
                "Введите номер семечки числом."
            )
            self.pos_var.set(str(self.current + 1))
            return

        if 1 <= pos <= len(self.records):
            self.current = pos - 1
            self._update_all()
        else:
            messagebox.showerror(
                "Ошибка перехода",
                f"Допустимый диапазон: от 1 до {len(self.records)}."
            )
            self.pos_var.set(str(self.current + 1))

    # =========================
    # Обновление отображения
    # =========================

    def _update_all(self):
        self._update_nav()
        self._update_form()
        self._load_current_media()
        self._update_statusbar()

        # Рендерим через after_idle, чтобы canvas успел получить размеры.
        self.canvas.after_idle(self._render_image)

    def _update_nav(self):
        total = len(self.records)
        self.total_lbl.config(text=f"из {total}")

        if total:
            self.pos_var.set(str(self.current + 1))
            self.pos_entry.config(state="normal")

            self.btn_prev.config(
                state="normal" if self.current > 0 else "disabled"
            )
            self.btn_next.config(
                state="normal" if self.current < total - 1 else "disabled"
            )
        else:
            self.pos_var.set("0")
            self.pos_entry.config(state="disabled")
            self.btn_prev.config(state="disabled")
            self.btn_next.config(state="disabled")

    def _update_form(self):
        if not self.records:
            for var in (
                self.file_index_var,
                self.index_var,
                self.status_var,
                self.length_var,
                self.width_var,
                self.thickness_var,
                self.mass_var,
                self.shape_var,
                self.length_px_var,
                self.width_px_var,
                self.angle_var,
                self.area_var,
                self.edge_width_var,
                self.mm_per_px_var,
            ):
                var.set("")
            return

        record = self.records[self.current]

        self.file_index_var.set(record.get("file_index", ""))
        self.index_var.set(record.get("index", ""))
        self.status_var.set(record.get("status", ""))

        self.length_var.set(record.get("length_mm", ""))
        self.width_var.set(record.get("width_mm", ""))

        self.thickness_var.set(record.get("thickness_mm", ""))
        self.mass_var.set(record.get("mass_g", ""))
        self.shape_var.set(record.get("shape", ""))

        self.length_px_var.set(record.get("length_px", ""))
        self.width_px_var.set(record.get("width_px", ""))
        self.angle_var.set(record.get("angle_deg", ""))
        self.area_var.set(record.get("area_px", ""))
        self.edge_width_var.set(record.get("edge_width_px", ""))
        self.mm_per_px_var.set(record.get("mm_per_px", ""))

        self._refresh_combo_values()

    def _refresh_combo_values(self):
        # Если в текущей записи есть значение, которого ещё нет в списке,
        # добавляем его в выпадающий список.
        shape_values = list(self.shape_options)
        current_shape = self.shape_var.get().strip()

        if current_shape and current_shape not in shape_values:
            shape_values.append(current_shape)

        self.shape_box.configure(values=shape_values)

        status_values = list(self.status_options)
        current_status = self.status_var.get().strip()

        if current_status and current_status not in status_values:
            status_values.append(current_status)

        self.status_box.configure(values=status_values)

    # =========================
    # Загрузка изображения и контура
    # =========================

    def _load_current_media(self):
        self.contour_path = None
        self.contour_meta = {}
        self.contour_points = []
        self.contour_dirty = False

        self.drag_point_index = None
        self.pan_active = False

        self.contour_line_items = []
        self.contour_point_items = []

        if not self.records:
            self.base_image = None
            return

        record = self.records[self.current]
        image_path = self._find_image_path(record)

        try:
            if image_path and os.path.isfile(image_path):
                img = Image.open(image_path).convert("RGB")
                self.base_image = img

                file_index = self._get_contour_file_index(record, image_path)
                self.contour_path = self._get_contour_path(file_index)

                if self.contour_path and os.path.isfile(self.contour_path):
                    try:
                        with open(self.contour_path, "r", encoding="utf-8") as fh:
                            data = json.load(fh)

                        if isinstance(data, dict):
                            self.contour_meta = data
                            raw_points = data.get("points", [])
                        elif isinstance(data, list):
                            self.contour_meta = {"points": data}
                            raw_points = data
                        else:
                            self.contour_meta = {}
                            raw_points = []

                        self.contour_points = self._parse_raw_points(raw_points)

                    except Exception:
                        self.contour_meta = {}
                        self.contour_points = []
                else:
                    self.contour_meta = {}
                    self.contour_points = []
            else:
                self.base_image = self._make_placeholder_image(record)
                self.contour_path = None
                self.contour_meta = {}
                self.contour_points = []
        except Exception:
            self.base_image = self._make_placeholder_image(record)
            self.contour_path = None
            self.contour_meta = {}
            self.contour_points = []

    def _find_image_path(self, record):
        """
        Ищем изображение по file_index, затем по index,
        затем по номеру текущей строки.
        """
        file_index = str(record.get("file_index", "")).strip()
        index = str(record.get("index", "")).strip()

        candidates = []

        if file_index:
            if file_index.lower().endswith(IMAGE_EXTENSIONS):
                candidates.append(file_index)
            else:
                for ext in IMAGE_EXTENSIONS:
                    candidates.append(file_index + ext)

        if index:
            if index.lower().endswith(IMAGE_EXTENSIONS):
                candidates.append(index)
            else:
                for ext in IMAGE_EXTENSIONS:
                    candidates.append(index + ext)

                if index.isdigit():
                    number = int(index)
                    candidates.extend([
                        f"{number}.jpg",
                        f"{number}.jpeg",
                        f"{number}.png",
                        f"{number:02d}.jpg",
                        f"{number:02d}.png",
                        f"{number:03d}.jpg",
                        f"{number:03d}.png",
                        f"{number:04d}.jpg",
                        f"{number:04d}.png",
                    ])

        pos1 = self.current + 1
        candidates.extend([
            f"{pos1}.jpg",
            f"{pos1}.jpeg",
            f"{pos1}.png",
            f"{pos1:02d}.jpg",
            f"{pos1:02d}.png",
            f"{pos1:03d}.jpg",
            f"{pos1:03d}.png",
            f"{pos1:04d}.jpg",
            f"{pos1:04d}.png",
        ])

        seen = set()
        unique_candidates = []

        for candidate in candidates:
            key = candidate.lower()
            if key and key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)

        for candidate in unique_candidates:
            actual_name = self.image_files.get(candidate.lower())
            if actual_name:
                return os.path.join(OUT_DIR, actual_name)

        return None

    def _get_contour_file_index(self, record, image_path):
        file_index = str(record.get("file_index", "")).strip()

        if not file_index and image_path:
            file_index = os.path.splitext(os.path.basename(image_path))[0]

        if not file_index:
            return None

        if file_index.lower().endswith(IMAGE_EXTENSIONS):
            file_index = os.path.splitext(file_index)[0]

        return file_index

    def _get_contour_path(self, file_index):
        if not file_index:
            return None

        contour_name = f"{file_index}_contour.json"
        actual_name = self.json_files.get(contour_name.lower())

        if actual_name:
            return os.path.join(OUT_DIR, actual_name)

        return os.path.join(OUT_DIR, contour_name)

    def _parse_raw_points(self, raw_points):
        parsed = []

        if not isinstance(raw_points, (list, tuple)):
            return parsed

        for point in raw_points:
            try:
                if isinstance(point, dict):
                    x = point.get("x")
                    y = point.get("y")
                elif isinstance(point, (list, tuple)) and len(point) >= 2:
                    x = point[0]
                    y = point[1]
                else:
                    continue

                x = float(x)
                y = float(y)

                if math.isfinite(x) and math.isfinite(y):
                    parsed.append((x, y))
            except Exception:
                continue

        return parsed

    def _make_placeholder_image(self, record):
        file_index = str(record.get("file_index", "")) or str(record.get("index", "")) or str(self.current + 1)

        img = Image.new("RGB", (760, 500), (245, 245, 245))
        draw = ImageDraw.Draw(img)
        font = self._get_font()

        lines = [
            "Изображение не найдено",
            f"Ожидался файл: {file_index}.jpg / {file_index}.png",
            "Проверьте папку out и поля file_index/index в seeds.csv.",
        ]

        y = 190
        for line in lines:
            draw.text((30, y), line, fill=(140, 0, 0), font=font)
            y += 28

        return img

    # =========================
    # Масштабирование
    # =========================

    def _clamp_zoom(self, scale):
        return max(MIN_ZOOM, min(MAX_ZOOM, float(scale)))

    def _compute_display_scale(self, orig_w, orig_h):
        if self.zoom_mode == "manual":
            return self._clamp_zoom(self.zoom_factor)

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w <= 20 or canvas_h <= 20:
            canvas_w = MAX_FALLBACK_WIDTH
            canvas_h = MAX_FALLBACK_HEIGHT

        avail_w = max(80, canvas_w - FIT_PADDING)
        avail_h = max(80, canvas_h - FIT_PADDING)

        scale = min(avail_w / float(orig_w), avail_h / float(orig_h))
        scale = min(scale, MAX_FIT_SCALE)

        return self._clamp_zoom(scale)

    def zoom_in(self):
        self._zoom_by(1.25)

    def zoom_out(self):
        self._zoom_by(0.8)

    def zoom_actual(self):
        if self.base_image is None:
            return

        self.zoom_mode = "manual"
        self.zoom_factor = 1.0
        self._render_image()

    def zoom_fit(self):
        if self.base_image is None:
            return

        self.zoom_mode = "fit"
        self._render_image()

    def _zoom_by(self, factor):
        if self.base_image is None:
            return

        current = self.display_scale if self.display_scale > 0 else 1.0
        new_scale = self._clamp_zoom(current * float(factor))

        self.zoom_mode = "manual"
        self.zoom_factor = new_scale

        self._render_image()

    def _on_canvas_configure(self, event):
        size = (event.width, event.height)

        if size == self._last_canvas_size:
            return

        self._last_canvas_size = size

        if self.base_image is not None and self.zoom_mode == "fit":
            self._render_image()

    def _on_ctrl_mousewheel(self, event):
        factor = 1.25 if event.delta > 0 else 0.8
        self._zoom_by(factor)

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

    # =========================
    # Отрисовка
    # =========================

    def _render_image(self):
        self.canvas.delete("all")

        self.contour_line_items = []
        self.contour_point_items = []

        if self.base_image is None:
            self.zoom_lbl.config(text="100%")
            return

        orig_w, orig_h = self.base_image.size

        if orig_w <= 0 or orig_h <= 0:
            self.zoom_lbl.config(text="100%")
            return

        scale = self._compute_display_scale(orig_w, orig_h)
        self.display_scale = scale

        disp_w = max(1, int(round(orig_w * scale)))
        disp_h = max(1, int(round(orig_h * scale)))

        img = self.base_image.resize((disp_w, disp_h), RESAMPLE)
        draw = ImageDraw.Draw(img)

        record = {}
        length_override = None
        width_override = None

        if self.records:
            record = self.records[self.current]
            length_override = parse_float(self.length_var.get())
            width_override = parse_float(self.width_var.get())

        self._draw_mm_ruler(
            img=img,
            record=record,
            scale=scale,
            length_override=length_override,
            width_override=width_override
        )

        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo, tags="base_image")

        self._draw_contour_overlay()

        self.canvas.configure(scrollregion=(0, 0, disp_w, disp_h))
        self.zoom_lbl.config(text=f"{int(round(scale * 100))}%")

    def _draw_contour_overlay(self):
        self.contour_line_items = []
        self.contour_point_items = []

        if not self.contour_points or self.display_scale <= 0:
            return

        scale = self.display_scale

        # Линия контура.
        if len(self.contour_points) >= 2:
            coords = []

            for x, y in self.contour_points:
                coords.extend([x * scale, y * scale])

            # Замыкаем контур, если точек больше двух.
            if len(self.contour_points) > 2:
                x0, y0 = self.contour_points[0]
                coords.extend([x0 * scale, y0 * scale])

            black_line = self.canvas.create_line(
                coords,
                fill="#000000",
                width=4,
                tags=("contour", "contour_line")
            )

            green_line = self.canvas.create_line(
                coords,
                fill="#00ff00",
                width=2,
                tags=("contour", "contour_line")
            )

            self.contour_line_items = [black_line, green_line]

        # Точки контура.
        r = POINT_HANDLE_RADIUS

        for i, (x, y) in enumerate(self.contour_points):
            sx = x * scale
            sy = y * scale

            item = self.canvas.create_oval(
                sx - r,
                sy - r,
                sx + r,
                sy + r,
                outline="#000000",
                fill="#ffff00",
                width=1,
                tags=("contour", "contour_pt", f"pt:{i}")
            )

            self.contour_point_items.append(item)

    def _refresh_contour_line(self):
        if not self.contour_line_items or not self.contour_points:
            return

        scale = self.display_scale
        coords = []

        for x, y in self.contour_points:
            coords.extend([x * scale, y * scale])

        if len(self.contour_points) > 2:
            x0, y0 = self.contour_points[0]
            coords.extend([x0 * scale, y0 * scale])

        for item in self.contour_line_items:
            try:
                self.canvas.coords(item, *coords)
            except Exception:
                pass

    def _update_point_visual(self, index):
        if index < 0 or index >= len(self.contour_point_items):
            return

        if index >= len(self.contour_points):
            return

        item = self.contour_point_items[index]
        x, y = self.contour_points[index]

        sx = x * self.display_scale
        sy = y * self.display_scale
        r = POINT_HANDLE_RADIUS

        try:
            self.canvas.coords(item, sx - r, sy - r, sx + r, sy + r)
        except Exception:
            pass

        self._refresh_contour_line()

    # =========================
    # Интерактивное редактирование контура
    # =========================

    def _event_to_image_coords(self, event):
        if self.base_image is None or self.display_scale <= 0:
            return None, None

        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)

        x = cx / self.display_scale
        y = cy / self.display_scale

        w, h = self.base_image.size

        x = max(0.0, min(float(w - 1), float(x)))
        y = max(0.0, min(float(h - 1), float(y)))

        return x, y

    def _find_point_item_at(self, event):
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)

        r = POINT_HIT_RADIUS

        items = self.canvas.find_overlapping(cx - r, cy - r, cx + r, cy + r)

        best_item = None
        best_dist = None

        for item in items:
            tags = self.canvas.gettags(item)

            if "contour_pt" not in tags:
                continue

            coords = self.canvas.coords(item)

            if len(coords) != 4:
                continue

            center_x = (coords[0] + coords[2]) / 2.0
            center_y = (coords[1] + coords[3]) / 2.0

            dist = math.hypot(center_x - cx, center_y - cy)

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_item = item

        return best_item

    def _point_index_from_item(self, item):
        tags = self.canvas.gettags(item)

        for tag in tags:
            if tag.startswith("pt:"):
                try:
                    return int(tag.split(":", 1)[1])
                except Exception:
                    return None

        return None

    def _on_canvas_left_press(self, event):
        if self.base_image is None:
            return

        item = self._find_point_item_at(event)

        if item is not None:
            index = self._point_index_from_item(item)

            if index is not None and 0 <= index < len(self.contour_points):
                self.drag_point_index = index
                return

        # Если точка не найдена — начинаем панорамирование.
        self.pan_active = True
        self.canvas.scan_mark(event.x, event.y)

    def _on_canvas_left_motion(self, event):
        if self.drag_point_index is not None:
            x, y = self._event_to_image_coords(event)

            if x is None or y is None:
                return

            index = self.drag_point_index

            if 0 <= index < len(self.contour_points):
                self.contour_points[index] = (x, y)
                self.contour_dirty = True
                self._update_point_visual(index)

        elif self.pan_active:
            self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _on_canvas_left_release(self, event):
        if self.drag_point_index is not None:
            self.drag_point_index = None

            if self.contour_dirty:
                self._save_contour(show_message=False)

        self.pan_active = False

    def _on_canvas_double_click(self, event):
        """
        Двойной клик — добавить точку контура.
        """
        if self.base_image is None:
            return

        if self.contour_path is None:
            messagebox.showwarning(
                "Контур",
                "Нет имени файла для контура.\n"
                "Нужно поле file_index или имя изображения."
            )
            return

        # Если двойной клик попал в существующую точку, не добавляем новую.
        if self._find_point_item_at(event) is not None:
            return

        x, y = self._event_to_image_coords(event)

        if x is None or y is None:
            return

        insert_index = self._find_insert_index(x, y)
        self.contour_points.insert(insert_index, (x, y))

        self.contour_dirty = True
        self._draw_contour_overlay()
        self._save_contour(show_message=False)

    def _on_canvas_right_click(self, event):
        """
        ПКМ по точке — удалить точку контура.
        """
        if self.base_image is None:
            return

        item = self._find_point_item_at(event)

        if item is None:
            return

        index = self._point_index_from_item(item)

        if index is not None and 0 <= index < len(self.contour_points):
            del self.contour_points[index]

            self.contour_dirty = True
            self._draw_contour_overlay()
            self._save_contour(show_message=False)

    def _find_insert_index(self, x, y):
        n = len(self.contour_points)

        if n == 0:
            return 0

        if n == 1:
            return 1

        segments = []

        if n == 2:
            segments.append((0, 1))
        else:
            for i in range(n):
                segments.append((i, (i + 1) % n))

        best_i = None
        best_dist = None

        for i, j in segments:
            x1, y1 = self.contour_points[i]
            x2, y2 = self.contour_points[j]

            dist = self._point_segment_distance(x, y, x1, y1, x2, y2)

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_i = i

        if best_i is None:
            return n

        return best_i + 1

    @staticmethod
    def _point_segment_distance(px, py, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1

        length_sq = dx * dx + dy * dy

        if length_sq <= 1e-12:
            return math.hypot(px - x1, py - y1)

        t = ((px - x1) * dx + (py - y1) * dy) / length_sq
        t = max(0.0, min(1.0, t))

        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        return math.hypot(px - proj_x, py - proj_y)

    # =========================
    # Сохранение контура
    # =========================

    def _save_contour(self, show_message=False):
        if self.contour_path is None:
            if show_message:
                messagebox.showwarning(
                    "Сохранение контура",
                    "Нет файла контура для сохранения."
                )
            return False

        if self.base_image is None:
            if show_message:
                messagebox.showwarning(
                    "Сохранение контура",
                    "Нет изображения для сохранения контура."
                )
            return False

        try:
            payload = dict(self.contour_meta) if isinstance(self.contour_meta, dict) else {}

            record = self.records[self.current] if self.records else {}

            file_index = str(record.get("file_index", "")).strip()

            if not file_index:
                base = os.path.basename(self.contour_path)

                if base.lower().endswith("_contour.json"):
                    file_index = base[:-len("_contour.json")]
                else:
                    file_index = os.path.splitext(base)[0]

            payload["file_index"] = file_index

            if str(record.get("index", "")).strip() != "":
                payload["index"] = record.get("index", "")
            elif "index" not in payload:
                payload["index"] = ""

            payload["points_space"] = payload.get("points_space", "crop")

            w, h = self.base_image.size
            payload["image_width"] = int(w)
            payload["image_height"] = int(h)

            points = []

            for x, y in self.contour_points:
                try:
                    xi = int(round(float(x)))
                    yi = int(round(float(y)))
                    points.append([xi, yi])
                except Exception:
                    continue

            payload["points"] = points

            out_dir = os.path.dirname(self.contour_path)

            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            with open(self.contour_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)

            self.json_files[os.path.basename(self.contour_path).lower()] = os.path.basename(self.contour_path)

            self.contour_dirty = False

            if show_message:
                messagebox.showinfo(
                    "Сохранение контура",
                    f"Контур сохранён:\n{self.contour_path}"
                )
            else:
                self.statusbar.config(
                    text=f"Контур сохранён: {os.path.basename(self.contour_path)}"
                )

            return True

        except Exception as exc:
            if show_message:
                messagebox.showerror(
                    "Ошибка сохранения контура",
                    f"Не удалось сохранить контур:\n{exc}"
                )
            else:
                self.statusbar.config(
                    text=f"Ошибка сохранения контура: {exc}"
                )

            return False

    # =========================
    # Линейка
    # =========================

    def _get_font(self):
        if self._font_cache is not None:
            return self._font_cache

        candidates = [
            "arial.ttf",
            "DejaVuSans.ttf",
            "Arial.ttf",
            "verdana.ttf",
        ]

        for name in candidates:
            try:
                font = ImageFont.truetype(name, 16)
                self._font_cache = font
                return font
            except Exception:
                pass

        try:
            # Новые версии Pillow поддерживают размер у load_default.
            font = ImageFont.load_default(size=16)
        except Exception:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

        self._font_cache = font
        return font

    def _choose_ruler_step(self, bar_mm, px_per_mm):
        """
        Выбирает шаг делений линейки.
        """
        steps = [
            0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5,
            1, 2, 5, 10, 20, 50, 100
        ]

        for step in steps:
            if step <= 0:
                continue

            px_per_tick = step * px_per_mm
            tick_count = bar_mm / step

            if px_per_tick >= 10 and tick_count <= 20:
                return step

        return None

    def _draw_mm_ruler(self, img, record, scale, length_override=None, width_override=None):
        """
        Рисует вертикальную миллиметровую линейку слева.
        Длина линейки старается соответствовать длине семечки.
        Без фона, красным цветом.
        """
        width, height = img.size

        if width < 140 or height < 100:
            return

        mm_per_px = parse_float(record.get("mm_per_px"))

        if not mm_per_px or mm_per_px <= 0 or scale <= 0:
            return

        # Пикселей на миллиметр в текущем экранном изображении.
        # Исходный mm_per_px — мм на один пиксель исходного изображения.
        # После масштабирования: display_px_per_mm = scale / mm_per_px.
        px_per_mm = scale / mm_per_px

        if px_per_mm <= 0 or not math.isfinite(px_per_mm):
            return

        length_mm = length_override if length_override and length_override > 0 else parse_float(record.get("length_mm"))
        width_mm = width_override if width_override and width_override > 0 else parse_float(record.get("width_mm"))

        # Основная цель — длина семечки. Если её нет, берём ширину.
        target_mm = None

        if length_mm and length_mm > 0:
            target_mm = float(length_mm)
        elif width_mm and width_mm > 0:
            target_mm = float(width_mm)

        # Если в CSV нет размеров, делаем линейку примерно на половину высоты изображения.
        if not target_mm or target_mm <= 0:
            target_mm = (height * 0.5) / px_per_mm

        max_px = height * 0.82
        min_px = 18.0

        bar_px = target_mm * px_per_mm

        if bar_px > max_px:
            bar_px = max_px
            target_mm = bar_px / px_per_mm

        if bar_px < min_px:
            bar_px = min(min_px, max_px)
            target_mm = bar_px / px_per_mm

        if bar_px < 5:
            return

        step_mm = self._choose_ruler_step(target_mm, px_per_mm)

        draw = ImageDraw.Draw(img)
        font = self._get_font()

        RED = (255, 0, 0)

        label = f"{target_mm:.4g} мм"

        x_line = 24
        tick_len = 14

        y0 = max(16, int((height - bar_px) / 2))
        y1 = y0 + int(bar_px)

        if y1 > height - 16:
            y1 = height - 16
            y0 = max(16, y1 - int(bar_px))

        # Текст без подложки.
        text_y = max(4, y0 - 22)

        if text_y < 4:
            text_y = min(height - 20, y1 + 6)

        draw.text((8, text_y), label, fill=RED, font=font)

        # Основная вертикальная линия.
        draw.line(
            [(x_line, y0), (x_line, y1)],
            fill=RED,
            width=3
        )

        # Верхняя и нижняя граница линейки.
        draw.line(
            [(x_line, y0), (x_line + tick_len + 4, y0)],
            fill=RED,
            width=3
        )

        draw.line(
            [(x_line, y1), (x_line + tick_len + 4, y1)],
            fill=RED,
            width=3
        )

        # Деления.
        if step_mm:
            count = int(math.floor(target_mm / step_mm + 1e-6))

            for i in range(1, count + 1):
                mm = i * step_mm

                if mm > target_mm + 1e-9:
                    break

                yy = y0 + int(mm * px_per_mm)

                if yy > y1:
                    break

                draw.line(
                    [(x_line, yy), (x_line + tick_len, yy)],
                    fill=RED,
                    width=2
                )

                # Подписи делений, если хватает места.
                if px_per_mm * step_mm >= 28 and count <= 12:
                    tick_label = f"{mm:.4g}"
                    draw.text(
                        (x_line + tick_len + 4, yy - 8),
                        tick_label,
                        fill=RED,
                        font=font
                    )

    # =========================
    # Редактирование и контроль
    # =========================

    def _mark_dirty(self, event=None):
        self.dirty = True
        self.unsaved = True
        self._update_statusbar()

    def _mark_dirty_and_render(self, event=None):
        self._mark_dirty(event)
        self._render_image()

    def _save_current(self, strict=True):
        """
        Проверяет поля текущей формы и записывает их в self.records.
        """
        if not self.records:
            return True

        was_dirty = self.dirty
        record = self.records[self.current]

        def optional_float(var, label):
            text = var.get().strip()

            if not text:
                return "", True

            value = parse_float(text)

            if value is None:
                if strict:
                    messagebox.showerror(
                        "Ошибка ввода",
                        f"Поле «{label}» должно быть числом.\n"
                        f"Текущее значение: '{text}'"
                    )
                    return None, False

                return text, True

            return format_number(value), True

        length_mm, ok = optional_float(self.length_var, "Длина, мм")
        if not ok:
            return False

        width_mm, ok = optional_float(self.width_var, "Ширина, мм")
        if not ok:
            return False

        thickness_mm, ok = optional_float(self.thickness_var, "Толщина, мм")
        if not ok:
            return False

        mass_g, ok = optional_float(self.mass_var, "Масса, г")
        if not ok:
            return False

        shape = self.shape_var.get().strip()
        status = self.status_var.get().strip()

        record["length_mm"] = length_mm
        record["width_mm"] = width_mm
        record["thickness_mm"] = thickness_mm
        record["mass_g"] = mass_g
        record["shape"] = shape
        record["status"] = status

        # Если ввели новое значение формы/статуса, добавляем его в списки.
        if shape and shape not in self.shape_options:
            self.shape_options.append(shape)
            self.shape_options.sort()
            self.shape_box.configure(values=self.shape_options)

        if status and status not in self.status_options:
            self.status_options.append(status)
            self.status_options.sort()
            self.status_box.configure(values=self.status_options)

        if was_dirty:
            self.dirty = False
            self.unsaved = True

        return True

    # =========================
    # Сохранение CSV
    # =========================

    def save(self):
        # Если текущая форма изменена, сначала проверяем и переносим изменения.
        if self.dirty and not self._save_current(strict=True):
            return False

        if not self.records:
            messagebox.showwarning("Сохранение", "Нет записей для сохранения.")
            return False

        if not self.output_path:
            return self.save_as()

        return self._write_csv(self.output_path)

    def save_as(self):
        if self.dirty and not self._save_current(strict=True):
            return False

        if not self.records:
            messagebox.showwarning("Сохранение", "Нет записей для сохранения.")
            return False

        initial_name = (
            os.path.basename(self.output_path)
            if self.output_path
            else os.path.basename(DEFAULT_OUTPUT_CSV)
        )

        path = filedialog.asksaveasfilename(
            title="Сохранить новый CSV",
            defaultextension=".csv",
            initialdir=OUT_DIR,
            initialfile=initial_name,
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ]
        )

        if not path:
            return False

        self.output_path = path
        return self._write_csv(path)

    def _get_fieldnames(self):
        """
        Формирует список колонок для выходного CSV.
        Сначала стандартные поля, затем существующие дополнительные.
        """
        names = []

        for name in DEFAULT_CSV_FIELDS:
            if name not in names:
                names.append(name)

        for name in self.fieldnames:
            if name in SKIP_CSV_FIELDS:
                continue

            if name not in names:
                names.append(name)

        # Если вдруг в записях появились ещё какие-то поля.
        for record in self.records:
            for key in record.keys():
                if key in SKIP_CSV_FIELDS:
                    continue

                if key not in names:
                    names.append(key)

        return names

    def _write_csv(self, path):
        try:
            fieldnames = self._get_fieldnames()

            with open(path, "w", newline="", encoding="utf-8-sig") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=fieldnames,
                    extrasaction="ignore"
                )
                writer.writeheader()

                for record in self.records:
                    row = {key: record.get(key, "") for key in fieldnames}
                    writer.writerow(row)

            self.unsaved = False
            self.dirty = False
            self._update_statusbar()

            messagebox.showinfo(
                "Сохранение",
                f"CSV сохранён:\n{path}"
            )

            return True

        except Exception as exc:
            messagebox.showerror(
                "Ошибка сохранения",
                f"Не удалось сохранить CSV:\n{exc}"
            )
            return False

    # =========================
    # Строка состояния и закрытие
    # =========================

    def _update_statusbar(self):
        if self.load_error:
            text = self.load_error
        elif not self.records:
            text = "Нет записей. Проверьте папку out и файл seeds.csv."
        else:
            record = self.records[self.current]
            image_path = self._find_image_path(record)

            if image_path:
                image_name = os.path.basename(image_path)
            else:
                file_index = record.get("file_index", "")
                index = record.get("index", "")
                image_name = file_index or index or "?"

            text = (
                f"Запись {self.current + 1}/{len(self.records)} | "
                f"file_index={record.get('file_index', '')} | "
                f"index={record.get('index', '')} | "
                f"файл={image_name}"
            )

            if self.unsaved:
                text += " | есть несохранённые изменения"

            if self.output_path:
                text += f" | последний файл: {self.output_path}"

        self.statusbar.config(text=text)

    def _on_close(self):
        if self.contour_dirty:
            self._save_contour(show_message=False)

        if self.unsaved:
            answer = messagebox.askyesnocancel(
                "Выход",
                "Сохранить изменения перед выходом?"
            )

            # Cancel или закрытие диалога
            if answer is None:
                return

            # Yes
            if answer:
                if not self.save():
                    return

        self.root.destroy()


# =========================
# Точка входа
# =========================

def main():
    # Улучшает чёткость на Windows с масштабированием, если поддерживается.
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    root = tk.Tk()
    SeedInspectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()