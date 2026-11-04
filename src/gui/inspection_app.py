from __future__ import annotations

import json
import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageTk

from core.contour_io import parse_raw_points
from core.csv_utils import merge_fieldnames, read_csv_robust, write_csv
import re
from core.io_utils import fmt_float, parse_float
from core.models import AUG_EXTRA_FIELDS, DEFAULT_CSV_FIELDS, EXTRA_EDITABLE_FIELDS, IMAGE_EXTENSIONS, SKIP_CSV_FIELDS, is_augmentation
from core.seed_geometry import compute_median_color_hex
from gui.common.file_scan import scan_directory
from gui.common.ruler import draw_canvas_ruler
from gui.common.widgets import (
    инициализировать_состояние_приложения,
    создать_блок_навигации,
    создать_верхнюю_панель,
    создать_панель_масштаба,
    создать_панель_модулей,
    создать_статус_строку,
    выбрать_папку,
)
from gui.common.zoomable import ZoomableCanvas, дельта_прокрутки_колесом, load_image_async, RESAMPLE

POINT_HANDLE_RADIUS = 5
POINT_HIT_RADIUS = 9

BASE_COLUMNS = DEFAULT_CSV_FIELDS
ALL_COLUMNS = BASE_COLUMNS + EXTRA_EDITABLE_FIELDS

# --- Пакетный режим разметки has_groove -----------------------------------
BATCH_COLS = 13                       # столбцов в сетке
BATCH_ROWS = 4                        # строк в сетке
BATCH_SIZE = BATCH_COLS * BATCH_ROWS  # изображений на одну страницу (32)
BATCH_GAP = 12                        # отступ между ячейками, px
BATCH_PLACEHOLDER_SIZE = (240, 160)   # размер ячейки, если файл не найден
BATCH_BORDER = "#5A5A5A"              # рамка неизменённой ячейки
BATCH_BORDER_MODIFIED = "#FF8C00"     # рамка ячейки с изменённым значением

# Варианты выпадающего списка режимов просмотра.
MODE_NORMAL = "Обычный режим"     # одиночный вид с разметкой
MODE_WITH_GROOVE = "with groove"  # пакетный режим (сетка 8×4)
MODE_WITHOUT_GROOVE = "without groove"  # пакетный режим (сетка 8×4)
MODE_OPTIONS = (MODE_NORMAL, MODE_WITH_GROOVE, MODE_WITHOUT_GROOVE)


def format_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.10g}"


class InspectionApp:
    def __init__(
        self,
        root: tk.Tk,
        data_dir: str,
        switch_callback: Callable[..., None] | None = None,
        initial_index: int = 0,
        initial_file_index: str | None = None,
    ) -> None:
        инициализировать_состояние_приложения(
            self, root, data_dir, switch_callback, initial_index, initial_file_index
        )
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
        # Защита от устаревших асинхронных результатов (см. _load_current_media).
        self._image_gen = 0
        self._last_render_size: tuple[int, int] = (0, 0)

        # --- Состояние пакетного режима разметки has_groove ---
        self.batch_mode = False                       # включён ли пакетный режим
        self.batch_label = ""                         # выбранный вариант режима (для статус-строки)
        self.batch_page = 0                           # номер текущей пачки (с нуля)
        self._batch_gen = 0                           # поколение отложенных загрузок (защита от устаревших ответов)
        self._batch_rects: list[tuple[int, int, int, int, int]] = []  # (x0, y0, x1, y1, индекс записи)
        self._batch_photos: dict[int, ImageTk.PhotoImage] = {}        # ссылки на PhotoImage ячеек
        self._batch_check_vars: dict[int, tk.BooleanVar] = {}         # переменные чекбоксов ячеек
        self._batch_original: dict[int, str] = {}     # индекс записи -> значение has_groove до правок
        self._batch_size_cache: dict[str, tuple[int, int]] = {}       # кэш размеров файлов изображений
        self._unsaved_before_batch = False            # были ли правки таблицы до входа в режим

        self.image_files: dict[str, str] = {}
        self.json_files: dict[str, str] = {}
        self._build_ui()
        self._bind_keys()
        # Сначала создаём окно, а разбор CSV и сканирование папки выполняем
        # в простое: раньше приложение открывалось медленно, потому что вся
        # обработка шла до отрисовки первого кадра.
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
        # Данные перечитаются с диска: прежние индексы пакетных правок и
        # кэш размеров изображений становятся недостоверными.
        self._batch_original.clear()
        self._batch_size_cache.clear()
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
            if self._is_aug_row(fi, clean):
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

    @staticmethod
    def _is_aug_row(file_index: str, row: dict[str, str]) -> bool:
        return (
            is_augmentation(row)
            or bool(str(row.get("aug_of") or "").strip())
            or bool(re.search(r"_aug\d+$", str(file_index or "").strip()))
        )

    def _build_ui(self) -> None:
        self.root.title("Контроль параметров семечек")
        self.root.geometry("1360x840")
        self.root.minsize(1050, 650)

        self.statusbar = создать_статус_строку(self.root, текст="Загрузка...", отступ=(8, 3))

        top = создать_верхнюю_панель(self.root)

        # 1. Единый блок переключения модулей
        создать_панель_модулей(top, "inspection", self._nav)

        ttk.Separator(top, orient="vertical").pack(side=tk.LEFT, fill="y", padx=8)

        # 2. Выбор папки
        ttk.Button(top, text="📁 Папка...", command=self._choose_folder).pack(side=tk.LEFT, padx=2)

        ttk.Separator(top, orient="vertical").pack(side=tk.LEFT, fill="y", padx=6)

        # 2.1 Выбор режима просмотра: одиночный вид или пакетная разметка
        ttk.Label(top, text="Режим:").pack(side=tk.LEFT, padx=(0, 2))
        self.mode_var = tk.StringVar(value=MODE_NORMAL)
        self.mode_box = ttk.Combobox(
            top, textvariable=self.mode_var, values=list(MODE_OPTIONS),
            state="readonly", width=17,
        )
        self.mode_box.pack(side=tk.LEFT, padx=2)
        self.mode_box.bind("<<ComboboxSelected>>", self._on_mode_change)

        ttk.Separator(top, orient="vertical").pack(side=tk.LEFT, fill="y", padx=6)

        # 3. Навигация по семенам (единый стиль)
        (self.btn_prev, self.pos_lbl, self.pos_var, self.pos_entry,
         self.total_lbl, self.btn_next) = создать_блок_навигации(
            top, self.prev_seed, self.next_seed, self._on_goto
        )

        ttk.Separator(top, orient="vertical").pack(side=tk.LEFT, fill="y", padx=6)

        # 4. Единый блок масштабирования
        _, self.zoom_lbl, zoom_btns = создать_панель_масштаба(top, self)
        self.btn_zoom_in = zoom_btns["zoom_in"]
        self.btn_zoom_out = zoom_btns["zoom_out"]
        self.btn_zoom_actual = zoom_btns["zoom_actual"]
        self.btn_zoom_fit = zoom_btns["zoom_fit"]

        # 5. Кнопки сохранения справа
        self.btn_save_contour = ttk.Button(top, text="Сохранить контур", command=lambda: self._save_contour(show_message=True))
        self.btn_save_contour.pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(top, text="Сохранить как...", command=self.save_as).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(top, text="Сохранить", command=self.save).pack(side=tk.RIGHT, padx=(4, 0))

        main = self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill='both', expand=True, padx=8, pady=(0, 8))

        img_frame = ttk.LabelFrame(main, text="Изображение", padding=6)
        main.add(img_frame, weight=3)
        self.zoom_widget = ZoomableCanvas(img_frame)
        self.zoom_widget.set_render_callback(self._draw_contour_overlay)
        self.zoom_widget.set_zoom_callback(lambda pct: self.zoom_lbl.config(text=f"{pct}%"))
        # Заменяем (а не дополняем) обработчики перетаскивания, которые
        # устанавливает ZoomableCanvas в конструкторе: пока захвачена точка
        # контура, наши обработчики возвращают "break", поэтому ЛКМ двигает
        # только точку, но не изображение; если точка не захвачена, управление
        # переходит к scan_mark/scan_dragto, и картинку можно просто двигать.
        self.zoom_widget.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.zoom_widget.canvas.bind("<B1-Motion>", self._on_left_motion)
        self.zoom_widget.canvas.bind("<ButtonRelease-1>", self._on_left_release)

        def _pan_mark(event: tk.Event) -> None:
            """Начать перетаскивание, только если не захвачена точка контура."""
            if self.drag_point_index is None:
                self.zoom_widget.canvas.scan_mark(event.x, event.y)

        def _pan_dragto(event: tk.Event) -> None:
            if self.drag_point_index is None:
                self.zoom_widget.canvas.scan_dragto(event.x, event.y, gain=1)

        # add="+" ставит наши обработчики поверх прежних; они возвращают
        # "break", пока тащится точка, и эти обработчики тогда не срабатывают.
        self.zoom_widget.canvas.bind("<ButtonPress-1>", _pan_mark, add="+")
        self.zoom_widget.canvas.bind("<B1-Motion>", _pan_dragto, add="+")
        self.zoom_widget.canvas.bind("<Double-Button-1>", self._on_double_click)
        self.zoom_widget.canvas.bind("<Button-3>", self._on_right_click)

        params_frame = ttk.LabelFrame(main, text="Параметры", padding=10)
        main.add(params_frame, weight=1)
        self._build_form(params_frame)

        # --- Панель пакетного режима (по умолчанию скрыта) ---------------
        # Сетка 8×4 на холсте с полосами прокрутки: изображения выводятся в
        # исходном пиксельном размере, без масштабирования и без разметки.
        self.batch_frame = ttk.LabelFrame(
            self.root,
            text="Пакетная разметка has_groove — сетка 8×4, исходный размер изображений",
            padding=6,
        )
        bwrap = ttk.Frame(self.batch_frame)
        bwrap.pack(fill=tk.BOTH, expand=True)
        bwrap.columnconfigure(0, weight=1)
        bwrap.rowconfigure(0, weight=1)
        self.batch_canvas = tk.Canvas(bwrap, bg="#202020", highlightthickness=0)
        self.batch_canvas.grid(row=0, column=0, sticky="nsew")
        bvbar = ttk.Scrollbar(bwrap, orient="vertical", command=self.batch_canvas.yview)
        bvbar.grid(row=0, column=1, sticky="ns")
        bhbar = ttk.Scrollbar(bwrap, orient="horizontal", command=self.batch_canvas.xview)
        bhbar.grid(row=1, column=0, sticky="ew")
        self.batch_canvas.configure(yscrollcommand=bvbar.set, xscrollcommand=bhbar.set)
        # Клик по самому изображению перекляетает has_groove (чекбокс в углу
        # ячейки обрабатывается своим событием, на холст оно не попадает).
        self.batch_canvas.bind("<Button-1>", self._on_batch_click)
        self.batch_canvas.bind("<MouseWheel>", self._on_batch_wheel)
        self.batch_canvas.bind("<Button-4>", lambda e: self.batch_canvas.yview_scroll(-1, "units"))
        self.batch_canvas.bind("<Button-5>", lambda e: self.batch_canvas.yview_scroll(1, "units"))

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
        d = выбрать_папку(self.data_dir, "Выберите папку с семенами (out)")
        if d:
            self.data_dir = os.path.abspath(d)
            self._load_data()
            self.current = 0
            self._update_all()

    def _build_form(self, parent: ttk.Frame) -> None:
        form = ttk.Frame(parent)
        form.pack(fill=tk.BOTH, expand=True)

        def add_field(text: str, r: int, readonly: bool = False) -> tk.StringVar:
            ttk.Label(form, text=text).grid(row=r, column=0, sticky="w", padx=(0, 10), pady=3)
            var = tk.StringVar()
            entry = ttk.Entry(form, textvariable=var, width=20, **({"state": "readonly"} if readonly else {}))
            entry.grid(row=r, column=1, sticky="ew", pady=3)
            if not readonly:
                entry.bind("<KeyRelease>", self._mark_dirty_and_render)
                entry.bind("<FocusOut>", self._mark_dirty_and_render)
            return var

        def add_readonly(text: str, r: int) -> tk.StringVar:
            """Создать только для чтения поле формы (см. ``add_field``)."""
            return add_field(text, r, readonly=True)

        def add_entry(text: str, r: int) -> tk.StringVar:
            """Создать редактируемое поле формы (см. ``add_field``)."""
            return add_field(text, r)

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
        if self.batch_mode:
            # Пакетное листание: шаг равен одной пачке (32 записи).
            new_page = self.batch_page + delta
            if 0 <= new_page < self._batch_pages():
                self.batch_page = new_page
                self._render_batch_page()
                self._update_nav()
                self._update_statusbar()
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
        if self.batch_mode:
            # Поле ввода в пакетном режиме задаёт номер пачки (а не семечки).
            try:
                page = int(self.pos_var.get().strip())
            except ValueError:
                messagebox.showerror("Ошибка перехода", "Введите номер пачки числом.")
                self.pos_var.set(str(self.batch_page + 1))
                return
            pages = self._batch_pages()
            if 1 <= page <= pages:
                self.batch_page = page - 1
                self._render_batch_page()
                self._update_nav()
                self._update_statusbar()
            else:
                messagebox.showerror("Ошибка перехода", f"Допустимый диапазон: от 1 до {pages}.")
                self.pos_var.set(str(self.batch_page + 1))
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
        if self.batch_mode:
            # Одиночный вид скрыт: обновляем только пакетную сетку и статус.
            self._render_batch_page()
            self._update_nav()
            self._update_statusbar()
            return
        self._update_nav()
        self._update_form()
        self._load_current_media()
        self._update_statusbar()

    def _update_nav(self) -> None:
        if self.batch_mode:
            # В пакетном режиме кнопки листают пачки по 32 записи; на границах
            # списка (начало/конец) они блокируются — без зацикливания.
            pages = self._batch_pages()
            self.total_lbl.config(text=f"пачек: {pages}")
            self.pos_var.set(str(self.batch_page + 1 if pages else 0))
            self.btn_prev.config(state="normal" if self.batch_page > 0 else "disabled")
            self.btn_next.config(state="normal" if self.batch_page < pages - 1 else "disabled")
            return
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

    # ------------------------------------------------------------------
    # Пакетный режим разметки has_groove (сетка 8×4, исходный размер)
    # ------------------------------------------------------------------

    def _batch_pages(self) -> int:
        """Количество пачек по 32 записи (0, если записей нет)."""
        if not self.records:
            return 0
        return -(-len(self.records) // BATCH_SIZE)

    @staticmethod
    def _groove_bool(record: dict[str, str]) -> bool:
        """Значение has_groove записи в виде True/False."""
        return str(record.get("has_groove", "")).strip() in ("1", "True", "true")

    def _batch_is_modified(self, gi: int) -> bool:
        """Отличается ли текущее значение has_groove от исходного."""
        original = self._batch_original.get(gi)
        if original is None or not (0 <= gi < len(self.records)):
            return False
        return str(self.records[gi].get("has_groove", "")).strip() != original

    def _on_mode_change(self, event: tk.Event | None = None) -> None:
        mode = self.mode_var.get()
        if mode in (MODE_WITH_GROOVE, MODE_WITHOUT_GROOVE):
            self._enter_batch_mode(mode)
        else:
            self._exit_batch_mode()

    def _set_single_controls_enabled(self, enabled: bool) -> None:
        """Включить/выключить элементы одиночного вида (зум и контур)."""
        state = "normal" if enabled else "disabled"
        for widget in (self.btn_zoom_in, self.btn_zoom_out, self.btn_zoom_actual,
                       self.btn_zoom_fit, self.btn_save_contour):
            try:
                widget.config(state=state)
            except Exception:
                pass

    def _enter_batch_mode(self, label: str) -> None:
        """Переключить главный вид в пакетный режим разметки has_groove."""
        if self.batch_mode:
            # Уже в пакетном режиме: меняется только название варианта.
            self.batch_label = label
            self._update_statusbar()
            return
        # Правки формы одиночного вида переносим в память, чтобы не потерять.
        if self.dirty and not self._save_current(strict=True):
            self.mode_var.set(MODE_NORMAL)
            return
        if self.unsaved:
            # Сохраняем накопленные правки таблицы заранее: кнопка «Сохранить»
            # в пакетном режиме пишет только столбец has_groove.
            ans = messagebox.askyesno(
                "Сохранение",
                "Есть несохранённые изменения таблицы.\n"
                "Сохранить их в CSV перед переключением в пакетный режим?",
            )
            if ans and not self._write_csv(self.output_path, show_message=False):
                self.mode_var.set(MODE_NORMAL)
                return
        if self.contour_dirty:
            ans = messagebox.askyesnocancel(
                "Несохраненный контур",
                "Контур был изменен, но не сохранен.\n"
                "Сохранить контур перед переключением в пакетный режим?",
            )
            if ans is None:
                self.mode_var.set(MODE_NORMAL)
                return
            if ans and not self._save_contour(show_message=False):
                self.mode_var.set(MODE_NORMAL)
                return
        self._unsaved_before_batch = self.unsaved
        self.batch_mode = True
        self.batch_label = label
        # Показываем пачку, в которой находится текущая одиночная запись.
        self.batch_page = (self.current // BATCH_SIZE) if self.records else 0
        self.main_paned.pack_forget()
        self.batch_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._set_single_controls_enabled(False)
        self.pos_lbl.config(text="Пачка:")
        self._render_batch_page()
        self._update_nav()
        self._update_statusbar()

    def _exit_batch_mode(self) -> None:
        """Вернуть одиночный вид с разметкой изображения."""
        if not self.batch_mode:
            return
        self.batch_mode = False
        self.batch_label = ""
        self._batch_gen += 1  # отменяем отложенные загрузки картинок пачки
        # Освобождаем память: картинки и чекбоксы скрытой сетки не нужны.
        self.batch_canvas.delete("all")
        self._batch_photos.clear()
        self._batch_check_vars.clear()
        self._batch_rects.clear()
        self.batch_frame.pack_forget()
        self.main_paned.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._set_single_controls_enabled(True)
        self.pos_lbl.config(text="Семечка:")
        self._update_all()

    def _peek_image_size(self, path: str | None) -> tuple[int, int] | None:
        """Размер изображения в пикселях (читаем только заголовок файла)."""
        if not path:
            return None
        cached = self._batch_size_cache.get(path)
        if cached is not None:
            return cached
        try:
            with Image.open(path) as im:
                size = im.size
        except Exception:
            return None
        if size[0] > 0 and size[1] > 0:
            self._batch_size_cache[path] = size
            return size
        return None

    @staticmethod
    def _batch_placeholder(w: int, h: int) -> Image.Image:
        """Заглушка для записи, у которой не нашлось изображения."""
        w, h = max(40, w), max(40, h)
        img = Image.new("RGB", (w, h), (245, 245, 245))
        draw = ImageDraw.Draw(img)
        draw.text((8, h // 2 - 6), "изображение не найдено", fill=(140, 0, 0))
        return img

    def _render_batch_page(self) -> None:
        canvas = self.batch_canvas
        canvas.delete("all")
        self._batch_photos.clear()
        self._batch_check_vars.clear()
        self._batch_rects.clear()
        self._batch_gen += 1
        gen = self._batch_gen

        # Номер пачки приводим к допустимому диапазону.
        pages = self._batch_pages()
        self.batch_page = min(max(0, self.batch_page), pages - 1) if pages else 0
        if not self.records:
            canvas.configure(scrollregion=(0, 0, 1, 1))
            return

        start = self.batch_page * BATCH_SIZE
        chunk = self.records[start:start + BATCH_SIZE]
        count = len(chunk)
        cols = min(BATCH_COLS, count)
        rows = (count + cols - 1) // cols

        # Размеры ячеек: заголовки файлов читаются без декодирования пикселей.
        paths: list[str | None] = []
        sizes: list[tuple[int, int]] = []
        for rec in chunk:
            path = self._find_image_path(rec)
            paths.append(path)
            sizes.append(self._peek_image_size(path) or BATCH_PLACEHOLDER_SIZE)

        # Ширина столбца и высота строки — максимум по ячейкам, чтобы сетка
        # оставалась ровной при слегка разном размере кадров.
        col_w = [0] * cols
        row_h = [0] * rows
        for i, (w, h) in enumerate(sizes):
            r, c = divmod(i, cols)
            col_w[c] = max(col_w[c], w)
            row_h[r] = max(row_h[r], h)
        x_pos: list[int] = []
        acc = 0
        for w in col_w:
            x_pos.append(acc)
            acc += w + BATCH_GAP
        y_pos: list[int] = []
        acc = 0
        for h in row_h:
            y_pos.append(acc)
            acc += h + BATCH_GAP
        total_w = (x_pos[-1] + col_w[-1]) if cols else 1
        total_h = (y_pos[-1] + row_h[-1]) if rows else 1
        canvas.configure(scrollregion=(0, 0, total_w, total_h))
        canvas.xview_moveto(0.0)
        canvas.yview_moveto(0.0)

        for i, rec in enumerate(chunk):
            r, c = divmod(i, cols)
            x, y = x_pos[c], y_pos[r]
            w, h = sizes[i]
            gi = start + i
            # Рамка ячейки (подсвечивается оранжевым после изменения значения).
            modified = self._batch_is_modified(gi)
            canvas.create_rectangle(
                x - 1, y - 1, x + w + 1, y + h + 1,
                outline=BATCH_BORDER_MODIFIED if modified else BATCH_BORDER,
                width=2 if modified else 1,
                tags=("cell_border", f"border:{gi}"),
            )
            # Чекбокс в левом верхнем углу изображения.
            var = tk.BooleanVar(value=self._groove_bool(rec))
            cb = ttk.Checkbutton(canvas, variable=var, command=lambda g=gi: self._on_batch_check(g))
            canvas.create_window(x, y, anchor="nw", window=cb, tags=("cell_cb",))
            self._batch_check_vars[gi] = var
            self._batch_rects.append((x, y, x + w, y + h, gi))
            # Изображение: только исходные пиксели, без масштабирования.
            if paths[i]:
                load_image_async(paths[i], self._make_batch_loader(gen, x, y, gi), canvas)
            else:
                self._place_batch_image(gen, x, y, gi, None)

    def _make_batch_loader(self, gen: int, x: int, y: int, gi: int):
        """Замыкание-callback для асинхронной загрузки картинки ячейки."""

        def _on_img(img: Image.Image | None) -> None:
            self._place_batch_image(gen, x, y, gi, img)

        return _on_img

    def _place_batch_image(self, gen: int, x: int, y: int, gi: int, img: Image.Image | None) -> None:
        """Поставить картинку ячейки на холст в её исходном размере."""
        if gen != self._batch_gen or not self.batch_mode:
            return  # страница уже сменилась — результат устарел
        try:
            if not self.batch_canvas.winfo_exists():
                return
        except Exception:
            return
        if img is None:
            img = self._batch_placeholder(*BATCH_PLACEHOLDER_SIZE)
        # PhotoImage создаётся из исходного изображения без каких-либо
        # преобразований: на экране ровно столько пикселей, сколько в файле.
        photo = ImageTk.PhotoImage(img)
        self._batch_photos[gi] = photo
        self.batch_canvas.create_image(x, y, anchor="nw", image=photo, tags=("cell_img", f"img:{gi}"))
        # Рамка и чекбокс всегда поверх картинки.
        self.batch_canvas.tag_raise("cell_border")
        self.batch_canvas.tag_raise("cell_cb")

    def _on_batch_wheel(self, event: tk.Event) -> None:
        """Прокрутка сетки колесом мыши (с Shift — по горизонтали)."""
        delta = дельта_прокрутки_колесом(event)
        if delta == 0:
            return
        if getattr(event, "state", 0) & 0x0001:
            self.batch_canvas.xview_scroll(delta, "units")
        else:
            self.batch_canvas.yview_scroll(delta, "units")

    def _on_batch_click(self, event: tk.Event) -> None:
        """Клик по изображению переключает has_groove этой ячейки."""
        if not self.batch_mode:
            return
        cx = self.batch_canvas.canvasx(event.x)
        cy = self.batch_canvas.canvasy(event.y)
        for x0, y0, x1, y1, gi in self._batch_rects:
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                self._set_batch_groove(gi, "0" if self._groove_bool(self.records[gi]) else "1")
                return

    def _on_batch_check(self, gi: int) -> None:
        """Клик по чекбоксу: Tk уже переключил его переменную."""
        var = self._batch_check_vars.get(gi)
        if var is None or not (0 <= gi < len(self.records)):
            return
        self._set_batch_groove(gi, "1" if bool(var.get()) else "0")

    def _set_batch_groove(self, gi: int, raw_value: str) -> None:
        """Изменить has_groove записи в памяти и обновить только её ячейку."""
        if not (0 <= gi < len(self.records)):
            return
        record = self.records[gi]
        if gi not in self._batch_original:
            self._batch_original[gi] = str(record.get("has_groove", "")).strip()
        record["has_groove"] = raw_value
        # Синхронизируем чекбокс (программная установка команду не вызывает).
        expected = raw_value in ("1", "True", "true")
        var = self._batch_check_vars.get(gi)
        if var is not None and bool(var.get()) != expected:
            var.set(expected)
        self._update_batch_border(gi)
        self.unsaved = True
        self._update_statusbar()

    def _update_batch_border(self, gi: int) -> None:
        """Обновить рамку одной ячейки, не перерисовывая сетку."""
        items = self.batch_canvas.find_withtag(f"border:{gi}")
        if not items:
            return
        modified = self._batch_is_modified(gi)
        self.batch_canvas.itemconfig(
            items[0],
            outline=BATCH_BORDER_MODIFIED if modified else BATCH_BORDER,
            width=2 if modified else 1,
        )

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
        """Вычислить медианный цвет по текущему изображению и точкам контура."""
        src_img = getattr(self.zoom_widget, "_source_image", None) or self.zoom_widget.image
        if src_img is None or not self.contour_points or len(self.contour_points) < 3:
            return ""
        return compute_median_color_hex(src_img, self.contour_points)

    def _on_contour_modified(self) -> None:
        """Обновлять живой показ цвета при изменении контура."""
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

        # 1. Сначала загружаем данные контура, чтобы self.contour_points был
        # готов до первого вызова отрисовки (и при синхронном, и при
        # асинхронном способе загрузки).
        file_index = str(record.get("file_index", "")).strip()
        if not file_index and image_path:
            file_index = os.path.splitext(os.path.basename(image_path))[0]
        file_index = self._base_name(file_index)
        loaded = False
        if file_index:
            # scan_directory хранит JSON-файлы по базовому имени без расширения.
            actual = self.json_files.get(f"{file_index}_contour".lower())
            if actual:
                loaded = self._load_contour_file(os.path.join(self.data_dir, actual))
            if not loaded:
                # Возможно, файла ещё не было в момент сканирования (например, он только что создан).
                self.contour_path = os.path.join(self.data_dir, f"{file_index}_contour.json")
                if not os.path.isfile(self.contour_path):
                    self.contour_path = None
        # Если точного файла "<file_index>_contour.json" не нашлось, ищем
        # контур, соответствующий изображению, найденному на диске.
        if self.contour_path is None and not loaded and image_path:
            img_base = self._base_name(image_path)
            actual = self.json_files.get(f"{img_base}_contour".lower())
            if actual:
                loaded = self._load_contour_file(os.path.join(self.data_dir, actual))

        # 2. Декодируем и устанавливаем изображение.
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

        # 3. Явно запускаем отрисовку наложения (для синхронной загрузки из кэша или мгновенного показа)
        self._draw_contour_overlay()
        src_img = getattr(self.zoom_widget, "_source_image", None) or self.zoom_widget.image
        if not record.get("color_hex") and self.contour_points and src_img is not None:
            c_hex = self._recompute_current_color()
            if c_hex:
                record["color_hex"] = c_hex
                self.color_hex_var.set(c_hex)
                self._update_color_badge(c_hex)

    def _load_contour_file(self, path: str) -> bool:
        """Прочитать точки контура из *path*; True — успех."""
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
        """Вернуть *name* без каталога и без известного расширения."""
        base = os.path.basename(str(name).strip())
        lowered = base.lower()
        if lowered.endswith(IMAGE_EXTENSIONS) or lowered.endswith(".json"):
            base = os.path.splitext(base)[0]
        return base

    def _image_by_base(self, base: str) -> str | None:
        """Найти изображение по базовому имени (без учёта регистра, любое расширение)."""
        if not base:
            return None
        key = base.lower()
        # Быстрый путь: точное совпадение базового имени из скана папки
        # (scan_directory хранит изображения по имени без расширения).
        actual = self.image_files.get(key)
        if actual:
            return os.path.join(self.data_dir, actual)
        # Запасной путь: терпим лишние пробелы и частичные имена из CSV.
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
        # Последний вариант: схема имени "<file_index>_<номер семечки>",
        # которую использует экспортёр (например file_index="1", index="3" -> "1_3.jpg").
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
            cw, ch = self.zoom_widget.canvas.winfo_width(), self.zoom_widget.canvas.winfo_height()
            if (cw, ch) != self._last_render_size:
                # Геометрия изменилась: пересобираем растровое изображение под новый масштаб.
                self._last_render_size = (cw, ch)
                self.zoom_widget.render()
            self._draw_contour_overlay()
        self.zoom_lbl.config(text=f"{self.zoom_widget.get_zoom_percent()}%")

    def _draw_contour_overlay(self) -> None:
        canvas = self.zoom_widget.canvas
        # Сначала убираем элементы предыдущего кадра: иначе каждое движение
        # мыши во время перетаскивания оставляло бы старые точки и линии на
        # холсте, и виджет быстро замирал бы под тысячами накопленных элементов.
        canvas.delete("contour")
        self.contour_line_items = []
        self.contour_point_items = []
        if self.zoom_widget.display_scale <= 0:
            return
        scale = self.zoom_widget.display_scale
        if len(self.contour_points) >= 2:
            # Соединяем каждую точку с двумя соседями по ломаной и замыкаем
            # контур на первую точку (отрезок между последней и первой тоже рисуем).
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

        # Рисуем векторную линейку
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
        # Ищем по своему списку точек, а не через find_overlapping(): элемент
        # базового изображения покрывает весь холст и попадал бы под каждый
        # клик (а устаревшие элементы путают определение индекса).
        if not self.contour_points or self.zoom_widget.display_scale <= 0:
            return None
        scale = self.zoom_widget.display_scale
        best_index: int | None = None
        best_dist = float(POINT_HIT_RADIUS) + 1.0
        for i, (x, y) in enumerate(self.contour_points):
            dist = math.hypot(x * scale - cx, y * scale - cy)
            if dist <= POINT_HIT_RADIUS and dist < best_dist:
                best_dist, best_index = dist, i
        del canvas  # элементы холста для поиска не нужны
        return best_index

    def _on_left_press(self, event: tk.Event) -> None:
        idx = self._find_point_at(event)
        if idx is not None and 0 <= idx < len(self.contour_points):
            # Захватываем точку и начинаем её тащить. Возврат "break" мешает
            # обработчикам перетаскивания (scan_mark/scan_dragto) двигать
            # всю картинку: во время перетаскивания двигается только точка.
            self.drag_point_index = idx
            self._update_point_handles()
            return "break"
        # Под курсором точки нет — пусть картинку двигают панорамные
        # обработчики _pan_mark/_pan_dragto (подключены через add="+");
        # этот обработчик здесь ничего не делает.
        return None

    def _update_point_handles(self) -> None:
        """Передвигает только перетаскиваемую ручку и линии контура (быстрый путь)."""
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
        # Замыкающий отрезок (последняя -> первая точка) синхронизируем с
        # перетаскиваемой точкой — так же, как его строит _draw_contour_overlay.
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
            # Никогда не даём панорамному обработчику двигать картинку во время перетаскивания точки.
            return "break"

    def _on_left_release(self, event: tk.Event) -> None:
        if self.drag_point_index is not None:
            self.drag_point_index = None
            self._on_contour_modified()
            return "break"

    @staticmethod
    def _dist_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
        """Расстояние от точки P до отрезка AB (2D)."""
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        len2 = vx * vx + vy * vy
        if len2 <= 0.0:
            return math.hypot(wx, wy)
        t = max(0.0, min(1.0, (wx * vx + wy * vy) / len2))
        return math.hypot(wx - t * vx, wy - t * vy)

    def _insertion_index(self, pt: tuple[float, float]) -> int:
        pts = self.contour_points
        n = len(pts)
        if n < 2:
            return n
        px, py = pt
        best_seg, best_d = 0, float("inf")
        for i in range(n):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % n]  # замыкающий отрезок: последняя -> первая
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
            # JSON ещё нет: создаём его рядом с изображением, чтобы новому
            # контуру было куда сохраняться.
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
        # Вставляем точку после её двух ближайших соседей (концов ближайшего
        # существующего отрезка контура), а не в конец списка: иначе новая
        # точка соединялась бы напрямую с первой и последней точками контура.
        self.contour_points.insert(self._insertion_index(pt), pt)
        self.contour_dirty = True
        self._draw_contour_overlay()
        self._on_contour_modified()
        return "break"

    def _on_right_click(self, event: tk.Event) -> None:
        idx = self._find_point_at(event)
        if idx is not None and 0 <= idx < len(self.contour_points):
            # На самом деле удаляем точку (старый код оставлял те же элементы
            # холста с устаревшими тегами "pt:<i>", которые лишь заново
            # связывали линии между оставшимися точками).
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

            # Пересчитываем цвет по изменённому контуру и обновляем запись и интерфейс
            hex_code = self._recompute_current_color()
            if hex_code:
                record["color_hex"] = hex_code
                self.color_hex_var.set(hex_code)
                self._update_color_badge(hex_code)

            # Сохраняем обновлённый цвет в seeds.csv
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
        # Схлопываем серию быстрых нажатий в одну перерисовку наложения за цикл простоя.
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

    def _подготовить_сохранение(self) -> bool:
        """Сохранить правки перед записью CSV; ``False`` — записывать нельзя.

        Общая часть методов ``save`` и ``save_as``: сначала фиксируются
        правки полей и контура, затем проверяется наличие записей.

        Параметры: отсутствуют (используется состояние приложения).

        Возвращает ``True``, если можно продолжать запись CSV, и ``False``,
        если пользователь отменил сохранение правок либо записей нет
        (во втором случае показывается предупреждение).
        """
        if self.dirty and not self._save_current(strict=True):
            return False
        if self.contour_dirty:
            self._save_contour(show_message=False)
        if not self.records:
            messagebox.showwarning("Сохранение", "Нет записей для сохранения.")
            return False
        return True

    def save(self) -> bool:
        # В пакетном режиме «Сохранить» пишет только изменённые значения
        # has_groove — остальные столбцы и строки файла не затрагиваются.
        if self.batch_mode:
            return self._save_batch_grooves()
        if not self._подготовить_сохранение():
            return False
        if not self.output_path:
            self.output_path = os.path.join(self.data_dir, "seeds.csv")
        return self._write_csv(self.output_path)

    def _save_batch_grooves(self) -> bool:
        if not self.output_path:
            self.output_path = os.path.join(self.data_dir, "seeds.csv")
        changed = [gi for gi in self._batch_original if self._batch_is_modified(gi)]
        if not changed:
            # Правок нет — файл не трогаем.
            self.unsaved = self._unsaved_before_batch
            self._update_statusbar()
            messagebox.showinfo("Сохранение", "Изменений has_groove нет — файл не изменялся.")
            return True
        try:
            fieldnames, rows = read_csv_robust(self.output_path)
        except Exception as exc:
            messagebox.showerror("Ошибка сохранения", f"Не удалось перечитать CSV:\n{exc}")
            return False
        # Строки без аугментаций идут в файле в том же порядке, что и в
        # self.records, поэтому сопоставляем записи по позиции.
        row_positions = [
            i for i, row in enumerate(rows)
            if not self._is_aug_row(str(row.get("file_index") or "").strip(), row)
        ]
        if "has_groove" not in fieldnames:
            fieldnames.append("has_groove")
        updated = 0
        for gi in sorted(changed):
            if gi >= len(row_positions):
                break
            rows[row_positions[gi]]["has_groove"] = str(self.records[gi].get("has_groove", "")).strip()
            updated += 1
        try:
            write_csv(self.output_path, fieldnames, rows)
        except Exception as exc:
            messagebox.showerror("Ошибка сохранения", str(exc))
            return False
        # Правки зафиксированы: исходными считаем уже новые значения.
        self._batch_original.clear()
        self.unsaved = self._unsaved_before_batch
        # Убираем подсветку изменённых ячеек текущей пачки.
        start = self.batch_page * BATCH_SIZE
        for gi in range(start, min(start + BATCH_SIZE, len(self.records))):
            self._update_batch_border(gi)
        self._update_statusbar()
        messagebox.showinfo("Сохранение", f"Обновлено значений has_groove: {updated}\n{self.output_path}")
        return True

    def save_as(self) -> bool:
        if not self._подготовить_сохранение():
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
        existing = list(self.fieldnames)
        if getattr(self, "aug_records", None):
            existing = existing + AUG_EXTRA_FIELDS
        names = merge_fieldnames(existing, list(self.records), base_fields=ALL_COLUMNS)
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
        elif self.batch_mode:
            total = len(self.records)
            start = self.batch_page * BATCH_SIZE
            end = min(total, start + BATCH_SIZE)
            modified = sum(1 for gi in self._batch_original if self._batch_is_modified(gi))
            text = (
                f"Пакетный режим «{self.batch_label}» | пачка {self.batch_page + 1}/{self._batch_pages()} "
                f"| записи {start + 1}–{end} из {total} | изменено has_groove: {modified}"
            )
            if self.unsaved:
                text += " | есть несохранённые изменения"
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