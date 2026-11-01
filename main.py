import os
import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from utils import (
	calculate_fit_scale,
	clamp_display_scale,
	compute_size_stats,
	count_seed_statuses,
	export_seed_items,
	format_calibration_message,
	format_export_message,
	format_param_label,
	format_seed_status_message,
	format_size_stats,
	get_checkbox_definitions,
	get_default_params,
	get_param_definitions,
	imread_unicode,
	imwrite_unicode,
	normalize_params,
	parse_mm_per_px_text,
	process_image,
	round_param_value,
	sanitize_filename,
	select_items_for_export,
	try_calibrate_ruler,
)

RESAMPLE = Image.Resampling.LANCZOS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "out")


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
		try:
			ctypes.windll.shcore.SetProcessDpiAwareness(2)
		except Exception:
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


def make_source_name(image_path):
	"""
	Создаёт безопасное имя группы файлов из пути изображения.

	Это функция работы с путями, поэтому оставлена в main,
	но используется и в GUI, и в пакетной обработке, чтобы не дублировать код.
	"""
	if image_path:
		return sanitize_filename(
			os.path.splitext(os.path.basename(image_path))[0]
		)
	return "image"


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

		self.current_view = "original"
		self.display_scale = 1.0
		self.fit_mode = "width"
		self._last_canvas_size = (0, 0)

		self.status_var = tk.StringVar(
			value="Откройте фото, подберите параметры и нажмите «Обработать»."
		)

		self._init_vars()
		self._build_ui()

	def _init_vars(self):
		self.vars = {}

		self.param_defs = list(get_param_definitions())
		self.checkbox_defs = list(get_checkbox_definitions())

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

		for _, key, default in self.checkbox_defs:
			self.vars[key] = tk.BooleanVar(value=default)

		self.mm_per_px_var = tk.StringVar(value="")
		self.stats_var = tk.StringVar(value="—")

	def _build_ui(self):
		self.root.title("Выделение и нарезка семян ячменя")
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

		self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)
		self.canvas.bind("<Control-Button-4>", lambda e: self._zoom_factor(1.25, e))
		self.canvas.bind("<Control-Button-5>", lambda e: self._zoom_factor(0.8, e))

		self.canvas.bind("<MouseWheel>", self._on_mousewheel)
		self.canvas.bind("<Button-4>", self._on_mousewheel)
		self.canvas.bind("<Button-5>", self._on_mousewheel)

		self.canvas.bind("<ButtonPress-1>", lambda e: self.canvas.scan_mark(e.x, e.y))
		self.canvas.bind("<B1-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))

	def _build_param_panel(self):
		pf = self.param_frame

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
		ttk.Label(pf, text="Порог: Otsu").pack(anchor="w", padx=4, pady=(4, 0))

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
		return format_param_label(
			self.param_titles[key],
			self.vars[key].get(),
			self.param_kinds[key],
			self.param_steps[key],
		)

	def _round_param(self, key, raw_value):
		return round_param_value(
			raw_value,
			self.param_lows[key],
			self.param_highs[key],
			self.param_steps[key],
			self.param_kinds[key],
		)

	def _on_scale(self, key, raw_value):
		value = self._round_param(key, raw_value)
		self.vars[key].set(value)

		if key in self.scale_value_labels:
			self.scale_value_labels[key].config(text=self._format_param(key))

	def get_params(self):
		params = {
			key: var.get()
			for key, var in self.vars.items()
		}
		params = normalize_params(params)
		params["mm_per_px"] = self._parse_mm_per_px()
		return params

	def _parse_mm_per_px(self):
		return parse_mm_per_px_text(self.mm_per_px_var.get())

	# ------------------------------------------------------------
	# View / zoom
	# ------------------------------------------------------------

	def _current_image(self):
		if self.current_view == "result":
			return self.result_bgr
		return self.original_bgr

	def _clamp_scale(self, scale, img):
		h, w = img.shape[:2]
		return clamp_display_scale(
			scale,
			w,
			h,
			self.MIN_SCALE,
			self.MAX_SCALE,
			self.MAX_DISPLAY_DIM,
		)

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

		scale = calculate_fit_scale(
			w,
			h,
			cw,
			ch,
			self.fit_mode,
			self.display_scale,
		)
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

			counts = count_seed_statuses(self.seed_items)
			self.status_var.set(format_seed_status_message(counts))

		except Exception as exc:
			messagebox.showerror("Ошибка обработки", str(exc))
			self.status_var.set("Ошибка обработки.")

	def calibrate_ruler_from_image(self):
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
		self.status_var.set(
			f"Калибровка выполнена: {format_calibration_message(mm_per_px)}."
		)

	def _set_view(self, view):
		if view == "result":
			if self.result_bgr is None:
				messagebox.showwarning("Нет результата", "Сначала выполните обработку.")
				return
		else:
			if self.original_bgr is None:
				messagebox.showwarning("Нет изображения", "Сначала откройте фото.")
				return

		self.current_view = view
		self.canvas.after_idle(self._refresh_view_after_change)

	def show_original(self):
		self._set_view("original")

	def show_result(self):
		self._set_view("result")

	def _source_name(self):
		return make_source_name(self.image_path)

	def save_crops(self):
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
			messagebox.showwarning(
				"Нет данных",
				"Нет объектов для сохранения по текущим фильтрам."
			)
			return

		try:
			mm_per_px = self._parse_mm_per_px()
		except ValueError as exc:
			messagebox.showerror("Ошибка масштаба", str(exc))
			return

		source_name = self._source_name()

		try:
			summary = export_seed_items(
				self.original_bgr,
				selected,
				DEFAULT_OUT_DIR,
				source_name,
				padding=int(self.vars["padding"].get()),
				mm_per_px=mm_per_px,
			)
		except Exception as exc:
			messagebox.showerror("Ошибка", f"Не удалось сохранить фрагменты: {exc}")
			return

		if summary["csv_error"]:
			messagebox.showerror(
				"Ошибка CSV",
				f"Не удалось сохранить CSV: {summary['csv_error']}"
			)
			return

		self.status_var.set(
			format_export_message(
				summary["saved"],
				summary["selected"],
				source_name,
				DEFAULT_OUT_DIR,
				summary["contour_errors"],
			)
		)

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


# ============================================================
# Пакетная обработка (без GUI)
# ============================================================

def collect_image_paths(args):
	"""
	Собирает список .jpg/.jpeg файлов из переданных аргументов.
	Каждый аргумент может быть файлом или папкой.

	Это функция работы с путями, поэтому оставлена в main.
	"""
	paths = []

	for arg in args:
		arg = os.path.abspath(arg)

		if os.path.isfile(arg):
			ext = os.path.splitext(arg)[1].lower()
			if ext in (".jpg", ".jpeg"):
				paths.append(arg)

		elif os.path.isdir(arg):
			for name in sorted(os.listdir(arg)):
				ext = os.path.splitext(name)[1].lower()
				if ext in (".jpg", ".jpeg"):
					paths.append(os.path.join(arg, name))

		else:
			print(f"[ПРЕДУПРЕЖДЕНИЕ] Путь не найден, пропущен: {arg}")

	return paths


def batch_process_file(image_path, params, out_dir):
	"""
	Обрабатывает один файл в пакетном режиме:
	1. Читает изображение.
	2. Калибрует линейку (автоопределение мм/px).
	3. Сегментирует и замеряет семена.
	4. Сохраняет кропы, контуры и обновляет CSV.
	"""
	img = imread_unicode(image_path)

	if img is None:
		print(f"[ОШИБКА] Не удалось прочитать: {image_path}")
		return

	source_name = make_source_name(image_path)
	h, w = img.shape[:2]

	print(f"  Обработка: {os.path.basename(image_path)} ({w}x{h})")

	mm_per_px, calibration_error = try_calibrate_ruler(img)

	if mm_per_px is not None:
		print(f"    Калибровка: {format_calibration_message(mm_per_px)}")
	else:
		print(f"    [ПРЕДУПРЕЖДЕНИЕ] Калибровка не удалась: {calibration_error}")
		print("    Продолжаю без масштаба (замеры в px).")

	params = dict(params)
	params["mm_per_px"] = mm_per_px
	params = normalize_params(params)

	try:
		result = process_image(img, params)
	except Exception as exc:
		print(f"    [ОШИБКА] Обработка не удалась: {exc}")
		return

	items = result["items"]
	print(f"    Найдено объектов: {len(items)}")

	selected = select_items_for_export(
		items,
		save_only_focused=params["save_only_focused"],
		skip_blurry=False,
	)

	if not selected:
		print("    Нет объектов для сохранения после фильтрации.")
		return

	try:
		summary = export_seed_items(
			img,
			selected,
			out_dir,
			source_name,
			padding=int(params["padding"]),
			mm_per_px=mm_per_px,
		)
	except Exception as exc:
		print(f"    [ОШИБКА] Не удалось сохранить фрагменты: {exc}")
		return

	if summary["csv_error"]:
		print(f"    [ОШИБКА] Не удалось обновить CSV: {summary['csv_error']}")

	counts = count_seed_statuses(items)

	print(
		f"    Сохранено: {summary['saved']} шт. | "
		f"фокус: {counts['focused']}, размыто: {counts['blurry']}, "
		f"всего найдено: {counts['total']}"
	)

	if summary["contour_errors"]:
		print(f"    Ошибки контуров: {summary['contour_errors']}")


def batch_main(paths):
	"""
	Точка входа для пакетной обработки.
	Принимает список путей (файлы/папки), собирает .jpg и обрабатывает.
	"""
	image_files = collect_image_paths(paths)

	if not image_files:
		print("Не найдено ни одного .jpg файла по указанным путям.")
		sys.exit(1)

	print(f"Найдено файлов для обработки: {len(image_files)}")
	print(f"Папка результатов: {DEFAULT_OUT_DIR}")
	print("-" * 60)

	os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)

	params = get_default_params()

	for i, image_path in enumerate(image_files, 1):
		print(f"[{i}/{len(image_files)}]")
		batch_process_file(image_path, params, DEFAULT_OUT_DIR)

	print("-" * 60)
	print(f"Пакетная обработка завершена. Результат в: {DEFAULT_OUT_DIR}")


# ============================================================
# Точка входа
# ============================================================

def main():
	setup_dpi_awareness()

	# Если переданы аргументы командной строки — пакетный режим
	if len(sys.argv) > 1:
		batch_main(sys.argv[1:])
		return

	# Иначе — обычный GUI
	root = tk.Tk()

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