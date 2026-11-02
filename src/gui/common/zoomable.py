"""Zoomable canvas widget shared by all GUI applications."""
from __future__ import annotations

import threading
import tkinter as tk
from collections import OrderedDict
from tkinter import ttk
from typing import Callable

from PIL import Image, ImageTk

# BILINEAR is ~5-10x faster than LANCZOS; for a screen preview the visual
# difference is negligible and this is what made switching seeds feel slow.
RESAMPLE = Image.Resampling.BILINEAR

MIN_ZOOM = 0.02
MAX_ZOOM = 32.0
MAX_FIT_SCALE = 12.0
FIT_PADDING = 26
MAX_DISPLAY_DIM = 12000
MAX_FALLBACK_W = 1100
MAX_FALLBACK_H = 800

# Decoded full-size images are cached (LRU) so that navigating back and
# forth between seeds never hits the disk / JPEG decoder again.
SOURCE_CACHE_MAX_BYTES = 256 * 1024 * 1024  # ~256 MB of decoded RGB images


class _SourceCache:
    """Small thread-safe LRU cache for decoded source images."""

    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max_bytes
        self._bytes = 0
        self._items: OrderedDict[str, Image.Image] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, path: str) -> Image.Image | None:
        with self._lock:
            img = self._items.get(path)
            if img is not None:
                self._items.move_to_end(path)
            return img

    def put(self, path: str, img: Image.Image) -> None:
        size = img.width * img.height * len(img.getbands())
        if size > self._max_bytes:
            return
        with self._lock:
            if path in self._items:
                self._bytes -= self._items[path].width * self._items[path].height * len(self._items[path].getbands())
                del self._items[path]
            self._items[path] = img
            self._bytes += size
            while self._bytes > self._max_bytes and len(self._items) > 1:
                _, old = self._items.popitem(last=False)
                self._bytes -= old.width * old.height * len(old.getbands())


_SOURCE_CACHE = _SourceCache(SOURCE_CACHE_MAX_BYTES)


def load_image_async(
    path: str,
    callback: Callable[[Image.Image | None], None],
    canvas: tk.Misc,
    generation: int = 0,
) -> None:
    """Decode *path* off the UI thread and hand the result to *callback*."""
    cached = _SOURCE_CACHE.get(path)
    if cached is not None:
        callback(cached.copy())
        return

    app_root = canvas.winfo_toplevel()
    result_box: list[object] = []

    def _worker() -> None:
        img: Image.Image | None = None
        try:
            with Image.open(path) as fh:
                if max(fh.width, fh.height) > 4096:
                    fh.draft("RGB", (4096, 4096))
                img = fh.convert("RGB")
                img.load()
        except Exception:
            img = None
        if img is not None:
            _SOURCE_CACHE.put(path, img)
            result_box.append(img.copy())
        else:
            result_box.append(None)

    def _check_result() -> None:
        if not app_root.winfo_exists():
            return
        if result_box:
            payload = result_box[0]
            if isinstance(payload, Image.Image):
                callback(payload)
            else:
                callback(None)
        elif t.is_alive():
            try:
                app_root.after(15, _check_result)
            except Exception:
                pass

    t = threading.Thread(target=_worker, name="image-decode", daemon=True)
    t.start()
    try:
        app_root.after(10, _check_result)
    except Exception:
        pass
    del generation


class ZoomableCanvas:
    """Encapsulates zoom/pan logic on a tk.Canvas with scrollbars."""

    def __init__(self, parent: ttk.Frame | tk.Frame, bg: str = "#202020") -> None:
        self._frame = parent
        self._frame.columnconfigure(0, weight=1)
        self._frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(parent, bg=bg, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        vbar = ttk.Scrollbar(parent, orient="vertical", command=self.canvas.yview)
        vbar.grid(row=0, column=1, sticky="ns")
        hbar = ttk.Scrollbar(parent, orient="horizontal", command=self.canvas.xview)
        hbar.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

        self._source_image: Image.Image | None = None
        self._image: Image.Image | None = None
        self._tk_image: ImageTk.PhotoImage | None = None
        self.display_scale: float = 1.0
        self.zoom_mode: str = "fit"
        self.zoom_factor: float = 1.0
        self._last_size: tuple[int, int] = (0, 0)
        self._render_after_id: str | None = None
        self._on_render: Callable[[], None] | None = None
        self._on_zoom: Callable[[int], None] | None = None

        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_wheel)
        self.canvas.bind("<Control-Button-4>", lambda e: self._zoom_by(1.25, e.x, e.y))
        self.canvas.bind("<Control-Button-5>", lambda e: self._zoom_by(0.8, e.x, e.y))
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<Button-4>", self._on_scroll)
        self.canvas.bind("<Button-5>", self._on_scroll)
        self.canvas.bind("<ButtonPress-1>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B1-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))

    @property
    def image(self) -> Image.Image | None:
        return self._image

    def set_render_callback(self, cb: Callable[[], None]) -> None:
        """Register a callback invoked after each render (for overlays)."""
        self._on_render = cb

    def set_zoom_callback(self, cb: Callable[[int], None]) -> None:
        """Register a callback invoked whenever zoom percentage changes."""
        self._on_zoom = cb
        if self._source_image is not None:
            cb(self.get_zoom_percent())

    def set_image(self, img: Image.Image | None) -> None:
        self._source_image = img
        if img is None:
            self._image = None
        self.zoom_mode = "fit"
        self.render()

    def render(self) -> None:
        src = self._source_image
        if src is None:
            return
        if src.mode != "RGB":
            src = src.convert("RGB")
            self._source_image = src
        w, h = src.size
        if w <= 0 or h <= 0:
            return
        scale = self._compute_scale(w, h)
        self.display_scale = scale
        dw, dh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))

        if (dw, dh) != (w, h):
            if dw * 2 <= w and dh * 2 <= h:
                try:
                    resized = src.reduce(2)
                except Exception:
                    resized = src.resize((dw, dh), RESAMPLE)
                while resized.width >= dw * 2 and resized.height >= dh * 2:
                    try:
                        resized = resized.reduce(2)
                    except Exception:
                        break
                resized = resized.resize((dw, dh), RESAMPLE)
            else:
                resized = src.resize((dw, dh), RESAMPLE)
        else:
            resized = src.copy()

        self._image = resized
        self.canvas.delete("base")
        self._tk_image = ImageTk.PhotoImage(resized)
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_image, tags="base")
        self.canvas.tag_lower("base")
        self.canvas.configure(scrollregion=(0, 0, dw, dh))

        if self._on_zoom:
            self._on_zoom(self.get_zoom_percent())
        if self._on_render:
            self._on_render()

    def request_render(self, delay: int = 60) -> None:
        """Schedule a coalesced render(); repeated calls collapse into one."""
        if self._render_after_id is not None:
            try:
                self.canvas.after_cancel(self._render_after_id)
            except Exception:
                pass
        self._render_after_id = self.canvas.after(delay, self._deferred_render)

    def _deferred_render(self) -> None:
        self._render_after_id = None
        self.render()

    def redraw_display(self) -> None:
        """Re-convert current display copy (*self._image*) to PhotoImage."""
        if self._image is None:
            return
        self.canvas.delete("base")
        self._tk_image = ImageTk.PhotoImage(self._image)
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_image, tags="base")
        self.canvas.tag_lower("base")
        w, h = self._image.size
        self.canvas.configure(scrollregion=(0, 0, max(1, w), max(1, h)))
        if self._on_render:
            self._on_render()

    def zoom_in(self) -> None:
        self._zoom_by(1.25)

    def zoom_out(self) -> None:
        self._zoom_by(0.8)

    def zoom_actual(self) -> None:
        if self._source_image is None:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        self._zoom_by(1.0 / self.display_scale if self.display_scale > 0 else 1.0, cw / 2.0, ch / 2.0)

    def zoom_fit(self) -> None:
        if self._source_image is None:
            return
        self.zoom_mode = "fit"
        self.render()
        self.canvas.xview_moveto(0.0)
        self.canvas.yview_moveto(0.0)

    def get_zoom_percent(self) -> int:
        return int(round(self.display_scale * 100))

    def clear(self) -> None:
        self.canvas.delete("all")
        self._image = None
        self._source_image = None
        self._tk_image = None
        if self._on_zoom:
            self._on_zoom(100)

    def _zoom_by(self, factor: float, anchor_x: float | None = None, anchor_y: float | None = None) -> None:
        if self._source_image is None:
            return
        old_scale = self.display_scale if self.display_scale > 0 else 1.0
        new_scale = self._clamp(old_scale * factor)
        if abs(new_scale - old_scale) < 1e-5:
            return

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        vx = anchor_x if anchor_x is not None else cw / 2.0
        vy = anchor_y if anchor_y is not None else ch / 2.0

        # Координаты на исходном изображении под курсором / центром
        img_x = self.canvas.canvasx(vx) / old_scale
        img_y = self.canvas.canvasy(vy) / old_scale

        self.zoom_mode = "manual"
        self.zoom_factor = new_scale
        self.render()

        # Восстанавливаем точку под курсором
        new_canvas_x = img_x * new_scale
        new_canvas_y = img_y * new_scale
        sr = self.canvas.cget("scrollregion")
        if sr:
            parts = [float(p) for p in sr.split()]
            sr_w = max(1.0, parts[2] - parts[0])
            sr_h = max(1.0, parts[3] - parts[1])
            if sr_w > cw:
                frac_x = max(0.0, min(1.0, (new_canvas_x - vx) / sr_w))
                self.canvas.xview_moveto(frac_x)
            else:
                self.canvas.xview_moveto(0.0)
            if sr_h > ch:
                frac_y = max(0.0, min(1.0, (new_canvas_y - vy) / sr_h))
                self.canvas.yview_moveto(frac_y)
            else:
                self.canvas.yview_moveto(0.0)

    def _clamp(self, scale: float) -> float:
        return max(MIN_ZOOM, min(MAX_ZOOM, float(scale)))

    def _compute_scale(self, img_w: int, img_h: int) -> float:
        if self.zoom_mode == "manual":
            return self._clamp(self.zoom_factor)
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 20 or ch <= 20:
            cw, ch = MAX_FALLBACK_W, MAX_FALLBACK_H
        aw, ah = max(80, cw - FIT_PADDING), max(80, ch - FIT_PADDING)
        scale = min(aw / float(img_w), ah / float(img_h))
        scale = min(scale, MAX_FIT_SCALE)
        max_by_dim = float(MAX_DISPLAY_DIM) / float(max(img_w, img_h))
        scale = min(scale, max_by_dim)
        return self._clamp(scale)

    def _on_configure(self, event: tk.Event) -> None:
        size = (event.width, event.height)
        if size == self._last_size:
            return
        self._last_size = size
        if self._source_image is not None and self.zoom_mode == "fit":
            self.request_render()

    def _on_ctrl_wheel(self, event: tk.Event) -> None:
        factor = 1.25 if event.delta > 0 else 0.8
        self._zoom_by(factor, anchor_x=event.x, anchor_y=event.y)

    def _on_scroll(self, event: tk.Event) -> None:
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
