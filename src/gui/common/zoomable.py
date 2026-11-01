"""Zoomable canvas widget shared by all GUI applications."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from PIL import Image, ImageTk

RESAMPLE = Image.Resampling.LANCZOS

MIN_ZOOM = 0.02
MAX_ZOOM = 32.0
MAX_FIT_SCALE = 12.0
FIT_PADDING = 26
MAX_DISPLAY_DIM = 12000
MAX_FALLBACK_W = 1100
MAX_FALLBACK_H = 800


class ZoomableCanvas:
    """Encapsulates zoom/pan logic on a tk.Canvas with scrollbars.

    Usage:
        zc = ZoomableCanvas(parent_frame)
        zc.pack(fill="both", expand=True)
        zc.set_image(pil_image)
    """

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
        self._on_render: Callable[[], None] | None = None

        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_wheel)
        self.canvas.bind("<Control-Button-4>", lambda e: self._zoom_by(1.25))
        self.canvas.bind("<Control-Button-5>", lambda e: self._zoom_by(0.8))
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

    def set_image(self, img: Image.Image | None) -> None:
        self._source_image = img
        if img is None:
            # keep the previous display scale so that a later redraw_display()
            # does not work with stale geometry when nothing can be rendered
            self._image = None
        self.zoom_mode = "fit"
        self.render()

    def render(self) -> None:
        self.canvas.delete("all")
        src = self._source_image
        if src is None:
            # Nothing to show yet (e.g. window not mapped and no fallback).
            # Do NOT drop a previously rendered display copy: callers draw
            # overlays into self.image right after render() returns.
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
        # Resize a pristine copy every time: the source must never be mutated,
        # otherwise overlays drawn into it would accumulate / disappear between
        # renders (the aug-inspector contour bug).
        resized = src.resize((dw, dh), RESAMPLE) if (dw, dh) != (w, h) else src.copy()
        self._image = resized
        self._tk_image = ImageTk.PhotoImage(resized)
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_image, tags="base")
        self.canvas.configure(scrollregion=(0, 0, dw, dh))
        if self._on_render:
            self._on_render()

    def redraw_display(self) -> None:
        """Re-convert the current display copy (*self._image*) to a PhotoImage.

        Used after somebody drew on ``self.image`` (e.g. the mm ruler): the
        canvas bitmap is refreshed from the mutated display copy and the
        render callback re-draws vector overlays on top of it.  The source
        image stays untouched, so a later ``render()`` starts from clean data.
        """
        if self._image is None:
            return
        self.canvas.delete("all")
        self._tk_image = ImageTk.PhotoImage(self._image)
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_image, tags="base")
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
        self.zoom_mode = "manual"
        self.zoom_factor = 1.0
        self.render()

    def zoom_fit(self) -> None:
        if self._source_image is None:
            return
        self.zoom_mode = "fit"
        self.render()

    def get_zoom_percent(self) -> int:
        return int(round(self.display_scale * 100))

    def clear(self) -> None:
        self.canvas.delete("all")
        self._image = None
        self._source_image = None
        self._tk_image = None

    def _zoom_by(self, factor: float) -> None:
        if self._source_image is None:
            return
        current = self.display_scale if self.display_scale > 0 else 1.0
        self.zoom_mode = "manual"
        self.zoom_factor = self._clamp(current * factor)
        self.render()

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
            self.render()

    def _on_ctrl_wheel(self, event: tk.Event) -> None:
        self._zoom_by(1.25 if event.delta > 0 else 0.8)

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