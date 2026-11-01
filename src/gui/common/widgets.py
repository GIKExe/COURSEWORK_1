"""Reusable Tkinter widgets."""
from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont


class ToolTip:
    """Simple hover tooltip for any Tkinter widget."""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self._widget = widget
        self._text = text
        self._tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _show(self, event: tk.Event | None = None) -> None:
        if self._tip or not self._text:
            return
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 5
        self._tip = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        try:
            tw.attributes("-topmost", True)
        except Exception:
            pass
        tk.Label(
            tw, text=self._text, justify="left", background="#ffffe0",
            foreground="#000000", relief="solid", borderwidth=1,
            font=tkfont.nametofont("TkDefaultFont"), wraplength=320,
        ).pack(ipadx=4, ipady=2)

    def _hide(self, event: tk.Event | None = None) -> None:
        if self._tip:
            self._tip.destroy()
            self._tip = None