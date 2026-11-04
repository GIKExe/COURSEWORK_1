from __future__ import annotations

import os
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, ttk
from typing import Any, Callable


class ToolTip:
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


def инициализировать_состояние_приложения(
    приложение: Any,
    root: tk.Tk,
    data_dir: str,
    switch_callback: Callable[..., None] | None = None,
    initial_index: int = 0,
    initial_file_index: str | None = None,
) -> None:
    приложение.root = root
    приложение.data_dir = os.path.abspath(data_dir)
    приложение._switch = switch_callback
    приложение.initial_index = initial_index
    приложение.initial_file_index = initial_file_index
    приложение.records = []


def создать_панель_модулей(
    parent: tk.Widget,
    активный_модуль: str,
    при_переключении: Callable[[str], None],
) -> ttk.Frame:
    модули = (
        ("segmentation", "✂ Сегментация"),
        ("inspection", "🔍 Инспектор"),
        ("aug_inspection", "🔥 Аугментации"),
    )
    панель = ttk.Frame(parent)
    панель.pack(side=tk.LEFT)
    for имя, текст in модули:
        if имя == активный_модуль:
            ttk.Button(панель, text=текст, state="disabled").pack(side=tk.LEFT, padx=1)
        else:
            ttk.Button(панель, text=текст, command=lambda m=имя: при_переключении(m)).pack(
                side=tk.LEFT, padx=1
            )
    return панель


def создать_панель_масштаба(
    parent: tk.Widget,
    приложение: Any,
) -> tuple[ttk.Frame, ttk.Label, dict[str, ttk.Button]]:
    фрейм = ttk.Frame(parent)
    фрейм.pack(side=tk.LEFT)
    кнопки: dict[str, ttk.Button] = {}
    for метод, текст, ширина in (
        ("zoom_in", "+", 3),
        ("zoom_out", "-", 3),
        ("zoom_actual", "1:1", 4),
        ("zoom_fit", "По окну", None),
    ):
        btn = ttk.Button(
            фрейм,
            text=текст,
            width=ширина,
            command=lambda m=метод: getattr(приложение.zoom_widget, m)(),
        )
        btn.pack(side=tk.LEFT, padx=1)
        кнопки[метод] = btn
    надпись = ttk.Label(фрейм, text="100%", width=6, anchor="center")
    надпись.pack(side=tk.LEFT, padx=3)
    return фрейм, надпись, кнопки


def создать_верхнюю_панель(parent: tk.Widget) -> ttk.Frame:
    панель = ttk.Frame(parent, padding=(6, 5))
    панель.pack(side=tk.TOP, fill=tk.X)
    return панель


def создать_блок_навигации(
    parent: tk.Widget,
    при_назад: Callable[[], None],
    при_вперёд: Callable[[], None],
    при_вводе_позиции: Callable[..., Any],
) -> tuple[ttk.Button, ttk.Label, tk.StringVar, ttk.Entry, ttk.Label, ttk.Button]:
    btn_prev = ttk.Button(parent, text="◀ Назад", width=10, command=при_назад)
    btn_prev.pack(side=tk.LEFT, padx=2)
    nav = ttk.Frame(parent)
    nav.pack(side=tk.LEFT, padx=4)
    pos_lbl = ttk.Label(nav, text="Семечка:")
    pos_lbl.pack(side=tk.LEFT)
    pos_var = tk.StringVar(value="0")
    pos_entry = ttk.Entry(nav, textvariable=pos_var, width=6, justify="center")
    pos_entry.pack(side=tk.LEFT, padx=4)
    pos_entry.bind("<Return>", при_вводе_позиции)
    total_lbl = ttk.Label(nav, text="из 0")
    total_lbl.pack(side=tk.LEFT)
    btn_next = ttk.Button(parent, text="Вперёд ▶", width=10, command=при_вперёд)
    btn_next.pack(side=tk.LEFT, padx=2)
    return btn_prev, pos_lbl, pos_var, pos_entry, total_lbl, btn_next


def создать_статус_строку(
    parent: tk.Widget,
    текст: str | None = None,
    переменная: tk.Variable | None = None,
    отступ: tuple[int, int] = (6, 3),
) -> ttk.Label:
    label = ttk.Label(
        parent,
        text=текст,
        textvariable=переменная,
        anchor="w",
        padding=отступ,
        relief="sunken",
    )
    label.pack(side=tk.BOTTOM, fill=tk.X)
    return label


def выбрать_папку(начальный_каталог: str, заголовок: str) -> str | None:
    путь = filedialog.askdirectory(initialdir=начальный_каталог, title=заголовок)
    return путь or None