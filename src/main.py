"""Entry point: GUI mode (segmentation / inspection / aug inspection) or batch mode."""
from __future__ import annotations

import gc
import os
import sys
import tkinter as tk
import tkinter.font as tkfont

from gui.common.dpi import setup_dpi_awareness

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.getcwd()
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "out")


def _setup_tk_defaults(root: tk.Tk) -> None:
    try:
        dpi = root.winfo_fpixels("1i")
        root.tk.call("tk", "scaling", dpi / 72.0)
    except:  # noqa: E722, S110
        pass

    try:
        tkfont.nametofont("TkDefaultFont").configure(size=10)
    except:  # noqa: E722, S110
        pass


def run_gui(initial_screen: str = "segmentation") -> None:
    """Run the GUI loop, allowing screen switching via callbacks."""
    screen: str | None = initial_screen
    current_out_dir: str = DEFAULT_OUT_DIR
    current_seed_index: int = 0
    current_file_index: str | None = None

    while screen:
        next_screen: list[str | None] = [None]
        root = tk.Tk()
        _setup_tk_defaults(root)
        
        def switch(target: str, data_dir: str | None = None, seed_index: int = 0, file_index: str | None = None) -> None:
            nonlocal screen, current_out_dir, current_seed_index, current_file_index
            next_screen[0] = target
            if data_dir:
                current_out_dir = data_dir
            current_seed_index = seed_index
            current_file_index = file_index
            root.destroy()

        app = None
        match screen:
            case "segmentation":
                from gui.segmentation_app import SegmentationApp
                app = SegmentationApp(root, current_out_dir, switch_callback=switch)
            case "inspection":
                from gui.inspection_app import InspectionApp
                app = InspectionApp(root, current_out_dir, switch_callback=switch, initial_index=current_seed_index, initial_file_index=current_file_index)
            case "aug_inspection":
                from gui.aug_inspection_app import AugInspectionApp
                app = AugInspectionApp(root, current_out_dir, switch_callback=switch, initial_index=current_seed_index, initial_file_index=current_file_index)
            case _:
                root.destroy()
                break

        root.mainloop()
        if app is not None:
            if hasattr(app, "data_dir") and getattr(app, "data_dir", None):
                current_out_dir = app.data_dir
            elif hasattr(app, "out_dir") and getattr(app, "out_dir", None):
                current_out_dir = app.out_dir
        # Окно уничтожено — освобождаем объекты прежнего экрана прямо здесь,
        # в главном потоке. tkinter-переменные (StringVar и др.) при
        # финализации вызывают Tcl-команды, а из фонового потока _tkinter
        # ждёт, пока главный поток войдёт в mainloop, и кидает
        # «RuntimeError: main thread is not in main loop». Фоновым потоком
        # у нас является декодер картинок (image-decode): пока он «долбит»
        # мусор от прошлого экрана, загрузка изображения нового экрана
        # виснет на секунды — контур на холсте уже нарисован, а картинки
        # нет. Собираем мусор сразу же, в главном потоке и вне mainloop:
        # там этот вызов безопасен (та же нить, интерпрейс ещё жив).
        app = None
        root = None
        gc.collect()
        screen = next_screen[0]


def main() -> None:
    setup_dpi_awareness()
    if len(sys.argv) > 1:
        from cli.batch import batch_main
        batch_main(sys.argv[1:], DEFAULT_OUT_DIR)
    else:
        run_gui("segmentation")


if __name__ == "__main__":
    main()