"""Entry point: GUI mode (segmentation / inspection / aug inspection) or batch mode."""
from __future__ import annotations

import os
import sys
import tkinter as tk
import tkinter.font as tkfont

from gui.common.dpi import setup_dpi_awareness

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "out")
print(DEFAULT_OUT_DIR)


def _setup_tk_defaults(root: tk.Tk) -> None:
    try:
        dpi = root.winfo_fpixels("1i")
        root.tk.call("tk", "scaling", dpi / 72.0)
    except Exception:
        pass
    try:
        tkfont.nametofont("TkDefaultFont").configure(size=10)
    except Exception:
        pass


def run_gui(initial_screen: str = "segmentation") -> None:
    """Run the GUI loop, allowing screen switching via callbacks."""
    screen: str | None = initial_screen
    while screen:
        root = tk.Tk()
        _setup_tk_defaults(root)
        next_screen: list[str | None] = [None]

        def switch(target: str) -> None:
            nonlocal screen
            next_screen[0] = target
            root.destroy()

        match screen:
            case "segmentation":
                from gui.segmentation_app import SegmentationApp
                SegmentationApp(root, DEFAULT_OUT_DIR, switch_callback=switch)
            case "inspection":
                from gui.inspection_app import InspectionApp
                InspectionApp(root, DEFAULT_OUT_DIR, switch_callback=switch)
            case "aug_inspection":
                from gui.aug_inspection_app import AugInspectionApp
                aug_dir = os.path.join(BASE_DIR, "out_aug")
                data_dir = aug_dir if os.path.isdir(aug_dir) else DEFAULT_OUT_DIR
                AugInspectionApp(root, data_dir, switch_callback=switch)
            case _:
                root.destroy()
                break

        root.mainloop()
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