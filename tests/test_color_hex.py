import os
import sys
import tempfile
import time
import tkinter as tk
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath("src"))

from core.seed_geometry import compute_median_color_hex, compute_median_rgb
from core.models import DEFAULT_CSV_FIELDS
from core.export import build_seed_csv_row, export_seed_items
from gui.inspection_app import InspectionApp
from gui.aug_inspection_app import AugInspectionApp


def test_core_color_calculation():
    print("Testing core color calculation...")
    # Synthetic image: 100x100 BGR, mostly (50, 100, 200) -> RGB (200, 100, 50)
    # With bright specular highlights (255, 255, 255)
    img_bgr = np.full((100, 100, 3), (50, 100, 200), dtype=np.uint8)
    # Add specular outlier highlights in 10% of pixels
    img_bgr[20:30, 20:30] = [255, 255, 255]
    img_bgr[70:80, 70:80] = [0, 0, 0]

    cnt = np.array([[[10, 10]], [[90, 10]], [[90, 90]], [[10, 90]]], dtype=np.int32)

    # In BGR mode (default for OpenCV images)
    rgb = compute_median_rgb(img_bgr, cnt, is_rgb=False)
    assert rgb == (200, 100, 50), f"Expected (200, 100, 50), got {rgb}"
    hex_code = compute_median_color_hex(img_bgr, cnt, is_rgb=False)
    assert hex_code == "#C86432", f"Expected #C86432, got {hex_code}"

    # In RGB mode (e.g. PIL Image)
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    hex_code_pil = compute_median_color_hex(pil_img, cnt)
    assert hex_code_pil == "#C86432", f"Expected #C86432 from PIL, got {hex_code_pil}"

    # Dict points
    dict_pts = [{"x": 10, "y": 10}, {"x": 90, "y": 10}, {"x": 90, "y": 90}, {"x": 10, "y": 90}]
    hex_dict = compute_median_color_hex(img_bgr, dict_pts, is_rgb=False)
    assert hex_dict == "#C86432", f"Expected #C86432 from dict points, got {hex_dict}"

    print("Core color calculation test passed!")


def test_gui_inspection():
    print("Testing InspectionApp GUI...")
    root = tk.Tk()
    root.withdraw()

    app = InspectionApp(root, "out", initial_index=0)
    for _ in range(50):
        root.update_idletasks()
        root.update()
        if app.zoom_widget._source_image is not None or app.zoom_widget.image is not None:
            break
        time.sleep(0.05)

    assert len(app.records) > 0, "No records loaded in InspectionApp"
    rec0 = app.records[0]
    print(f"Record 0 file_index: {rec0.get('file_index')}, color: {rec0.get('color_hex')}")
    assert rec0.get("color_hex") == "#D9BB7C", f"Expected #D9BB7C, got {rec0.get('color_hex')}"

    # Check UI entries
    assert app.color_hex_var.get() == "#D9BB7C", f"UI hex var mismatch: {app.color_hex_var.get()}"
    badge_bg = app.color_badge.cget("bg").lower()
    assert badge_bg == "#d9bb7c", f"Badge bg mismatch: {badge_bg}"

    src_img = app.zoom_widget._source_image or app.zoom_widget.image
    assert src_img is not None, "Image not loaded in zoom widget"

    # Simulate modifying contour point
    old_color = app.color_hex_var.get()
    # Contract contour to top-left corner
    app.contour_points = [(p[0] * 0.5, p[1] * 0.5) for p in app.contour_points]
    app._on_contour_modified()
    new_color = app.color_hex_var.get()
    print(f"Old color: {old_color} -> New contour live color: {new_color}")
    assert new_color.startswith("#"), f"Invalid live color: {new_color}"

    # Simulate saving contour
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_csv = os.path.join(tmp_dir, "seeds.csv")
        test_json = os.path.join(tmp_dir, "test_contour.json")
        app.contour_path = test_json
        app.output_path = test_csv
        saved = app._save_contour(show_message=False)
        assert saved, "Save contour failed"
        assert os.path.isfile(test_json), "JSON not saved"
        assert os.path.isfile(test_csv), "CSV not saved"
        assert app.records[0]["color_hex"] == new_color, "Record color not updated on save"

        # Check CSV content has color_hex
        with open(test_csv, "r", encoding="utf-8-sig") as f:
            header = f.readline().strip().split(",")
            first_row = f.readline().strip().split(",")
        assert "color_hex" in header, "color_hex column missing in CSV"
        c_idx = header.index("color_hex")
        assert first_row[c_idx] == new_color, f"CSV color_hex mismatch: {first_row[c_idx]} vs {new_color}"

    root.destroy()
    print("InspectionApp GUI test passed!")


def test_gui_aug_inspection():
    print("Testing AugInspectionApp GUI...")
    root = tk.Tk()
    root.withdraw()

    app = AugInspectionApp(root, "out", initial_index=0)
    for _ in range(50):
        root.update_idletasks()
        root.update()
        if app.zoom_widget._source_image is not None or app.zoom_widget.image is not None:
            break
        time.sleep(0.05)

    assert len(app.records) > 0, "No records loaded in AugInspectionApp"
    rec0 = app._current_record()
    print(f"Aug Record 0 file_index: {rec0.get('file_index')}, color: {rec0.get('color_hex')}")
    assert rec0.get("color_hex") == "#D9BB7C", f"Expected #D9BB7C, got {rec0.get('color_hex')}"
    assert app.prop_vars["color_hex"].get() == "#D9BB7C", f"Prop var color mismatch: {app.prop_vars['color_hex'].get()}"

    root.destroy()
    print("AugInspectionApp GUI test passed!")


if __name__ == "__main__":
    test_core_color_calculation()
    test_gui_inspection()
    test_gui_aug_inspection()
    print("\nALL COLOR TESTS PASSED SUCCESSFULLY!")
