"""GUI tests for: (1) aug inspector contour/ruler on every seed switch,
(2) regular inspector new-point insertion between nearest neighbours."""
import json
import os
import sys
import shutil
import tempfile
import tkinter as tk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PIL import Image, ImageDraw  # noqa: E402
from gui.aug_inspection_app import AugInspectionApp  # noqa: E402
from gui.inspection_app import InspectionApp  # noqa: E402


def make_dataset(dst: str, n_seeds: int = 3) -> None:
    os.makedirs(dst, exist_ok=True)
    rows = ["file_index,index,status,angle_deg,length_mm,width_mm,mm_per_px,is_aug,aug_of,split"]
    for i in range(1, n_seeds + 1):
        fi = f"{i}_seed"
        img = Image.new("RGB", (400, 300), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.ellipse((100, 80, 300, 220), fill=(180, 140, 90))
        img.save(os.path.join(dst, fi + ".jpg"))
        pts = [[100, 150], [200, 80], [300, 150], [200, 220]]
        with open(os.path.join(dst, f"{fi}_contour.json"), "w") as f:
            json.dump({"points": pts}, f)
        rows.append(f"{fi},{i},ok,12.5,6.2,3.1,0.021,0,,train")
    with open(os.path.join(dst, "seeds.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")


def pixel_color_near(canvas, x, y):
    # sample the PhotoImage currently shown on canvas
    return None


def test_aug():
    root = tk.Tk()
    root.geometry("1200x800+0+0")
    tmp = tempfile.mkdtemp()
    try:
        make_dataset(tmp)
        app = AugInspectionApp(root, tmp)
        import traceback
        try:
            app.zoom_widget.set_image(Image.new("RGB", (400, 300), (0, 0, 255)))
            print("MANUAL SET_IMAGE -> zoom image:", app.zoom_widget.image)
        except Exception:
            traceback.print_exc()
        print("source:", app.zoom_widget._source_image, "canvas winfo:", app.zoom_widget.canvas.winfo_width(), app.zoom_widget.canvas.winfo_height())
        for _ in range(5):
            root.update_idletasks()
            root.update()
        assert len(app.filtered_indices) == 3, app.filtered_indices
        results = []
        for pos in range(3):
            app.current_f_idx = pos
            app._update_view()
            root.update()
            img = app.zoom_widget.image
            assert img is not None, f"no display image at pos {pos}"
            scale = app.zoom_widget.display_scale
            # find contour color pixels (#00FF66) in the display copy
            data = img.convert("RGB")
            cnt = 0
            for px, py in [(p[0] * scale, p[1] * scale) for p in app.current_contour_pts]:
                pass
            # scan along the closing segment of the polygon for greenish pixels
            pts = app.current_contour_pts
            found = False
            for k in range(len(pts)):
                x0, y0 = pts[k]
                x1, y1 = pts[(k + 1) % len(pts)]
                steps = 60
                for s in range(steps + 1):
                    X = int(round((x0 + (x1 - x0) * s / steps) * scale))
                    Y = int(round((y0 + (y1 - y0) * s / steps) * scale))
                    if 0 <= X < data.width and 0 <= Y < data.height:
                        r, g, b = data.getpixel((X, Y))
                        if g > 180 and r < 120 and b < 160:
                            found = True
                            break
                if found:
                    break
            results.append(found)
            # ruler should also be present (red pixels from draw_mm_ruler)
            red_found = any(
                (lambda c: c[0] > 200 and c[1] < 60 and c[2] < 60)(data.getpixel((X, Y)))
                for X in range(20, 30) for Y in range(0, data.height, 7)
            )
            results.append(red_found)
        assert all(results), f"overlay missing on some seeds: {results}"
        print("AUG TEST PASSED: contour+ruler drawn on all", 3, "seeds")
    finally:
        root.destroy()
        shutil.rmtree(tmp, ignore_errors=True)


def test_inspect_insert():
    root = tk.Tk()
    root.geometry("1200x800+0+0")
    tmp = tempfile.mkdtemp()
    try:
        make_dataset(tmp, n_seeds=1)
        app = InspectionApp(root, tmp)
        root.update_idletasks()
        root.update()
        # load contour of first record via internal loader
        app.contour_points = [(100, 150), (200, 80), (300, 150), (200, 220)]
        app.contour_path = os.path.join(tmp, "1_seed_contour.json")
        # New point near middle of segment p1->p2 (between (200,80) and (300,150))
        pt = (252.0, 112.0)
        idx = app._insertion_index(pt)
        assert idx == 2, f"expected insertion after index 1 (got {idx})"
        # New point near closing segment p3->p0 ((200,220)->(100,150))
        pt2 = (148.0, 182.0)
        idx2 = app._insertion_index(pt2)
        assert idx2 == 4 or idx2 == 0, f"closing-segment insert should be at end/front, got {idx2}"
        # simulate double click to ensure it inserts (not appends) & saves
        app.zoom_widget.canvas.winfo_exists()
        scale = app.zoom_widget.display_scale
        ev_x = 252.0 * scale - app.zoom_widget.canvas.canvasx(0)
        ev_y = 112.0 * scale - app.zoom_widget.canvas.canvasy(0)
        class E:
            x = int(ev_x); y = int(ev_y)
        before = len(app.contour_points)
        app._on_double_click(E())
        assert len(app.contour_points) == before + 1
        # the new point must NOT be last (unless inserted into closing seg)
        # check its neighbours are the endpoints of the closest segment
        new_i = app.contour_points.index((252.0, 112.0))
        nb = [app.contour_points[(new_i - 1) % len(app.contour_points)],
              app.contour_points[(new_i + 1) % len(app.contour_points)]]
        assert (200, 80) in nb and (300, 150) in nb, f"neighbours wrong: {nb} idx={new_i}"
        # saved json contains the same order
        with open(app.contour_path) as f:
            saved = json.load(f)["points"]
        assert [tuple(map(float, p)) for p in saved][new_i] == (252.0, 112.0)
        print("INSPECT INSERT TEST PASSED: new point wired to nearest neighbours")
    finally:
        root.destroy()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_aug()
    test_inspect_insert()
    print("ALL TESTS PASSED")
