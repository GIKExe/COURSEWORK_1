"""Perf/responsiveness test for the aug inspector under simulated slow IO."""
import os, sys, time, threading
sys.path.insert(0, '/workspace/src')

# --- inject artificial slowness into PIL open (simulates big photos/disk) ---
import PIL.Image as _I
_orig_open = _I.open
def slow_open(*a, **k):
    time.sleep(0.15)  # 150ms decode penalty per uncached image
    return _orig_open(*a, **k)
_I.open = slow_open

import tkinter as tk
from gui.aug_inspection_app import AugInspectionApp

FIXTURE = '/workspace/tmp_repro/fixture'
root = tk.Tk()
root.withdraw()  # avoid map requirements; widgets still work

t0 = time.perf_counter()
app = AugInspectionApp(root, FIXTURE)
build_ms = (time.perf_counter() - t0) * 1000
print(f"UI build (must be fast, no dataset load blocking): {build_ms:.1f} ms")
assert build_ms < 300, "opening blocked on data load!"

# pump events until deferred load finishes and first image arrives
def pump(ms):
    end = time.perf_counter() + ms/1000
    while time.perf_counter() < end:
        root.update()
        time.sleep(0.002)

pump(600)
assert len(app.records) == 25, f"records not loaded: {len(app.records)}"
zc = app.zoom_widget
assert zc.image is not None, "first image not displayed"
print("first image shown OK, contour pts:", len(app.current_contour_pts))

# switch through all seeds measuring UI-blocking time per switch
blocked_times = []
for i in range(len(app.filtered_indices) - 1):
    t_switch = time.perf_counter()
    app.next_item()
    # measure how long the call itself took (this is what freezes the UI)
    blocked_times.append((time.perf_counter() - t_switch) * 1000)
    pump(60)  # let async delivery + render happen
    # after switching, contour overlay items must exist for THIS seed
    if app.current_contour_pts:
        n_items = len(zc.canvas.find_withtag("overlay"))
        assert n_items >= 2, f"contour not drawn after switch to #{i+2}: items={n_items}"
    assert zc.image is not None, f"image missing after switch #{i+2}"

worst = max(blocked_times)
avg = sum(blocked_times)/len(blocked_times)
print(f"per-switch UI block: avg {avg:.2f} ms, worst {worst:.2f} ms ({len(blocked_times)} switches)")
assert worst < 40, f"switching blocks the UI: {worst:.1f} ms"

# going back must be instant (cache hit): no sleep incurred
t = time.perf_counter()
app.prev_item(); app.prev_item(); app.prev_item()
back_ms = (time.perf_counter() - t) * 1000
print(f"3x prev via cache: {back_ms:.2f} ms")
assert back_ms < 120, "cache miss when navigating back"
pump(200)
assert zc.image is not None

# filter change responsiveness
t = time.perf_counter()
app.filter_mode.set("Только аугментации")
app._apply_filter()
filt_ms = (time.perf_counter()-t)*1000
print(f"filter apply: {filt_ms:.2f} ms"); assert filt_ms < 200
pump(300)
assert zc.image is not None

# checkbox toggles must not re-decode/re-resize photo (cheap redraw)
src_before = zc._source_image
t = time.perf_counter()
for _ in range(20):
    app.show_points_var.set(not app.show_points_var.get())
    app._redraw()
tog_ms = (time.perf_counter()-t)*1000
print(f"20 overlay toggles: {tog_ms:.2f} ms"); assert tog_ms < 150
assert zc._source_image is src_before, "toggle rebuilt source"

# zoom buttons coalesced
app.zoom_widget.zoom_in()
pump(300)
print("zoom percent:", app.zoom_widget.get_zoom_percent())
n_ov = len(zc.canvas.find_withtag("overlay"))
print("overlay items after zoom:", n_ov)
assert n_ov >= 2, "overlays lost after zoom"

root.destroy()
print("AUG PERF TEST PASSED")
