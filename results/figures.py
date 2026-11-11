# -*- coding: utf-8 -*-
"""Figures for the validation paper.  Reads the analysis outputs only."""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.font_manager import findfont, FontProperties
from PIL import Image, ImageDraw, ImageFont

RESULTS = Path(__file__).resolve().parent
ROOT = RESULTS.parent
FIG = RESULTS / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_P = "#D55E00"
C_NP = "#0072B2"
C_GRAY = "#555555"
C_GREEN = "#009E73"
C_PURPLE = "#CC79A7"


def ru(x, nd=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}".replace(".", ",")


def fmt_p(p):
    if p is None or not math.isfinite(p):
        return "—"
    if p < 0.001:
        return "<0,001"
    return ru(p, 3)


def load():
    T = RESULTS / "tables"
    wide = pd.read_csv(RESULTS / "dataset_seeds_wide.csv")
    long = pd.read_csv(RESULTS / "dataset_long_length_width.csv")
    overall = pd.read_csv(T / "overall_agreement_full.csv").set_index("label")
    photo = pd.read_csv(T / "per_photo_agreement.csv")
    rep = pd.read_csv(T / "repeatability.csv")
    pairs = pd.read_csv(T / "paired_comparisons.csv")
    cal = pd.read_csv(T / "calibration.csv")
    groove = pd.read_csv(T / "groove_by_photo.csv")
    kdf = pd.read_csv(T / "missing_k.csv")
    grid_l = pd.read_csv(T / "grid_length_bias.csv", index_col=0)
    grid_w = pd.read_csv(T / "grid_width_bias.csv", index_col=0)
    return wide, long, overall, photo, rep, pairs, cal, groove, kdf, grid_l, grid_w


# ---------------------------------------------------------------------------
def fig_agreement(wide, overall):
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 6.5))
    for i, (par, mcol, pcol, title) in enumerate([
        ("length", "manual_length_mm", "prog_length_mean_mm", "Длина"),
        ("width", "manual_width_mm", "prog_width_mean_mm", "Ширина"),
    ]):
        sub = wide[[mcol, pcol]].dropna()
        x, y = sub[mcol].to_numpy(), sub[pcol].to_numpy()
        o = overall.loc[par]
        ax = axes[i, 0]
        ax.scatter(x, y, s=9, alpha=0.45, color=C_NP, linewidths=0)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        pad = 0.04 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color="k", lw=0.9, ls="--", label="1:1")
        b, a = np.polyfit(x, y, 1)
        ax.plot([lo, hi], a + b * np.array([lo, hi]), color=C_P, lw=1.0,
                label=f"МНК: y = {ru(a, 2)} + {ru(b, 3)}x")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("Ручной замер, мм")
        ax.set_ylabel("Программа, мм")
        ax.set_title(f"({chr(97+2*i)}) {title}: программа–вручную")
        ax.text(0.03, 0.97,
                f"n = {int(o['n'])}\nсмещение = {ru(o['bias'], 3)} мм\n"
                f"MAE = {ru(o['mae'], 3)} мм\nr = {ru(o['r'], 3)}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
                bbox=dict(fc="white", ec="0.8", alpha=0.85, pad=2.5))
        ax.legend(loc="lower right", frameon=False, handlelength=1.5)

        ax = axes[i, 1]
        mean = (x + y) / 2.0
        diff = y - x
        ax.scatter(mean, diff, s=9, alpha=0.45, color=C_NP, linewidths=0)
        ax.axhline(0, color="k", lw=0.7, ls=":")
        ax.axhline(o["bias"], color=C_P, lw=1.1)
        ax.axhline(o["loa_low"], color=C_P, lw=0.9, ls="--")
        ax.axhline(o["loa_high"], color=C_P, lw=0.9, ls="--")
        ax.text(0.98, 0.96,
                f"смещение = {ru(o['bias'], 3)}\n"
                f"95% ПС = [{ru(o['loa_low'], 2)}; {ru(o['loa_high'], 2)}]\n"
                f"внутри ПС = {ru(o['within_loa_pct'], 1)}%",
                transform=ax.transAxes, va="top", ha="right", fontsize=7.5,
                bbox=dict(fc="white", ec="0.8", alpha=0.85, pad=2.5))
        ax.set_xlabel("Среднее (программа и вручную), мм")
        ax.set_ylabel("Разность (программа − вручную), мм")
        ax.set_title(f"({chr(98+2*i)}) {title}: Блэнд–Альтман")
        lo_d, hi_d = diff.min(), diff.max()
        pad_d = 0.08 * (hi_d - lo_d)
        ax.set_ylim(lo_d - pad_d, hi_d + pad_d)
    fig.tight_layout()
    fig.savefig(FIG / "fig_agreement.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_repeatability(wide, rep):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, par, col, title in [
        (axes[0], "length", "prog_length_sd_mm", "Длина"),
        (axes[1], "width", "prog_width_sd_mm", "Ширина"),
    ]:
        v = wide[col].dropna().to_numpy()
        ax.hist(v, bins=30, color=C_NP, alpha=0.75, edgecolor="white", linewidth=0.4)
        med = float(np.median(v))
        row = rep[(rep["group"] == "все 8 снимков") & (rep["parameter"] == par)].iloc[0]
        ax.axvline(med, color=C_P, lw=1.1, label=f"медиана = {ru(med, 3)} мм")
        ax.axvline(row["sw"], color="k", lw=0.9, ls="--",
                   label=f"объединённое СКО = {ru(row['sw'], 3)} мм")
        ax.set_xlabel("Внутризерновое СКО по 8 снимкам, мм")
        ax.set_ylabel("Число зёрен")
        ax.set_title(f"{title}: ICC = {ru(row['icc'], 3)}")
        ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_repeatability.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_photo_bias(photo):
    order = ["1_1", "1_2", "1_3", "2_1", "2_1p", "2_2", "2_2p", "2_3p"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
    ypos = np.arange(len(order))
    for ax, par, title in [(axes[0], "length", "Длина"),
                           (axes[1], "width", "Ширина")]:
        sub = photo[photo["parameter"] == par].set_index("photo").loc[order]
        colors = [C_P if x == "P" else C_NP for x in sub["orientation"]]
        markers = ["s" if x == "P" else "o" for x in sub["orientation"]]
        for y, (_, r), c, m in zip(ypos, sub.iterrows(), colors, markers):
            ax.errorbar(r["bias"], y,
                        xerr=[[r["bias"] - r["bias_ci_low"]], [r["bias_ci_high"] - r["bias"]]],
                        fmt=m, color=c, ms=5, capsize=2.5, lw=1.1)
            if r["focus_claim"] == "заявлен лучший":
                ax.plot(r["bias"], y, marker="*", color="k", ms=7, zorder=5)
        ax.axvline(0, color="k", lw=0.7, ls=":")
        if par == "width":
            ax.axvline(photo[photo["parameter"] == "width"]["bias"].mean(),
                       color=C_GRAY, lw=0.8, ls="--")
        ax.set_yticks(ypos)
        ax.set_yticklabels(order)
        ax.set_xlabel("Смещение (программа − вручную), мм")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    # legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color=C_NP, lw=1, label="non-P"),
        Line2D([0], [0], marker="s", color=C_P, lw=1, label="P (бороздкой вверх)"),
        Line2D([0], [0], marker="*", color="k", lw=0, label="заявлен лучший фокус"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "fig_photo_bias.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_orientation(long, pairs):
    pivL = long.pivot_table(index="serial", columns="photo", values="length_mm", aggfunc="mean")
    pivW = long.pivot_table(index="serial", columns="photo", values="width_mm", aggfunc="mean")
    cats = [("2_1p − 2_1", "2_1", "2_1p"),
            ("2_2p − 2_2", "2_2", "2_2p"),
            ("P − non-P\n(средние)", None, None)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=False)
    rng = np.random.default_rng(7)
    for ax, piv, par, title in [(axes[0], pivL, "length", "Длина"),
                                (axes[1], pivW, "width", "Ширина")]:
        data = []
        labels = []
        for name, a, b in cats:
            if a is None:
                pv = long[long["photo"].isin(["2_1p", "2_2p", "2_3p"])].groupby("serial")[f"{par}_mm"].mean()
                nv = long[~long["photo"].isin(["2_1p", "2_2p", "2_3p"])].groupby("serial")[f"{par}_mm"].mean()
                d = (pv - nv).dropna().to_numpy()
            else:
                d = (piv[b] - piv[a]).dropna().to_numpy()
            data.append(d)
            labels.append(name)
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color="k", lw=1.1),
                        whiskerprops=dict(color=C_GRAY, lw=0.8),
                        capprops=dict(color=C_GRAY, lw=0.8),
                        boxprops=dict(facecolor="white", edgecolor=C_GRAY, lw=0.8))
        for j, d in enumerate(data, 1):
            ax.scatter(np.full(len(d), j) + rng.normal(0, 0.055, len(d)), d,
                       s=3, alpha=0.25, color=C_NP, linewidths=0, zorder=3)
        ax.axhline(0, color="k", lw=0.7, ls=":")
        ax.set_xticks(range(1, len(cats) + 1))
        ax.set_xticklabels([f"{name}\nn = {len(d)}" for name, d in zip(labels, data)],
                           fontsize=7.5)
        ax.set_ylabel("Разность, мм")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        # annotate means from table where available
        for j, name in enumerate(["2_1p − 2_1", "2_2p − 2_2"]):
            row = pairs[(pairs["comparison"] == name) & (pairs["parameter"] == par)]
            if len(row):
                r = row.iloc[0]
                ax.text(j + 1, ax.get_ylim()[0], f"{ru(r['mean_diff'], 3)}\n{fmt_p(r['p_paired_t'])}",
                        ha="center", va="bottom", fontsize=6.8, color=C_P)
    fig.tight_layout()
    fig.savefig(FIG / "fig_orientation.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_status(long):
    long = long.copy()
    for par in ["length", "width"]:
        long[f"dev_{par}"] = long[f"{par}_mm"] - long.groupby("serial")[f"{par}_mm"].transform("mean")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, par, title in [(axes[0], "length", "Длина"), (axes[1], "width", "Ширина")]:
        groups = [long.loc[long["status"] == st, f"dev_{par}"].dropna().to_numpy()
                  for st in ["focused", "transition", "blurry"]]
        bp = ax.boxplot(groups, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color="k", lw=1.1),
                        whiskerprops=dict(color=C_GRAY, lw=0.8),
                        capprops=dict(color=C_GRAY, lw=0.8),
                        boxprops=dict(facecolor="white", edgecolor=C_GRAY, lw=0.8))
        ns = [len(d) for d in groups]
        for j, d in enumerate(groups, 1):
            ax.scatter(np.full(len(d), j) + np.random.default_rng(j).normal(0, 0.05, len(d)), d,
                       s=2.5, alpha=0.18, color=C_NP, linewidths=0, zorder=3)
        ax.axhline(0, color="k", lw=0.7, ls=":")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([f"{s}\nn = {n}" for s, n in
                            zip(["focused", "transition", "blurry"], ns)], fontsize=8)
        ax.set_ylabel("Отклонение от среднего зерна, мм")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_status.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_grid(grid_l, grid_w):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))
    for ax, grid, title, cmap, vmin, vmax, cbl in [
        (axes[0], grid_l, "Длина", "Reds", -0.1, 1.0, "Смещение, мм"),
        (axes[1], grid_w, "Ширина", "viridis", -1.5, -0.2, "Смещение, мм"),
    ]:
        data = grid.to_numpy(dtype=float)
        data = np.ma.masked_invalid(data)
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad("#e6e6e6")
        im = ax.imshow(data, aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax,
                       origin="upper")
        ax.set_xticks([0, 4, 9, 14, 19, 24])
        ax.set_xticklabels([1, 5, 10, 15, 20, 25])
        ax.set_yticks([0, 4, 9, 12])
        ax.set_yticklabels([1, 5, 10, 13])
        ax.set_xlabel("Позиция в ряду")
        ax.set_ylabel("Ряд")
        ax.set_title(title)
        ax.grid(False)
        # missing seed
        ax.plot(24, 10, marker="x", color="k", ms=7, mew=1.4)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(cbl)
    axes[0].text(23.55, 10, "нет\nзамера", fontsize=6, va="center", ha="right",
                 color="k", bbox=dict(fc="white", ec="0.8", alpha=0.85, pad=1.2))
    fig.tight_layout()
    fig.savefig(FIG / "fig_grid.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_groove(groove):
    order = ["1_1", "1_2", "1_3", "2_1", "2_1p", "2_2", "2_2p", "2_3p"]
    g = groove.set_index("photo").loc[order]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(order))
    colors = [C_P if o == "P" else C_NP for o in g["orientation"]]
    rate = 100 * g["rate"].to_numpy()
    lo = 100 * g["wilson_low"].to_numpy()
    hi = 100 * g["wilson_high"].to_numpy()
    ax.bar(x, rate, color=colors, alpha=0.85, width=0.65)
    ax.errorbar(x, rate, yerr=[rate - lo, hi - rate], fmt="none", ecolor="k",
                capsize=3, lw=0.9)
    for xi, r, hi_i in zip(x, rate, hi):
        ax.text(xi, hi_i + 1.5, f"{ru(r, 1)}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{o}\nn = {int(n)}" for o, n in zip(order, g["n"])], fontsize=7.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Доля зёрен с обнаруженной бороздкой, %")
    ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_P, label="P (бороздкой вверх)"),
                       Patch(color=C_NP, label="non-P")],
              frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig_groove.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_detection():
    photo = "1_2"
    img = cv2.imread(str(ROOT / "images" / f"{photo}.jpg"))
    if img is None:
        return
    H, W = img.shape[:2]
    target_w = 1800
    s = target_w / W
    small = cv2.resize(img, (target_w, int(round(H * s))), interpolation=cv2.INTER_AREA)
    coords = pd.read_csv(RESULTS / "coords_with_grid.csv")
    coords = coords[coords["photo"] == photo]
    # contours
    for idx in coords["index"]:
        p = ROOT / "out" / f"{photo}_{int(idx):03d}_contour.json"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        pts = np.asarray(d["points"], dtype=np.float32).reshape(-1, 2)
        pts[:, 0] += d["crop_origin_in_source"][0]
        pts[:, 1] += d["crop_origin_in_source"][1]
        pts = (pts * s).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(small, [pts], True, (0, 200, 0), 1, cv2.LINE_AA)
    # centres and grid labels
    for _, r in coords.iterrows():
        cv2.circle(small, (int(r["cx"] * s), int(r["cy"] * s)), 2, (0, 0, 255), -1)
    row_cy = coords.groupby("row_derived")["cy"].mean() * s
    col_cx = coords.groupby("pos_derived")["cx"].mean() * s
    for row, y in row_cy.items():
        cv2.putText(small, str(int(row)), (8, int(y) + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(small, str(int(row)), (8, int(y) + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    for col, x in col_cx.items():
        cv2.putText(small, str(int(col)), (int(x) - 6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(small, str(int(col)), (int(x) - 6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    out = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(9.5, 9.5 * out.shape[0] / out.shape[1]))
    ax.imshow(out)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(FIG / "fig_detection.png", dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_width_definition(wide):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8))
    # schematic ellipse
    ax = axes[0]
    a, b = 1.0, 0.36
    ell = patches.Ellipse((0, 0), 2 * a, 2 * b, facecolor="#cfe3f3",
                          edgecolor=C_NP, lw=1.2)
    ax.add_patch(ell)
    ax.annotate("", xy=(a, 0), xytext=(-a, 0),
                arrowprops=dict(arrowstyle="<->", color="k", lw=0.8))
    ax.annotate("", xy=(0, b), xytext=(0, -b),
                arrowprops=dict(arrowstyle="<->", color=C_P, lw=0.9))
    ax.text(0, b + 0.10, "W_max = 2b", color=C_P, ha="center", fontsize=9)
    ax.text(0.30, -0.10, "L = 2a", color="k", ha="left", fontsize=9)
    ax.text(-0.55, -0.16, "S = πab", ha="center", fontsize=8.5, color=C_GRAY)
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-0.78, 0.60)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Схема: средняя хорда", pad=6)
    fig.text(0.25, 0.055,
             "S/L = πb/2 = (π/4)·W_max ≈ 0,785·W_max",
             ha="center", va="bottom", fontsize=9,
             bbox=dict(fc="white", ec="0.8", pad=3))
    # real seed example
    ax = axes[1]
    p = ROOT / "out" / "1_2_001_contour.json"
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    pts = np.asarray(d["points"], dtype=np.float32).reshape(-1, 2)
    pts = pts + np.asarray(d["crop_origin_in_source"], dtype=np.float32)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    ax.plot(np.r_[pts[:, 0], pts[0, 0]], np.r_[pts[:, 1], pts[0, 1]],
            color=C_NP, lw=1.2)
    ax.plot(np.r_[box[:, 0], box[0, 0]], np.r_[box[:, 1], box[0, 1]],
            color=C_P, lw=1.0, ls="--")
    c = rect[0]
    ax.plot(c[0], c[1], "k.", ms=3)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.invert_yaxis()
    ax.set_xlim(pts[:, 0].min() - 4, pts[:, 0].max() + 4)
    ax.set_ylim(pts[:, 1].max() + 4, pts[:, 1].min() - 4)
    ax.set_title("Реальное зерно № 1 (1_2)", pad=6)
    cm = pd.read_csv(RESULTS / "contour_metrics.csv")
    row = cm[(cm["photo"] == "1_2") & (cm["serial"] == 1)].iloc[0]
    fig.text(0.75, 0.055,
             f"W_max (minAreaRect) ≈ {ru(row['rect_short_mm'], 2)} мм; "
             f"S/L ≈ {ru(row['width_hull_mm'], 2)} мм; "
             f"отношение ≈ {ru(row['width_hull_mm']/row['rect_short_mm'], 3)}",
             ha="center", va="bottom", fontsize=8.2,
             bbox=dict(fc="white", ec="0.8", pad=3))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.24, wspace=0.06)
    fig.savefig(FIG / "fig_width_definition.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_calibration(cal):
    order = ["1_1", "1_2", "1_3", "2_1", "2_1p", "2_2", "2_2p", "2_3p"]
    c = cal.set_index("photo").loc[order]
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    x = np.arange(len(order))
    v = c["rel_diff_pct"].to_numpy()
    colors = [C_P if val > 0 else C_NP for val in v]
    ax.bar(x, v, color=colors, alpha=0.85, width=0.65)
    ax.axhline(0, color="k", lw=0.7)
    ax.axhline(2, color=C_GRAY, lw=0.8, ls="--")
    ax.axhline(-2, color=C_GRAY, lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("Отличие от независимой\nоценки по линейке, %")
    ax.set_ylim(-3, 2.5)
    ax.grid(axis="x", visible=False)
    ax.text(7.35, 2.1, "±2%", ha="right", fontsize=7, color=C_GRAY)
    fig.tight_layout()
    fig.savefig(FIG / "fig_calibration.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_missing_k(kdf):
    fig, ax1 = plt.subplots(figsize=(6.6, 2.9))
    ax1.plot(kdf["k"], kdf["mae_length"], color=C_NP, marker="o", ms=2.6, lw=1.0,
             label="MAE длины, мм")
    ax1.set_xlabel("Пропущенная позиция k в ряду 11")
    ax1.set_ylabel("MAE длины, мм", color=C_NP)
    ax1.tick_params(axis="y", labelcolor=C_NP)
    ax1.set_xticks(range(1, 26, 2))
    ax1.set_ylim(0.33, 0.67)
    ax2 = ax1.twinx()
    ax2.plot(kdf["k"], kdf["r_length"], color=C_P, marker="s", ms=2.2, lw=0.9,
             alpha=0.8, label="r")
    ax2.set_ylabel("Коэффициент корреляции r", color=C_P)
    ax2.tick_params(axis="y", labelcolor=C_P)
    ax2.set_ylim(-0.02, 1.02)
    ax1.axvline(25, color="k", lw=0.8, ls="--")
    ax1.axvline(19, color=C_GREEN, lw=0.8, ls=":")
    ax1.text(24.7, 0.336, "k = 25", ha="right", va="bottom", fontsize=7)
    ax1.text(18.7, 0.336, "k = 19", ha="right", va="bottom", fontsize=7,
             color=C_GREEN)
    ax1.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_missing_k.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_worst(wide, long):
    font_path = findfont(FontProperties(family="DejaVu Sans"))
    # choose worst 5 by absolute bias for each parameter
    items = []
    for par in ["length", "width"]:
        sub = wide[["serial", "row", "pos", f"manual_{par}_mm", f"prog_{par}_mean_mm",
                    f"{par}_bias_mm"]].dropna()
        sub = sub.reindex(sub[f"{par}_bias_mm"].abs().sort_values(ascending=False).index)
        for _, r in sub.head(5).iterrows():
            # photo with the largest absolute deviation for this seed
            obs = long[(long["serial"] == r["serial"])][
                ["file_index", f"{par}_bias_mm"]].dropna()
            idx = obs[f"{par}_bias_mm"].abs().idxmax()
            items.append((par, r, obs.loc[idx, "file_index"]))
    cols = 5
    cell_w, cell_h = 180, 200
    margin_l, margin_t = 85, 35
    canvas = Image.new("RGB", (margin_l + cols * cell_w, margin_t + 2 * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    f_small = ImageFont.truetype(font_path, 15)
    f_mid = ImageFont.truetype(font_path, 17)
    f_head = ImageFont.truetype(font_path, 19)
    draw.text((10, 8), "Худшие по |смещению| зёрна", font=f_head, fill="black")
    for i, (par, r, file_index) in enumerate(items):
        row_i = 0 if par == "length" else 1
        col_i = i % 5
        x0 = margin_l + col_i * cell_w
        y0 = margin_t + row_i * cell_h
        crop_path = ROOT / "out" / f"{file_index}.jpg"
        if crop_path.exists():
            im = Image.open(crop_path).convert("RGB")
            im.thumbnail((cell_w - 12, 105), Image.LANCZOS)
            canvas.paste(im, (x0 + (cell_w - im.width) // 2, y0 + 2))
        manual = r[f"manual_{par}_mm"]
        prog = r[f"prog_{par}_mean_mm"]
        bias = r[f"{par}_bias_mm"]
        label = "Длина" if par == "length" else "Ширина"
        draw.text((x0 + 4, y0 + 107),
                  f"{label}, № {int(r['serial'])}", font=f_small, fill="black")
        draw.text((x0 + 4, y0 + 124),
                  f"ряд {int(r['row'])}, поз. {int(r['pos'])}", font=f_small, fill="black")
        draw.text((x0 + 4, y0 + 141),
                  f"{ru(manual, 2)} → {ru(prog, 2)} мм", font=f_small, fill="black")
        draw.text((x0 + 4, y0 + 158),
                  f"Δ = {bias:+.2f} мм".replace(".", ","), font=f_mid,
                  fill=(180, 0, 0) if bias > 0 else (0, 70, 160))
    canvas.save(FIG / "fig_worst.png", dpi=(200, 200))
    plt.close("all")


# ---------------------------------------------------------------------------
def main():
    wide, long, overall, photo, rep, pairs, cal, groove, kdf, grid_l, grid_w = load()
    fig_agreement(wide, overall)
    fig_repeatability(wide, rep)
    fig_photo_bias(photo)
    fig_orientation(long, pairs)
    fig_status(long)
    fig_grid(grid_l, grid_w)
    fig_groove(groove)
    fig_detection()
    fig_width_definition(wide)
    fig_calibration(cal)
    fig_missing_k(kdf)
    fig_worst(wide, long)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
