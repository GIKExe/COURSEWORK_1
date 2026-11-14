# -*- coding: utf-8 -*-
"""Agreement figures for the current (perpendicular-cross) geometry.

Unlike ``figures.py`` this script does not read the stale analysis tables: it
recomputes every metric from ``out/seeds.csv`` and ``results/manual_reference.csv``.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

PHOTO_ORDER = ["1_1", "1_2", "1_3", "2_1", "2_1p", "2_2", "2_2p", "2_3p"]


def ru(x, nd=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}".replace(".", ",")


def ccc(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=0), y.var(ddof=0)
    cov = np.mean((x - mx) * (y - my))
    return float(2.0 * cov / (vx + vy + (mx - my) ** 2))


def load_data():
    seeds = pd.read_csv(ROOT / "out" / "seeds.csv", encoding="utf-8-sig")
    manual = pd.read_csv(RESULTS / "manual_reference.csv")
    manual = manual[["serial", "length_mm", "width_mm"]].rename(
        columns={"serial": "index", "length_mm": "manual_length",
                 "width_mm": "manual_width"}
    )
    for col in ("length_px", "width_px", "mm_per_px"):
        seeds[col] = pd.to_numeric(seeds[col], errors="coerce")
    seeds["length_mm"] = seeds["length_px"] * seeds["mm_per_px"]
    seeds["width_mm"] = seeds["width_px"] * seeds["mm_per_px"]
    seeds["photo"] = seeds["file_index"].astype(str).str.rsplit("_", n=1).str[0]

    per_seed = seeds.groupby("index").agg(
        prog_length=("length_mm", "mean"),
        prog_width=("width_mm", "mean"),
        photos=("file_index", "count"),
    ).reset_index()
    wide = per_seed.merge(manual, on="index", how="left")
    wide = wide[wide["manual_length"].notna()].copy()
    return seeds, wide


def metrics(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = y - x
    bias = float(d.mean())
    sd = float(d.std(ddof=1))
    loa_low = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd
    return {
        "n": int(len(d)),
        "bias": bias,
        "sd": sd,
        "mae": float(np.abs(d).mean()),
        "rmse": float(np.sqrt((d ** 2).mean())),
        "r": float(np.corrcoef(x, y)[0, 1]),
        "ccc": ccc(x, y),
        "loa_low": loa_low,
        "loa_high": loa_high,
        "within_loa_pct": float(((d >= loa_low) & (d <= loa_high)).mean() * 100.0),
        "within_05_pct": float((np.abs(d) <= 0.5).mean() * 100.0),
        "within_10_pct": float((np.abs(d) <= 1.0).mean() * 100.0),
    }


def fig_agreement(wide):
    panels = [
        ("length", "manual_length", "prog_length", "Длина"),
        ("width", "manual_width", "prog_width", "Ширина"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 6.5))
    for i, (par, mcol, pcol, title) in enumerate(panels):
        sub = wide[[mcol, pcol]].dropna()
        x = sub[mcol].to_numpy(dtype=float)
        y = sub[pcol].to_numpy(dtype=float)
        st = metrics(x, y)

        ax = axes[i, 0]
        ax.scatter(x, y, s=9, alpha=0.45, color=C_NP, linewidths=0)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        pad = 0.04 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color="k", lw=0.9, ls="--", label="1:1")
        b, a = np.polyfit(x, y, 1)
        ax.plot([lo, hi], a + b * np.array([lo, hi]), color=C_P, lw=1.0,
                label=f"МНК: y = {ru(a, 2)} + {ru(b, 3)}x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("Ручной замер, мм")
        ax.set_ylabel("Программа, мм")
        ax.set_title(f"({chr(97 + 2 * i)}) {title}: программа–вручную")
        ax.text(0.03, 0.97,
                f"n = {st['n']}\nсмещение = {ru(st['bias'], 3)} мм\n"
                f"MAE = {ru(st['mae'], 3)} мм\nr = {ru(st['r'], 3)}\n"
                f"CCC = {ru(st['ccc'], 3)}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
                bbox=dict(fc="white", ec="0.8", alpha=0.85, pad=2.5))
        ax.legend(loc="lower right", frameon=False, handlelength=1.5)

        ax = axes[i, 1]
        mean = (x + y) / 2.0
        diff = y - x
        ax.scatter(mean, diff, s=9, alpha=0.45, color=C_NP, linewidths=0)
        ax.axhline(0, color="k", lw=0.7, ls=":")
        ax.axhline(st["bias"], color=C_P, lw=1.1)
        ax.axhline(st["loa_low"], color=C_P, lw=0.9, ls="--")
        ax.axhline(st["loa_high"], color=C_P, lw=0.9, ls="--")
        ax.text(0.98, 0.96,
                f"смещение = {ru(st['bias'], 3)}\n"
                f"95% ПС = [{ru(st['loa_low'], 2)}; {ru(st['loa_high'], 2)}]\n"
                f"внутри ПС = {ru(st['within_loa_pct'], 1)}%",
                transform=ax.transAxes, va="top", ha="right", fontsize=7.5,
                bbox=dict(fc="white", ec="0.8", alpha=0.85, pad=2.5))
        ax.set_xlabel("Среднее (программа и вручную), мм")
        ax.set_ylabel("Разность (программа − вручную), мм")
        ax.set_title(f"({chr(98 + 2 * i)}) {title}: Блэнд–Альтман")
        lo_d, hi_d = diff.min(), diff.max()
        pad_d = 0.08 * (hi_d - lo_d)
        ax.set_ylim(lo_d - pad_d, hi_d + pad_d)
    fig.tight_layout()
    fig.savefig(FIG / "fig_agreement_cross.png")
    plt.close(fig)


def fig_error_distribution(wide):
    panels = [
        ("manual_length", "prog_length", "Длина"),
        ("manual_width", "prog_width", "Ширина"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, (mcol, pcol, title) in zip(axes, panels):
        sub = wide[[mcol, pcol]].dropna()
        x = sub[mcol].to_numpy(dtype=float)
        y = sub[pcol].to_numpy(dtype=float)
        d = y - x
        st = metrics(x, y)
        bins = np.linspace(d.min(), d.max(), 31)
        ax.hist(d, bins=bins, color=C_NP, alpha=0.72, edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="k", lw=0.7, ls=":")
        ax.axvline(st["bias"], color=C_P, lw=1.1,
                   label=f"смещение = {ru(st['bias'], 3)} мм")
        ax.axvline(st["bias"] - st["sd"], color=C_GRAY, lw=0.8, ls="--")
        ax.axvline(st["bias"] + st["sd"], color=C_GRAY, lw=0.8, ls="--",
                   label=f"±1 СКО = {ru(st['sd'], 3)} мм")
        ax.set_xlabel("Разность (программа − вручную), мм")
        ax.set_ylabel("Число зёрен")
        ax.set_title(f"{title}: распределение ошибки")
        ax.text(0.03, 0.97,
                f"MAE = {ru(st['mae'], 3)} мм\n"
                f"RMSE = {ru(st['rmse'], 3)} мм\n"
                f"±0,5 мм: {ru(st['within_05_pct'], 1)}%\n"
                f"±1,0 мм: {ru(st['within_10_pct'], 1)}%",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
                bbox=dict(fc="white", ec="0.8", alpha=0.85, pad=2.5))
        ax.legend(loc="lower right", frameon=True, facecolor="white",
                  edgecolor="0.8", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(FIG / "fig_error_distribution_cross.png")
    plt.close(fig)


def per_photo_metrics(seeds):
    rows = []
    manual = pd.read_csv(RESULTS / "manual_reference.csv")[
        ["serial", "length_mm", "width_mm"]
    ].rename(columns={"serial": "index", "length_mm": "manual_length",
                      "width_mm": "manual_width"})
    for photo in PHOTO_ORDER:
        sub = seeds[seeds["photo"] == photo]
        if sub.empty:
            continue
        q = sub.groupby("index").agg(
            prog_length=("length_mm", "mean"), prog_width=("width_mm", "mean")
        ).reset_index().merge(manual, on="index", how="left")
        q = q[q["manual_length"].notna()]
        for par, mcol, pcol in [("length", "manual_length", "prog_length"),
                                ("width", "manual_width", "prog_width")]:
            st = metrics(q[mcol], q[pcol])
            rows.append({"photo": photo, "parameter": par, **st})
    return pd.DataFrame(rows)


def fig_photo_bias(photo_df):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
    ypos = np.arange(len(PHOTO_ORDER))
    for ax, par, title in [(axes[0], "length", "Длина"),
                           (axes[1], "width", "Ширина")]:
        sub = photo_df[photo_df["parameter"] == par].set_index("photo").loc[PHOTO_ORDER]
        bias = sub["bias"].to_numpy()
        mae = sub["mae"].to_numpy()
        for y, b, m in zip(ypos, bias, mae):
            ax.plot([b, m], [y, y], color=C_GRAY, lw=0.8, alpha=0.6, zorder=1)
        ax.scatter(bias, ypos, s=34, color=C_P, zorder=3, label="смещение")
        ax.scatter(mae, ypos, s=34, marker="D", color=C_NP, zorder=3, label="MAE")
        ax.axvline(0, color="k", lw=0.7, ls=":")
        overall_bias = float((photo_df[photo_df["parameter"] == par]["bias"]).mean())
        ax.axvline(overall_bias, color=C_P, lw=0.8, ls="--", alpha=0.7)
        ax.set_yticks(ypos)
        ax.set_yticklabels(PHOTO_ORDER)
        ax.set_xlabel("Ошибка, мм")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_photo_bias_cross.png")
    plt.close(fig)


def write_metrics(wide, photo_df):
    rows = []
    for par, mcol, pcol in [("length", "manual_length", "prog_length"),
                            ("width", "manual_width", "prog_width")]:
        st = metrics(wide[mcol], wide[pcol])
        rows.append({"scope": "all", "parameter": par, **st})
    for _, r in photo_df.iterrows():
        rows.append({"scope": r["photo"], "parameter": r["parameter"],
                     **{k: r[k] for k in ("n", "bias", "sd", "mae", "rmse", "r",
                                          "ccc", "loa_low", "loa_high",
                                          "within_loa_pct")}})
    pd.DataFrame(rows).to_csv(RESULTS / "geometry_agreement_current.csv",
                              index=False, encoding="utf-8-sig")


def main():
    seeds, wide = load_data()
    photo_df = per_photo_metrics(seeds)
    fig_agreement(wide)
    fig_error_distribution(wide)
    fig_photo_bias(photo_df)
    write_metrics(wide, photo_df)
    print("figures written to", FIG)
    print("metrics written to", RESULTS / "geometry_agreement_current.csv")


if __name__ == "__main__":
    main()
