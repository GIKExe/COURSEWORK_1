# -*- coding: utf-8 -*-
"""Additional robust / rank metrics and photo-factor checks.

Reads only files saved under results/ by the main validation run.  Writes
small CSVs and LaTeX fragments used in results/main.tex.  The program code,
images and the manual workbook are not touched.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(__file__).resolve().parent
TABLES = RESULTS / "tables"
RNG_SEED = 20261111

wide = pd.read_csv(RESULTS / "dataset_seeds_wide.csv")
long = pd.read_csv(RESULTS / "dataset_long_length_width.csv")
per_photo = pd.read_csv(TABLES / "per_photo_agreement.csv")
seeds = pd.read_csv(RESULTS / "seeds_final.csv")

# ---------------------------------------------------------------- robust metrics
def robust_metrics(diff: np.ndarray, ref: np.ndarray) -> dict:
    d = np.asarray(diff, dtype=float)
    r = np.asarray(ref, dtype=float)
    d = d[np.isfinite(d) & np.isfinite(r)]
    r = r[np.isfinite(d) & np.isfinite(r)]
    n = len(d)
    q1, q3 = np.percentile(d, [25, 75])
    rho, p_rho = stats.spearmanr(d, r)
    mape = float(np.mean(np.abs(d) / r) * 100.0)
    med_ape = float(np.median(np.abs(d) / r) * 100.0)
    return {
        "n": n,
        "mean_bias": float(d.mean()),
        "median_bias": float(np.median(d)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "trim10": float(stats.trim_mean(d, 0.10)),
        "trim20": float(stats.trim_mean(d, 0.20)),
        "mape": mape,
        "median_ape": med_ape,
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
    }


rows = []
for par, dcol, mcol, zcol in [
    ("Длина", "length_bias_mm", "manual_length_mm", "length_robust_z"),
    ("Ширина", "width_bias_mm", "manual_width_mm", "width_robust_z"),
]:
    sub = wide[wide[dcol].notna()]
    m = robust_metrics(sub[dcol].values, sub[mcol].values)
    m["parameter"] = par
    # without flagged outliers
    keep = sub[np.abs(sub[zcol]) <= 3.5]
    m["n_clean"] = int(len(keep))
    m["bias_clean"] = float(keep[dcol].mean())
    m["mae_clean"] = float(keep[dcol].abs().mean())
    m["rmse_clean"] = float(np.sqrt((keep[dcol] ** 2).mean()))
    rows.append(m)
rob = pd.DataFrame(rows)[[
    "parameter", "n", "mean_bias", "median_bias", "q1", "q3", "iqr",
    "trim10", "trim20", "mape", "median_ape", "spearman_rho", "spearman_p",
    "n_clean", "bias_clean", "mae_clean", "rmse_clean",
]]
rob.to_csv(TABLES / "robust_metrics.csv", index=False, encoding="utf-8-sig")


def num(x, nd=3):
    if x is None or not math.isfinite(float(x)):
        return "---"
    return r"\num{" + f"{float(x):.{nd}f}" + "}"


lines = [
    r"\begin{tabular}{@{}lrrrrrr@{}}",
    r"\toprule",
    r"Признак & Медиана смещения, мм & IQR, мм & Среднее, усечённое на 10\%, мм &"
    r" MAPE, \% & $\rho$ Спирмена & Без выбросов: смещение / MAE, мм \\",
    r"\midrule",
]
for _, r in rob.iterrows():
    lines.append(
        f"{r['parameter']} & ${'+' if r['median_bias'] >= 0 else '-'}{num(abs(r['median_bias']))}$ & "
        f"{num(r['iqr'])} & {num(r['trim10'])} & {num(r['mape'],1)} & {num(r['spearman_rho'])} & "
        f"${'+' if r['bias_clean'] >= 0 else '-'}{num(abs(r['bias_clean']))}$ / {num(r['mae_clean'])} \\\\"
    )
lines += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "tex_robust.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

# macros for the text
macros = []
for _, r in rob.iterrows():
    tag = "Len" if r["parameter"] == "Длина" else "Wid"
    macros += [
        rf"\newcommand{{\{tag}MedianBias}}{{{'+' if r['median_bias'] >= 0 else '-'}{num(abs(r['median_bias']))}}}",
        rf"\newcommand{{\{tag}Iqr}}{{{num(r['iqr'])}}}",
        rf"\newcommand{{\{tag}TrimTen}}{{{num(r['trim10'])}}}",
        rf"\newcommand{{\{tag}Mape}}{{{num(r['mape'],1)}}}",
        rf"\newcommand{{\{tag}Spearman}}{{{num(r['spearman_rho'])}}}",
        rf"\newcommand{{\{tag}CleanBias}}{{{'+' if r['bias_clean'] >= 0 else '-'}{num(abs(r['bias_clean']))}}}",
        rf"\newcommand{{\{tag}CleanMae}}{{{num(r['mae_clean'])}}}",
        rf"\newcommand{{\{tag}CleanN}}{{{r['n_clean']}}}",
    ]
(TABLES / "tex_extra_numbers.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- photo factors
def hex_to_rgb(h: str):
    h = str(h).strip().lstrip("#")
    if len(h) != 6:
        return (np.nan, np.nan, np.nan)
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


seeds["photo"] = seeds["file_index"].str.rsplit("_", n=1).str[0]
rgb = seeds["color_hex"].map(hex_to_rgb)
seeds["R"] = [v[0] for v in rgb]
seeds["G"] = [v[1] for v in rgb]
seeds["B"] = [v[2] for v in rgb]
seeds["brightness"] = (seeds["R"] + seeds["G"] + seeds["B"]) / 3.0

photo_bright = seeds.groupby("photo")["brightness"].median()
photo_bright_sd = seeds.groupby("photo")["brightness"].std()
photo_edge = seeds.groupby("photo")["edge_width_px"].median()

fac = per_photo.copy()
fac["brightness"] = fac["photo"].map(photo_bright)
fac["brightness_sd"] = fac["photo"].map(photo_bright_sd)
fac["edge_width_median"] = fac["photo"].map(photo_edge)
fac = fac[["photo", "parameter", "mm_per_px", "brightness", "brightness_sd",
           "edge_width_median", "bias", "mae", "r"]]
fac.to_csv(TABLES / "photo_factors.csv", index=False, encoding="utf-8-sig")

# correlations between photo-level factors and bias (n = 8 photos, per parameter)
corr_rows = []
for par in ["length", "width"]:
    sub = fac[fac["parameter"] == par]
    for factor in ["mm_per_px", "brightness", "brightness_sd", "edge_width_median"]:
        r_p, p_p = stats.pearsonr(sub[factor], sub["bias"])
        r_s, p_s = stats.spearmanr(sub[factor], sub["bias"])
        corr_rows.append({
            "parameter": "Длина" if par == "length" else "Ширина",
            "factor": factor,
            "pearson_r": float(r_p), "pearson_p": float(p_p),
            "spearman_rho": float(r_s), "spearman_p": float(p_s),
        })
corr = pd.DataFrame(corr_rows)
corr.to_csv(TABLES / "photo_factor_correlations.csv", index=False, encoding="utf-8-sig")

piv = fac.pivot(index="photo", columns="parameter", values="bias").reset_index()
meta = fac.drop_duplicates("photo")[["photo", "mm_per_px", "brightness", "brightness_sd", "edge_width_median"]]
piv = piv.merge(meta, on="photo", how="left").sort_values("photo")
fac_lines = [
    r"\begin{tabular}{@{}lrrrrrr@{}}",
    r"\toprule",
    r"Снимок & Масштаб, мм/px & Медианная яркость &"
    r" Разброс яркости (СКО) & Медиана ширины границы, px & $\Delta L$, мм & $\Delta W$, мм \\",
    r"\midrule",
]
for _, r in piv.iterrows():
    fac_lines.append(
        f"{str(r['photo']).replace('_', chr(92)+'_')} & {num(r['mm_per_px'],6)} & "
        f"{num(r['brightness'],1)} & {num(r['brightness_sd'],1)} & {num(r['edge_width_median'],2)} & "
        f"${'+' if r['length'] >= 0 else '-'}{num(abs(r['length']))}$ & "
        f"${'+' if r['width'] >= 0 else '-'}{num(abs(r['width']))}$ \\\\"
    )
fac_lines += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "tex_factors.tex").write_text("\n".join(fac_lines) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- color stability
long_rgb = long["color_hex"].map(hex_to_rgb)
long["brightness"] = [(v[0] + v[1] + v[2]) / 3.0 for v in long_rgb]
g2 = long.groupby("serial")
stab = pd.DataFrame({
    "serial": g2.size().index,
    "brightness_mean": g2["brightness"].mean().values,
    "brightness_sd": g2["brightness"].std().values,
    "length_sd": g2["length_mm"].std().values,
    "width_sd": g2["width_mm"].std().values,
})
stab = stab.merge(wide[["serial", "row", "pos"]], on="serial", how="left")
stab.to_csv(TABLES / "color_stability.csv", index=False, encoding="utf-8-sig")
rho_l, p_l = stats.spearmanr(stab["brightness_sd"], stab["length_sd"], nan_policy="omit")
rho_w, p_w = stats.spearmanr(stab["brightness_sd"], stab["width_sd"], nan_policy="omit")

report = []
report.append("ROBUST METRICS")
report.append(rob.to_string(index=False))
report.append("")
report.append("PHOTO FACTOR CORRELATIONS (n=8)")
report.append(corr.to_string(index=False))
report.append("")
report.append(f"COLOR STABILITY: median within-seed brightness SD = {stab['brightness_sd'].median():.2f}")
report.append(f"Spearman(brightness SD, length SD) = {rho_l:.3f}, p={p_l:.4g}")
report.append(f"Spearman(brightness SD, width SD) = {rho_w:.3f}, p={p_w:.4g}")
(TABLES.parent / "analysis_extra_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
print("\n".join(report))
