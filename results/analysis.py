# -*- coding: utf-8 -*-
"""
Validation analysis for the seed-measurement program.

Reads the exported program CSV, the parsed manual reference, saved contours,
and run logs.  Writes analysis-ready datasets and summary tables under
results/.  The program code, images and the manual workbook are not modified.

All paths are relative to the project root (parent of this file's directory).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(__file__).resolve().parent
ROOT = RESULTS.parent
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
TABLES.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

RNG_SEED = 20261111
BOOT = 5000

PHOTOS = ["1_1", "1_2", "1_3", "2_1", "2_1p", "2_2", "2_2p", "2_3p"]
P_PHOTOS = ["2_1p", "2_2p", "2_3p"]
NONP_PHOTOS = ["1_1", "1_2", "1_3", "2_1", "2_2"]
BEST_FOCUS = ["1_2", "2_1", "2_1p"]          # operator's claim, not verified
ORIENTATION = {p: ("P" if p in P_PHOTOS else "non-P") for p in PHOTOS}
FOCUS_CLAIM = {p: ("заявлен лучший" if p in BEST_FOCUS else "остальные") for p in PHOTOS}

PAIR_21 = ("2_1", "2_1p")
PAIR_22 = ("2_2", "2_2p")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def fnum(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return ""
    return f"{x:.{nd}f}"


def wilson_ci(k, n, z=1.959963984540054):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def robust_z(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(x)
    return 0.6744897501960817 * (x - med) / mad


def icc_1_1(mat):
    """One-way random-effects ICC(1,1) for a complete n x k matrix."""
    mat = np.asarray(mat, dtype=float)
    n, k = mat.shape
    means = mat.mean(axis=1)
    grand = mat.mean()
    ssb = k * np.sum((means - grand) ** 2)
    ssw = np.sum((mat - means[:, None]) ** 2)
    msb = ssb / (n - 1)
    msw = ssw / (n * (k - 1))
    icc = (msb - msw) / (msb + (k - 1) * msw)
    return float(icc), float(msw), float(msb)


def icc_boot(mat, n_boot=1000, seed=RNG_SEED):
    mat = np.asarray(mat, dtype=float)
    n = mat.shape[0]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(icc_1_1(mat[idx])[0])
        except Exception:
            pass
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def agreement(prog, manual, *, boot=BOOT, seed=RNG_SEED, label=""):
    """Agreement of program values with manual values, program - manual."""
    prog = np.asarray(prog, dtype=float)
    manual = np.asarray(manual, dtype=float)
    ok = np.isfinite(prog) & np.isfinite(manual)
    x, y = prog[ok], manual[ok]
    n = len(x)
    if n < 3:
        return {"label": label, "n": n}
    d = x - y
    bias = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    mae = float(np.abs(d).mean())
    rmse = float(np.sqrt(np.mean(d ** 2)))
    r = float(np.corrcoef(x, y)[0, 1])
    ccc = float(
        2.0 * r * x.std(ddof=1) * y.std(ddof=1)
        / (x.var(ddof=1) + y.var(ddof=1) + (x.mean() - y.mean()) ** 2)
    )
    lr = stats.linregress(y, x)
    slope, intercept = float(lr.slope), float(lr.intercept)
    se_slope, se_int = float(lr.stderr), float(lr.intercept_stderr)
    p_slope1 = float(2.0 * stats.t.sf(abs((slope - 1.0) / se_slope), n - 2))
    p_int0 = float(2.0 * stats.t.sf(abs(intercept / se_int), n - 2))
    loa_lo, loa_hi = bias - 1.96 * sd, bias + 1.96 * sd
    ratio = x / y
    # bootstrap percentile intervals
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (boot, n))
    db = d[idx]
    bias_b = db.mean(axis=1)
    mae_b = np.abs(db).mean(axis=1)
    rmse_b = np.sqrt((db ** 2).mean(axis=1))
    r_b = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in idx])
    ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    out = {
        "label": label,
        "n": int(n),
        "manual_mean": float(y.mean()),
        "manual_sd": float(y.std(ddof=1)),
        "prog_mean": float(x.mean()),
        "prog_sd": float(x.std(ddof=1)),
        "bias": bias,
        "bias_sd": sd,
        "bias_se": se,
        "bias_ci_low": bias - tcrit * se,
        "bias_ci_high": bias + tcrit * se,
        "bias_boot_low": ci(bias_b)[0],
        "bias_boot_high": ci(bias_b)[1],
        "mae": mae,
        "mae_ci_low": ci(mae_b)[0],
        "mae_ci_high": ci(mae_b)[1],
        "rmse": rmse,
        "rmse_ci_low": ci(rmse_b)[0],
        "rmse_ci_high": ci(rmse_b)[1],
        "r": r,
        "r_ci_low": ci(r_b)[0],
        "r_ci_high": ci(r_b)[1],
        "ccc": ccc,
        "slope": slope,
        "slope_ci_low": slope - tcrit * se_slope,
        "slope_ci_high": slope + tcrit * se_slope,
        "slope_se": se_slope,
        "intercept": intercept,
        "intercept_ci_low": intercept - tcrit * se_int,
        "intercept_ci_high": intercept + tcrit * se_int,
        "intercept_se": se_int,
        "p_slope1": p_slope1,
        "p_intercept0": p_int0,
        "loa_low": loa_lo,
        "loa_high": loa_hi,
        "within_loa_pct": float(np.mean(np.abs(d - bias) <= 1.96 * sd) * 100.0),
        "ratio_mean": float(ratio.mean()),
        "ratio_sd": float(ratio.std(ddof=1)),
        "bias_pct": float(100.0 * bias / y.mean()),
        "mae_pct": float(100.0 * mae / y.mean()),
        "median_abs_err": float(np.median(np.abs(d))),
        "max_abs_err": float(np.max(np.abs(d))),
        "diff": d,
        "prog": x,
        "manual": y,
    }
    return out


def table_from_agreements(rows):
    keys = [
        "label", "n", "manual_mean", "manual_sd", "prog_mean", "prog_sd",
        "bias", "bias_ci_low", "bias_ci_high", "bias_sd", "mae", "mae_ci_low",
        "mae_ci_high", "rmse", "rmse_ci_low", "rmse_ci_high", "r", "r_ci_low",
        "r_ci_high", "ccc", "slope", "slope_ci_low", "slope_ci_high",
        "intercept", "intercept_ci_low", "intercept_ci_high", "p_slope1",
        "p_intercept0", "loa_low", "loa_high", "within_loa_pct", "ratio_mean",
        "ratio_sd", "bias_pct", "mae_pct", "median_abs_err", "max_abs_err",
    ]
    return pd.DataFrame([{k: r.get(k) for k in keys} for r in rows])


# ---------------------------------------------------------------------------
# load and link data
# ---------------------------------------------------------------------------
def load_data():
    seeds = pd.read_csv(RESULTS / "seeds_final.csv")
    seeds["photo"] = seeds["file_index"].str.rsplit("_", n=1).str[0]
    seeds["index"] = seeds["index"].astype(int)
    seeds["serial"] = seeds["index"]
    seeds["length_mm"] = seeds["length_px"].astype(float) * seeds["mm_per_px"].astype(float)
    seeds["width_mm"] = seeds["width_px"].astype(float) * seeds["mm_per_px"].astype(float)
    seeds["orientation"] = seeds["photo"].map(ORIENTATION)
    seeds["focus_claim"] = seeds["photo"].map(FOCUS_CLAIM)

    manual = pd.read_csv(RESULTS / "manual_reference.csv")
    manual["serial"] = manual["serial"].astype(int)
    manual["row"] = manual["row"].astype(int)
    manual["pos"] = manual["pos"].astype(int)
    for c in ["length_mm", "width_mm", "thickness_mm", "mass_g"]:
        manual[c] = pd.to_numeric(manual[c], errors="coerce")

    # verify the manual grid numbering
    manual["row_check"] = (manual["serial"] - 1) // 25 + 1
    manual["pos_check"] = (manual["serial"] - 1) % 25 + 1
    assert (manual["row"] == manual["row_check"]).all()
    assert (manual["pos"] == manual["pos_check"]).all()

    long = seeds.merge(
        manual[["serial", "row", "pos", "length_mm", "width_mm", "thickness_mm", "mass_g"]],
        on="serial", how="left", suffixes=("", "_manual"),
    )
    long = long.rename(columns={
        "length_mm": "length_mm", "width_mm": "width_mm",
        "length_mm_manual": "manual_length_mm", "width_mm_manual": "manual_width_mm",
    })
    long["length_bias_mm"] = long["length_mm"] - long["manual_length_mm"]
    long["width_bias_mm"] = long["width_mm"] - long["manual_width_mm"]
    long["length_abs_err_mm"] = long["length_bias_mm"].abs()
    long["width_abs_err_mm"] = long["width_bias_mm"].abs()

    # wide per-seed summary
    g = long.groupby("serial")
    wide = g.agg(
        n_photos=("photo", "nunique"),
        prog_length_mean_mm=("length_mm", "mean"),
        prog_length_sd_mm=("length_mm", "std"),
        prog_length_min_mm=("length_mm", "min"),
        prog_length_max_mm=("length_mm", "max"),
        prog_width_mean_mm=("width_mm", "mean"),
        prog_width_sd_mm=("width_mm", "std"),
        prog_width_min_mm=("width_mm", "min"),
        prog_width_max_mm=("width_mm", "max"),
        n_groove=("has_groove", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
        mean_edge_width_px=("edge_width_px", "mean"),
    ).reset_index()
    wide["prog_length_range_mm"] = wide["prog_length_max_mm"] - wide["prog_length_min_mm"]
    wide["prog_width_range_mm"] = wide["prog_width_max_mm"] - wide["prog_width_min_mm"]
    wide["prog_length_cv_pct"] = 100.0 * wide["prog_length_sd_mm"] / wide["prog_length_mean_mm"]
    wide["prog_width_cv_pct"] = 100.0 * wide["prog_width_sd_mm"] / wide["prog_width_mean_mm"]
    wide["groove_rate"] = wide["n_groove"] / wide["n_photos"]

    st = pd.crosstab(long["serial"], long["status"])
    for c in ["focused", "transition", "blurry", "unknown"]:
        if c not in st.columns:
            st[c] = 0
    st = st.rename(columns={c: f"n_{c}" for c in st.columns}).reset_index()
    wide = wide.merge(st, on="serial", how="left")

    wide = wide.merge(
        manual[["serial", "row", "pos", "length_mm", "width_mm", "thickness_mm", "mass_g"]],
        on="serial", how="left",
    ).rename(columns={
        "length_mm": "manual_length_mm", "width_mm": "manual_width_mm",
        "thickness_mm": "manual_thickness_mm", "mass_g": "manual_mass_g",
    })
    wide["length_bias_mm"] = wide["prog_length_mean_mm"] - wide["manual_length_mm"]
    wide["width_bias_mm"] = wide["prog_width_mean_mm"] - wide["manual_width_mm"]
    wide["length_abs_err_mm"] = wide["length_bias_mm"].abs()
    wide["width_abs_err_mm"] = wide["width_bias_mm"].abs()
    wide["missing_manual"] = wide["manual_length_mm"].isna()
    wide["length_robust_z"] = robust_z(wide["length_bias_mm"])
    wide["width_robust_z"] = robust_z(wide["width_bias_mm"])
    wide["flag_length_outlier"] = wide["length_robust_z"].abs() > 3.5
    wide["flag_width_outlier"] = wide["width_robust_z"].abs() > 3.5

    return seeds, manual, long, wide


# ---------------------------------------------------------------------------
# contour-based alternative geometry (post hoc, from saved contours)
# ---------------------------------------------------------------------------
def contour_metrics(seeds, manual):
    rows = []
    match_err = []
    for r in seeds.itertuples(index=False):
        p = ROOT / "out" / f"{r.file_index}_contour.json"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        pts = np.asarray(d["points"], dtype=np.float32).reshape(-1, 2)
        if len(pts) < 3:
            continue
        hull = cv2.convexHull(pts).reshape(-1, 2)
        hull_area = float(cv2.contourArea(hull))
        rect = cv2.minAreaRect(pts)
        (_, _), (rw, rh), _ = rect
        rect_long, rect_short = float(max(rw, rh)), float(min(rw, rh))
        ang = math.radians(float(r.angle_deg))
        u = pts[:, 0] * math.cos(ang) + pts[:, 1] * math.sin(ang)
        v = -pts[:, 0] * math.sin(ang) + pts[:, 1] * math.cos(ang)
        length_extent = float(u.max() - u.min())
        width_perp = float(v.max() - v.min())
        width_hull = float(hull_area / float(r.length_px))
        width_exported = float(r.width_px)
        width_min = min(width_hull, rect_short)
        match_err.append(abs(width_exported - width_min))
        rows.append({
            "file_index": r.file_index, "serial": int(r.index), "photo": r.photo,
            "length_px": float(r.length_px), "width_px": width_exported,
            "area_px": float(r.area_px), "hull_area_px": hull_area,
            "rect_long_px": rect_long, "rect_short_px": rect_short,
            "length_extent_px": length_extent, "width_perp_px": width_perp,
            "width_hull_px": width_hull, "width_min_px": width_min,
            "branch": "hull" if width_hull <= rect_short else "rect",
            "width_match_err_px": abs(width_exported - width_min),
            "mm_per_px": float(r.mm_per_px),
        })
    cm = pd.DataFrame(rows)
    for c in ["length_extent_px", "width_perp_px", "rect_long_px", "rect_short_px",
              "width_hull_px", "width_min_px"]:
        cm[c.replace("_px", "_mm")] = cm[c] * cm["mm_per_px"]
    cm["length_mm"] = cm["length_px"] * cm["mm_per_px"]
    cm["width_mm"] = cm["width_px"] * cm["mm_per_px"]
    cm["hull_area_mm2"] = cm["hull_area_px"] * cm["mm_per_px"] ** 2
    man_alt = manual[["serial", "row", "pos", "length_mm", "width_mm"]].rename(
        columns={"length_mm": "manual_length_mm", "width_mm": "manual_width_mm"})
    cm = cm.merge(man_alt, on="serial", how="left")

    # per-seed means for comparisons
    alt = cm.groupby("serial").agg(
        length_exported_mm=("length_mm", "mean"),
        length_extent_mm=("length_extent_mm", "mean"),
        length_rect_mm=("rect_long_mm", "mean"),
        width_exported_mm=("width_mm", "mean"),
        width_hull_mm=("width_hull_mm", "mean"),
        width_rect_mm=("rect_short_mm", "mean"),
        width_perp_mm=("width_perp_mm", "mean"),
        mm_per_px=("mm_per_px", "mean"),
        hull_wins=("branch", lambda s: float(np.mean(np.asarray(s) == "hull"))),
    ).reset_index()
    alt = alt.merge(
        manual[["serial", "length_mm", "width_mm"]].rename(
            columns={"length_mm": "manual_length_mm", "width_mm": "manual_width_mm"}),
        on="serial", how="left",
    )
    return cm, alt


# ---------------------------------------------------------------------------
# main analysis
# ---------------------------------------------------------------------------
def main():
    seeds, manual, long, wide = load_data()
    long.to_csv(RESULTS / "dataset_long_length_width.csv", index=False, encoding="utf-8-sig")
    wide.to_csv(RESULTS / "dataset_seeds_wide.csv", index=False, encoding="utf-8-sig")

    report = []
    def say(*args):
        line = " ".join(str(a) for a in args)
        print(line)
        report.append(line)

    say("=" * 78)
    say("VALIDATION ANALYSIS")
    say("=" * 78)
    say(f"seeds CSV: {len(seeds)} rows, {seeds['photo'].nunique()} photos, "
        f"{seeds['serial'].nunique()} unique indices")
    say(f"manual reference: {len(manual)} records, {manual['length_mm'].notna().sum()} measured")
    say(f"long dataset: {len(long)} rows; wide dataset: {len(wide)} rows")

    # --- overall agreement ---
    matched = wide[~wide["missing_manual"]].copy()
    overall_rows = []
    for par, pcol, mcol in [
        ("length", "prog_length_mean_mm", "manual_length_mm"),
        ("width", "prog_width_mean_mm", "manual_width_mm"),
    ]:
        a = agreement(matched[pcol], matched[mcol], label=par)
        overall_rows.append(a)
        say(f"\nOVERALL {par}: n={a['n']} manual={a['manual_mean']:.4f}±{a['manual_sd']:.4f} "
            f"prog={a['prog_mean']:.4f}±{a['prog_sd']:.4f} bias={a['bias']:+.4f} "
            f"[{a['bias_ci_low']:+.4f},{a['bias_ci_high']:+.4f}] sd={a['bias_sd']:.4f} "
            f"MAE={a['mae']:.4f} [{a['mae_ci_low']:.4f},{a['mae_ci_high']:.4f}] "
            f"RMSE={a['rmse']:.4f} r={a['r']:.4f} CCC={a['ccc']:.4f} "
            f"slope={a['slope']:.4f} int={a['intercept']:+.4f} "
            f"LoA=[{a['loa_low']:+.4f},{a['loa_high']:+.4f}] ratio={a['ratio_mean']:.4f}")
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
                  for r in overall_rows]).to_csv(TABLES / "overall_agreement.csv", index=False, encoding="utf-8-sig")
    table_from_agreements(overall_rows).to_csv(TABLES / "overall_agreement_full.csv", index=False, encoding="utf-8-sig")

    # width ratio test: exported width is a mean chord; for an ideal ellipse
    # the ratio of mean chord to maximum width is pi/4.
    ratio = (matched["prog_width_mean_mm"] / matched["manual_width_mm"]).dropna()
    nrat = len(ratio)
    rse = ratio.std(ddof=1) / math.sqrt(nrat)
    rt = stats.t.ppf(0.975, nrat - 1)
    ttest = stats.ttest_1samp(ratio, np.pi / 4)
    ratio_row = {
        "n": nrat, "mean": float(ratio.mean()), "sd": float(ratio.std(ddof=1)),
        "median": float(ratio.median()),
        "ci_low": float(ratio.mean() - rt * rse), "ci_high": float(ratio.mean() + rt * rse),
        "pi_over_4": float(np.pi / 4), "p_vs_pi4": float(ttest.pvalue),
        "predicted_width_mean_pi4": float(matched["manual_width_mm"].mean() * np.pi / 4),
        "actual_width_mean": float(matched["prog_width_mean_mm"].mean()),
    }
    pd.DataFrame([ratio_row]).to_csv(TABLES / "width_ratio.csv", index=False, encoding="utf-8-sig")
    say(f"\nWIDTH RATIO: mean={ratio_row['mean']:.4f} sd={ratio_row['sd']:.4f} "
        f"CI=[{ratio_row['ci_low']:.4f},{ratio_row['ci_high']:.4f}] "
        f"pi/4={ratio_row['pi_over_4']:.4f} p={ratio_row['p_vs_pi4']:.3g}; "
        f"pi/4*manual mean={ratio_row['predicted_width_mean_pi4']:.4f} vs actual={ratio_row['actual_width_mean']:.4f}")

    # --- per-photo agreement ---
    photo_rows = []
    for ph in PHOTOS:
        sub = long[long["photo"] == ph]
        for par, pcol, mcol in [
            ("length", "length_mm", "manual_length_mm"),
            ("width", "width_mm", "manual_width_mm"),
        ]:
            a = agreement(sub[pcol], sub[mcol], boot=2000, label=f"{ph}:{par}")
            photo_rows.append(a)
    photo_df = table_from_agreements(photo_rows)
    photo_df["photo"] = [r["label"].split(":")[0] for r in photo_rows]
    photo_df["parameter"] = [r["label"].split(":")[1] for r in photo_rows]
    photo_df["orientation"] = photo_df["photo"].map(ORIENTATION)
    photo_df["focus_claim"] = photo_df["photo"].map(FOCUS_CLAIM)
    cal = pd.read_csv(RESULTS / "ruler_scale_check.csv")
    cal_map = dict(zip(cal["photo"], cal["mm_per_px_program"]))
    photo_df["mm_per_px"] = photo_df["photo"].map(cal_map)
    photo_df.to_csv(TABLES / "per_photo_agreement.csv", index=False, encoding="utf-8-sig")
    say("\nPER-PHOTO")
    for _, r in photo_df.iterrows():
        say(f"{r['photo']:>5} {r['parameter']:>6} n={int(r['n'])} "
            f"bias={r['bias']:+.3f} MAE={r['mae']:.3f} r={r['r']:.3f} "
            f"mm/px={r['mm_per_px']:.6f}")

    # --- subgroup agreement ---
    def subset_agreement(name, mask, pcol, mcol):
        sub = matched[mask]
        return agreement(sub[pcol], sub[mcol], boot=3000, label=name)

    subgroup_rows = []
    for par, pcol, mcol in [("length", "prog_length_mean_mm", "manual_length_mm"),
                            ("width", "prog_width_mean_mm", "manual_width_mm")]:
        subgroup_rows.append(subset_agreement("все зерна", pd.Series(True, index=matched.index), pcol, mcol))
        subgroup_rows.append(subset_agreement("без ряда 11", matched["row"] != 11, pcol, mcol))
        subgroup_rows.append(subset_agreement("только ряд 11", matched["row"] == 11, pcol, mcol))
        # P / non-P means per seed
        pv = long[long["photo"].isin(P_PHOTOS)].groupby("serial")[[pcol.replace("prog_", "") if False else "length_mm" if par == "length" else "width_mm"]].mean()
        nv = long[long["photo"].isin(NONP_PHOTOS)].groupby("serial")[["length_mm" if par == "length" else "width_mm"]].mean()
        pv = pv.rename(columns={("length_mm" if par == "length" else "width_mm"): "pmean"})
        nv = nv.rename(columns={("length_mm" if par == "length" else "width_mm"): "nmean"})
        mm = matched[["serial", mcol]].merge(pv, on="serial").merge(nv, on="serial")
        # label rows manually, not through agreement labels
        subgroup_rows.append(agreement(mm["pmean"], mm[mcol], boot=3000, label=f"P-снимки ({par})"))
        subgroup_rows.append(agreement(mm["nmean"], mm[mcol], boot=3000, label=f"non-P снимки ({par})"))
        # best-focus vs rest
        bv = long[long["photo"].isin(BEST_FOCUS)].groupby("serial")[["length_mm" if par == "length" else "width_mm"]].mean().rename(columns={("length_mm" if par == "length" else "width_mm"): "bmean"})
        rv = long[~long["photo"].isin(BEST_FOCUS)].groupby("serial")[["length_mm" if par == "length" else "width_mm"]].mean().rename(columns={("length_mm" if par == "length" else "width_mm"): "rmean"})
        mm2 = matched[["serial", mcol]].merge(bv, on="serial").merge(rv, on="serial")
        subgroup_rows.append(agreement(mm2["bmean"], mm2[mcol], boot=3000, label=f"заявленный лучший фокус ({par})"))
        subgroup_rows.append(agreement(mm2["rmean"], mm2[mcol], boot=3000, label=f"остальные снимки ({par})"))
    subgroup_df = table_from_agreements(subgroup_rows)
    subgroup_df.to_csv(TABLES / "subgroup_agreement.csv", index=False, encoding="utf-8-sig")
    say("\nSUBGROUPS")
    for _, r in subgroup_df.iterrows():
        say(f"{r['label']:<32} n={int(r['n'])} bias={r['bias']:+.3f} MAE={r['mae']:.3f} r={r['r']:.3f}")

    # --- repeatability / reproducibility ---
    rep_rows = []
    def add_rep(name, photos):
        sub = long[long["photo"].isin(photos)]
        for par, col in [("length", "length_mm"), ("width", "width_mm")]:
            mat = sub.pivot_table(index="serial", columns="photo", values=col, aggfunc="mean").dropna()
            k = mat.shape[1]
            icc, msw, msb = icc_1_1(mat.values)
            lo, hi = icc_boot(mat.values, n_boot=800)
            sw = math.sqrt(msw)
            sd_i = mat.std(axis=1, ddof=1)
            cv_i = 100.0 * sd_i / mat.mean(axis=1)
            rng_i = mat.max(axis=1) - mat.min(axis=1)
            rep_rows.append({
                "group": name, "parameter": par, "k": k, "n_seeds": mat.shape[0],
                "sw": sw, "sd_median": float(sd_i.median()), "cv_median_pct": float(cv_i.median()),
                "range_median": float(rng_i.median()), "range_max": float(rng_i.max()),
                "icc": icc, "icc_ci_low": lo, "icc_ci_high": hi,
                "sem": sw, "rc": 2.77 * sw,
            })
    add_rep("все 8 снимков", PHOTOS)
    add_rep("P-снимки (3)", P_PHOTOS)
    add_rep("non-P снимки (5)", NONP_PHOTOS)
    add_rep("заявленный лучший фокус (3)", BEST_FOCUS)
    add_rep("остальные снимки (5)", [p for p in PHOTOS if p not in BEST_FOCUS])
    rep_df = pd.DataFrame(rep_rows)
    rep_df.to_csv(TABLES / "repeatability.csv", index=False, encoding="utf-8-sig")
    say("\nREPEATABILITY")
    for _, r in rep_df.iterrows():
        say(f"{r['group']:<32} {r['parameter']:>6} k={int(r['k'])} n={int(r['n_seeds'])} "
            f"Sw={r['sw']:.3f} ICC={r['icc']:.3f} [{r['icc_ci_low']:.3f},{r['icc_ci_high']:.3f}] RC={r['rc']:.3f}")

    # --- paired orientation and focus ---
    pair_rows = []
    def paired(name, a_photo, b_photo, par, col):
        a = long[long["photo"] == a_photo].set_index("serial")[col]
        b = long[long["photo"] == b_photo].set_index("serial")[col]
        mm = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
        d = mm["b"] - mm["a"]  # b - a
        n = len(d)
        se = d.std(ddof=1) / math.sqrt(n)
        tcrit = stats.t.ppf(0.975, n - 1)
        ttest = stats.ttest_rel(mm["b"], mm["a"])
        try:
            wil = stats.wilcoxon(mm["b"], mm["a"])
            pwil = float(wil.pvalue)
        except Exception:
            pwil = np.nan
        pair_rows.append({
            "comparison": name, "parameter": par, "n": n,
            "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
            "ci_low": float(d.mean() - tcrit * se), "ci_high": float(d.mean() + tcrit * se),
            "p_paired_t": float(ttest.pvalue), "p_wilcoxon": pwil,
            "r_between": float(np.corrcoef(mm["a"], mm["b"])[0, 1]),
        })
    def paired_means(name, a_photos, b_photos, par, col):
        a = long[long["photo"].isin(a_photos)].groupby("serial")[col].mean()
        b = long[long["photo"].isin(b_photos)].groupby("serial")[col].mean()
        mm = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
        d = mm["b"] - mm["a"]
        n = len(d)
        se = d.std(ddof=1) / math.sqrt(n)
        tcrit = stats.t.ppf(0.975, n - 1)
        pair_rows.append({
            "comparison": name, "parameter": par, "n": n,
            "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
            "ci_low": float(d.mean() - tcrit * se), "ci_high": float(d.mean() + tcrit * se),
            "p_paired_t": float(stats.ttest_rel(mm["b"], mm["a"]).pvalue),
            "p_wilcoxon": float(stats.wilcoxon(mm["b"], mm["a"]).pvalue),
            "r_between": float(np.corrcoef(mm["a"], mm["b"])[0, 1]),
        })

    paired(f"{PAIR_21[1]} − {PAIR_21[0]}", PAIR_21[0], PAIR_21[1], "length", "length_mm")
    paired(f"{PAIR_21[1]} − {PAIR_21[0]}", PAIR_21[0], PAIR_21[1], "width", "width_mm")
    paired(f"{PAIR_22[1]} − {PAIR_22[0]}", PAIR_22[0], PAIR_22[1], "length", "length_mm")
    paired(f"{PAIR_22[1]} − {PAIR_22[0]}", PAIR_22[0], PAIR_22[1], "width", "width_mm")
    # P mean vs non-P mean
    for par, col in [("length", "length_mm"), ("width", "width_mm")]:
        pv = long[long["photo"].isin(P_PHOTOS)].groupby("serial")[col].mean()
        nv = long[long["photo"].isin(NONP_PHOTOS)].groupby("serial")[col].mean()
        mm = pd.concat([nv, pv], axis=1, keys=["a", "b"]).dropna()
        d = mm["b"] - mm["a"]
        se = d.std(ddof=1) / math.sqrt(len(d))
        tcrit = stats.t.ppf(0.975, len(d) - 1)
        pair_rows.append({
            "comparison": "P-среднее − non-P-среднее", "parameter": par, "n": len(d),
            "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
            "ci_low": float(d.mean() - tcrit * se), "ci_high": float(d.mean() + tcrit * se),
            "p_paired_t": float(stats.ttest_rel(mm["b"], mm["a"]).pvalue),
            "p_wilcoxon": float(stats.wilcoxon(mm["b"], mm["a"]).pvalue),
            "r_between": float(np.corrcoef(mm["a"], mm["b"])[0, 1]),
        })
    # best focus mean vs rest, paired by seed
    for par, col in [("length", "length_mm"), ("width", "width_mm")]:
        bv = long[long["photo"].isin(BEST_FOCUS)].groupby("serial")[col].mean()
        rv = long[~long["photo"].isin(BEST_FOCUS)].groupby("serial")[col].mean()
        mm = pd.concat([rv, bv], axis=1, keys=["a", "b"]).dropna()
        d = mm["b"] - mm["a"]
        se = d.std(ddof=1) / math.sqrt(len(d))
        tcrit = stats.t.ppf(0.975, len(d) - 1)
        pair_rows.append({
            "comparison": "лучший фокус − остальные", "parameter": par, "n": len(d),
            "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
            "ci_low": float(d.mean() - tcrit * se), "ci_high": float(d.mean() + tcrit * se),
            "p_paired_t": float(stats.ttest_rel(mm["b"], mm["a"]).pvalue),
            "p_wilcoxon": float(stats.wilcoxon(mm["b"], mm["a"]).pvalue),
            "r_between": float(np.corrcoef(mm["a"], mm["b"])[0, 1]),
        })
    # focus claim within the same orientation (avoid P/non-P confounding)
    for par, col in [("length", "length_mm"), ("width", "width_mm")]:
        paired_means("non-P: 1_2+2_1 − 1_1+1_3+2_2", ["1_1", "1_3", "2_2"],
                     ["1_2", "2_1"], par, col)
        paired_means("P: 2_1p − 2_2p+2_3p", ["2_2p", "2_3p"],
                     ["2_1p"], par, col)
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(TABLES / "paired_comparisons.csv", index=False, encoding="utf-8-sig")
    say("\nPAIRED COMPARISONS")
    for _, r in pair_df.iterrows():
        say(f"{r['comparison']:<32} {r['parameter']:>6} n={int(r['n'])} "
            f"diff={r['mean_diff']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}] "
            f"p_t={r['p_paired_t']:.2e} p_w={r['p_wilcoxon']:.2e} r={r['r_between']:.3f}")

    # --- status effect (within-seed deviations) ---
    status_rows = []
    for par, col in [("length", "length_mm"), ("width", "width_mm")]:
        long[f"dev_{par}"] = long[col] - long.groupby("serial")[col].transform("mean")
        for st in ["focused", "transition", "blurry"]:
            sub = long[long["status"] == st]
            status_rows.append({
                "parameter": par, "status": st, "n_obs": len(sub),
                "n_seeds": sub["serial"].nunique(),
                "mean_dev": float(sub[f"dev_{par}"].mean()),
                "sd_dev": float(sub[f"dev_{par}"].std(ddof=1)),
            })
    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(TABLES / "status_effect.csv", index=False, encoding="utf-8-sig")
    say("\nSTATUS EFFECT (within-seed deviation, mm)")
    for _, r in status_df.iterrows():
        say(f"{r['parameter']:>6} {r['status']:<10} n_obs={int(r['n_obs'])} "
            f"n_seeds={int(r['n_seeds'])} mean={r['mean_dev']:+.3f} sd={r['sd_dev']:.3f}")
    for par in ["length", "width"]:
        groups = [long.loc[long["status"] == st, f"dev_{par}"].dropna().to_numpy()
                  for st in ["focused", "transition", "blurry"]]
        h, p = stats.kruskal(*groups)
        say(f"Kruskal-Wallis {par}: H={h:.2f} p={p:.3g}")

    # --- by row ---
    row_rows = []
    for row, sub in matched.groupby("row"):
        row_rows.append({
            "row": int(row), "n": len(sub),
            "length_bias": float(sub["length_bias_mm"].mean()),
            "length_sd": float(sub["length_bias_mm"].std(ddof=1)),
            "length_mae": float(sub["length_abs_err_mm"].mean()),
            "width_bias": float(sub["width_bias_mm"].mean()),
            "width_sd": float(sub["width_bias_mm"].std(ddof=1)),
            "width_mae": float(sub["width_abs_err_mm"].mean()),
        })
    row_df = pd.DataFrame(row_rows)
    row_df.to_csv(TABLES / "by_row.csv", index=False, encoding="utf-8-sig")
    say("\nBY ROW")
    for _, r in row_df.iterrows():
        say(f"row {int(r['row']):>2} n={int(r['n'])} L bias={r['length_bias']:+.3f} "
            f"W bias={r['width_bias']:+.3f}")

    # --- grid maps ---
    for par in ["length", "width"]:
        grid = wide.pivot(index="row", columns="pos", values=f"{par}_bias_mm").reindex(
            index=range(1, 14), columns=range(1, 26))
        grid.to_csv(TABLES / f"grid_{par}_bias.csv", encoding="utf-8-sig")

    # --- outliers ---
    out_rows = []
    for par in ["length", "width"]:
        sub = matched.copy()
        zcol = f"{par}_robust_z"
        bcol = f"{par}_bias_mm"
        flagged = sub[sub[zcol].abs() > 3.5].sort_values(zcol, key=lambda s: s.abs(), ascending=False)
        for rank, (_, r) in enumerate(flagged.head(12).iterrows(), 1):
            out_rows.append({
                "parameter": par, "rank": rank, "serial": int(r["serial"]),
                "row": int(r["row"]), "pos": int(r["pos"]),
                "manual": float(r[f"manual_{par}_mm"]),
                "program": float(r[f"prog_{par}_mean_mm"]),
                "bias": float(r[bcol]), "robust_z": float(r[zcol]),
                "n_photos": int(r["n_photos"]),
            })
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(TABLES / "outliers.csv", index=False, encoding="utf-8-sig")
    say("\nOUTLIERS (robust |z|>3.5)")
    for par in ["length", "width"]:
        n = int((matched[f"{par}_robust_z"].abs() > 3.5).sum())
        say(f"{par}: {n} seeds")
    say(f"overlap length & width outliers: "
        f"{int((matched['flag_length_outlier'] & matched['flag_width_outlier']).sum())}")

    # --- manual reference summary ---
    man_rows = []
    for col, label in [("length_mm", "длина"), ("width_mm", "ширина"),
                       ("thickness_mm", "толщина"), ("mass_g", "масса")]:
        v = manual[col].dropna().to_numpy(dtype=float)
        man_rows.append({
            "parameter": label, "n": len(v), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)), "median": float(np.median(v)),
            "min": float(v.min()), "max": float(v.max()), "unique": int(len(np.unique(v))),
            "grid_0.05_pct": float(100.0 * np.mean(np.abs(v / 0.05 - np.round(v / 0.05)) < 1e-6)),
            "grid_0.1_pct": float(100.0 * np.mean(np.abs(v / 0.1 - np.round(v / 0.1)) < 1e-6)),
        })
    man_df = pd.DataFrame(man_rows)
    man_df.to_csv(TABLES / "manual_summary.csv", index=False, encoding="utf-8-sig")
    say("\nMANUAL SUMMARY")
    for _, r in man_df.iterrows():
        say(f"{r['parameter']:<8} n={int(r['n'])} mean={r['mean']:.4f} sd={r['sd']:.4f} "
            f"median={r['median']:.4f} min={r['min']:.4f} max={r['max']:.4f} "
            f"unique={int(r['unique'])} grid0.05={r['grid_0.05_pct']:.1f}% grid0.1={r['grid_0.1_pct']:.1f}%")
    say(f"TGW from manual mass: {manual['mass_g'].mean() * 1000:.2f} g")

    # mass outliers
    mass_z = robust_z(manual["mass_g"])
    mass_out = manual.loc[np.abs(mass_z) > 3.5, ["serial", "row", "pos", "mass_g"]].copy()
    mass_out["robust_z"] = mass_z[np.abs(mass_z) > 3.5]
    mass_out.to_csv(TABLES / "manual_mass_outliers.csv", index=False, encoding="utf-8-sig")
    say("manual mass outliers:")
    for _, r in mass_out.iterrows():
        say(f"  serial {int(r['serial'])} row {int(r['row'])} pos {int(r['pos'])} "
            f"mass={r['mass_g']:.5f} z={r['robust_z']:.1f}")
    clean_mass = manual.loc[~manual["serial"].isin(mass_out["serial"]), "mass_g"].dropna()
    say(f"manual mass without outliers: n={len(clean_mass)} mean={clean_mass.mean():.5f} g "
        f"sd={clean_mass.std(ddof=1):.5f} median={clean_mass.median():.5f} "
        f"TGW={clean_mass.mean()*1000:.2f} g")

    # --- groove ---
    gr_rows = []
    for ph in PHOTOS:
        sub = long[long["photo"] == ph]
        v = pd.to_numeric(sub["has_groove"], errors="coerce")
        n = int(v.notna().sum()); k = int((v == 1).sum())
        lo, hi = wilson_ci(k, n)
        gr_rows.append({
            "photo": ph, "orientation": ORIENTATION[ph], "n": n,
            "positive": k, "rate": k / n if n else np.nan,
            "wilson_low": lo, "wilson_high": hi,
        })
    gr_df = pd.DataFrame(gr_rows)
    gr_df.to_csv(TABLES / "groove_by_photo.csv", index=False, encoding="utf-8-sig")
    say("\nGROOVE BY PHOTO")
    for _, r in gr_df.iterrows():
        say(f"{r['photo']:>5} {r['orientation']:>5} {int(r['positive'])}/{int(r['n'])} "
            f"= {100*r['rate']:.1f}% [{100*r['wilson_low']:.1f},{100*r['wilson_high']:.1f}]")
    for name, phs in [("P all", P_PHOTOS), ("non-P all", NONP_PHOTOS),
                      ("P without 2_1/2_1p", ["2_2p", "2_3p"]),
                      ("non-P without 2_1/2_1p", ["1_1", "1_2", "1_3", "2_2"])]:
        sub = long[long["photo"].isin(phs)]
        v = pd.to_numeric(sub["has_groove"], errors="coerce")
        k = int((v == 1).sum()); n = int(v.notna().sum()); lo, hi = wilson_ci(k, n)
        say(f"{name:<26} {k}/{n} = {100*k/n:.1f}% [{100*lo:.1f},{100*hi:.1f}]")

    # --- calibration ---
    cal = pd.read_csv(RESULTS / "ruler_scale_check.csv")
    cal["rel_diff_pct"] = 100.0 * (cal["mm_per_px_program"] - cal["mm_per_px_ruler"]) / cal["mm_per_px_ruler"]
    cal.to_csv(TABLES / "calibration.csv", index=False, encoding="utf-8-sig")
    say("\nCALIBRATION")
    say(f"mean rel diff = {cal['rel_diff_pct'].mean():+.3f}% "
        f"sd = {cal['rel_diff_pct'].std(ddof=1):.3f}% "
        f"min = {cal['rel_diff_pct'].min():+.3f}% max = {cal['rel_diff_pct'].max():+.3f}%")
    tt = stats.ttest_1samp(cal["rel_diff_pct"], 0.0)
    say(f"t-test vs 0: t={tt.statistic:.2f} p={tt.pvalue:.3f}")

    # --- segmentation completeness from saved coordinates ---
    coords = pd.read_csv(RESULTS / "coords_from_contours.csv")
    seg_rows = []
    coords_out = []
    for ph, sub in coords.groupby("photo"):
        sub = sub.sort_values("cy").copy()
        med_h = float(sub["bh"].median())
        chosen = None
        for ratio in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]:
            gaps = sub["cy"].diff()
            rid = (gaps > ratio * med_h).cumsum()
            if rid.nunique() == 13:
                chosen = rid
                break
        if chosen is None:
            chosen = (sub["cy"].diff() > 0.55 * med_h).cumsum()
        sub["row_derived"] = chosen.astype(int) + 1
        sub["pos_derived"] = sub.groupby("row_derived")["cx"].rank(method="first").astype(int)
        sub["serial_derived"] = (sub["row_derived"] - 1) * 25 + sub["pos_derived"]
        sub["mapping_ok"] = sub["serial_derived"] == sub["index"]
        # nearest neighbour distance
        pts = sub[["cx", "cy"]].to_numpy()
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(d, np.inf)
        nn = d.min(axis=1)
        sub["nn_px"] = nn
        coords_out.append(sub)
        rc = sub.groupby("row_derived").size()
        seg_rows.append({
            "photo": ph, "n": len(sub), "rows": int(sub["row_derived"].nunique()),
            "min_per_row": int(rc.min()), "max_per_row": int(rc.max()),
            "nn_min_px": float(nn.min()), "nn_median_px": float(np.median(nn)),
            "bw_min": int(sub["bw"].min()), "bw_max": int(sub["bw"].max()),
            "bh_min": int(sub["bh"].min()), "bh_max": int(sub["bh"].max()),
            "mapping_mismatches": int((~sub["mapping_ok"]).sum()),
        })
    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(TABLES / "segmentation.csv", index=False, encoding="utf-8-sig")
    pd.concat(coords_out, ignore_index=True).to_csv(RESULTS / "coords_with_grid.csv", index=False, encoding="utf-8-sig")
    say("\nSEGMENTATION")
    for _, r in seg_df.iterrows():
        say(f"{r['photo']:>5} n={int(r['n'])} rows={int(r['rows'])} "
            f"per_row={int(r['min_per_row'])}-{int(r['max_per_row'])} "
            f"NN={r['nn_min_px']:.1f}/{r['nn_median_px']:.1f}px "
            f"mapping_mismatch={int(r['mapping_mismatches'])}")

    # --- determinism ---
    first = pd.read_csv(RESULTS / "seeds_after_1_2_first.csv")
    first = first[first["file_index"].str.startswith("1_2_")].copy()
    final12 = seeds[seeds["photo"] == "1_2"].copy()
    det_rows = []
    if len(first) == len(final12):
        num_cols = ["edge_width_px", "length_px", "width_px", "angle_deg", "area_px", "mm_per_px"]
        maxdiff = {}
        for c in num_cols:
            a = pd.to_numeric(first[c], errors="coerce").to_numpy()
            b = pd.to_numeric(final12[c], errors="coerce").to_numpy()
            maxdiff[c] = float(np.nanmax(np.abs(a - b)))
        det_rows.append({
            "rows_first": len(first), "rows_final": len(final12),
            "max_abs_diff_numeric": max(maxdiff.values()),
            "status_differences": int((first["status"].to_numpy() != final12["status"].to_numpy()).sum()),
            "color_differences": int((first["color_hex"].to_numpy() != final12["color_hex"].to_numpy()).sum()),
            "groove_differences": int((pd.to_numeric(first["has_groove"], errors="coerce").to_numpy() !=
                                       pd.to_numeric(final12["has_groove"], errors="coerce").to_numpy()).sum()),
        })
    det_df = pd.DataFrame(det_rows)
    det_df.to_csv(TABLES / "determinism.csv", index=False, encoding="utf-8-sig")
    say("\nDETERMINISM")
    say(det_df.to_string(index=False))

    # --- run log ---
    run_rows = []
    for line in open(RESULTS / "batch_log.jsonl", encoding="utf-8"):
        d = json.loads(line)
        if d["label"] == "1_2_first":
            continue
        out = d.get("stdout", "")
        m = re.search(r"Обработка: .*?\((\d+)x(\d+)\)", out)
        found = re.search(r"Найдено объектов: (\d+)", out)
        saved = re.search(r"Сохранено: (\d+)", out)
        foc = re.search(r"фокус: (\d+)", out)
        blur = re.search(r"размыто: (\d+)", out)
        calm = re.search(r"Калибровка: ([\d.]+) мм/px", out)
        run_rows.append({
            "label": d["label"], "file": Path(d["file"]).name,
            "width_px": int(m.group(1)) if m else np.nan,
            "height_px": int(m.group(2)) if m else np.nan,
            "found": int(found.group(1)) if found else np.nan,
            "saved": int(saved.group(1)) if saved else np.nan,
            "focused": int(foc.group(1)) if foc else np.nan,
            "blurry": int(blur.group(1)) if blur else np.nan,
            "mm_per_px": float(calm.group(1)) if calm else np.nan,
            "elapsed_s": d["elapsed_s"], "rc": d["returncode"],
        })
    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(TABLES / "run_log.csv", index=False, encoding="utf-8-sig")
    say("\nRUN LOG")
    say(run_df.to_string(index=False))

    # --- missing seed localization ---
    m11 = manual[(manual["row"] == 11) & manual["length_mm"].notna()].sort_values("pos")
    Lmap = dict(zip(wide["serial"], wide["prog_length_mean_mm"]))
    Wmap = dict(zip(wide["serial"], wide["prog_width_mean_mm"]))
    k_rows = []
    for k in range(1, 26):
        idx = [(11 - 1) * 25 + p for p in range(1, 26) if p != k]
        progL = np.array([Lmap[i] for i in idx], dtype=float)
        progW = np.array([Wmap[i] for i in idx], dtype=float)
        dL = progL - m11["length_mm"].to_numpy(dtype=float)
        dW = progW - m11["width_mm"].to_numpy(dtype=float)
        k_rows.append({
            "k": k, "mae_length": float(np.abs(dL).mean()),
            "r_length": float(np.corrcoef(progL, m11["length_mm"])[0, 1]),
            "mae_width": float(np.abs(dW).mean()),
            "r_width": float(np.corrcoef(progW, m11["width_mm"])[0, 1]),
        })
    k_df = pd.DataFrame(k_rows)
    k_df.to_csv(TABLES / "missing_k.csv", index=False, encoding="utf-8-sig")
    best_k = int(k_df.loc[k_df["mae_length"].idxmin(), "k"])
    say("\nMISSING-SEED k TEST (row 11)")
    say(f"best k by length MAE: {best_k}; MAE={k_df['mae_length'].min():.4f}; "
        f"k=25 MAE={k_df.loc[k_df['k'] == 25, 'mae_length'].iloc[0]:.4f} "
        f"r={k_df.loc[k_df['k'] == 25, 'r_length'].iloc[0]:.4f}")
    say(k_df[["k", "mae_length", "r_length"]].round(3).to_string(index=False))

    # shift test rows 12-13
    rows1213 = manual[manual["row"].isin([12, 13])].sort_values("serial")
    shift_rows = []
    for name, offset in [("прямое", 0),
                         ("программа = серийный − 1", -1),
                         ("программа = серийный + 1", 1)]:
        pairs = [(int(s), int(s) + offset) for s in rows1213["serial"]
                 if 1 <= int(s) + offset <= 325]
        prog = np.array([Lmap[b] for _, b in pairs], dtype=float)
        man = np.array([manual.loc[manual["serial"] == a, "length_mm"].iloc[0]
                        for a, _ in pairs], dtype=float)
        d = prog - man
        shift_rows.append({
            "mapping": name, "n": len(d), "r": float(np.corrcoef(prog, man)[0, 1]),
            "mae": float(np.abs(d).mean()), "bias": float(d.mean()),
        })
    shift_df = pd.DataFrame(shift_rows)
    shift_df.to_csv(TABLES / "shift_test.csv", index=False, encoding="utf-8-sig")
    say("\nSHIFT TEST (rows 12-13)")
    say(shift_df.to_string(index=False))

    # missing-seed details from coordinates
    cg = pd.read_csv(RESULTS / "coords_with_grid.csv")
    cg = cg[(cg["row_derived"] == 11) & (cg["pos_derived"] == 25)].copy()
    cg = cg.merge(wide[["serial", "prog_length_mean_mm", "prog_width_mean_mm"]], left_on="index", right_on="serial", how="left")
    cg[["photo", "index", "cx", "cy", "bw", "bh", "nn_px", "prog_length_mean_mm", "prog_width_mean_mm"]].to_csv(
        TABLES / "missing_seed_details.csv", index=False, encoding="utf-8-sig")
    say("\nMISSING SEED 275 DETAILS")
    say(cg[["photo", "cx", "cy", "bw", "bh", "nn_px"]].to_string(index=False))

    # --- contour alternative geometry ---
    cm, alt = contour_metrics(seeds, manual)
    cm.to_csv(RESULTS / "contour_metrics.csv", index=False, encoding="utf-8-sig")
    alt.to_csv(RESULTS / "contour_metrics_wide.csv", index=False, encoding="utf-8-sig")
    alt_rows = []
    for name, pcol, mcol in [
        ("экспортируемая длина (макс. хорда)", "length_exported_mm", "manual_length_mm"),
        ("протяжённость вдоль оси хорды", "length_extent_mm", "manual_length_mm"),
        ("длинная сторона minAreaRect", "length_rect_mm", "manual_length_mm"),
        ("экспортируемая ширина", "width_exported_mm", "manual_width_mm"),
        ("площадь hull / длина", "width_hull_mm", "manual_width_mm"),
        ("короткая сторона minAreaRect", "width_rect_mm", "manual_width_mm"),
        ("протяжённость поперёк оси", "width_perp_mm", "manual_width_mm"),
    ]:
        a = agreement(alt[pcol], alt[mcol], boot=2000, label=name)
        alt_rows.append(a)
    alt_df = table_from_agreements(alt_rows)
    alt_df["definition"] = [r["label"] for r in alt_rows]
    alt_df.to_csv(TABLES / "alternative_definitions.csv", index=False, encoding="utf-8-sig")
    say("\nALTERNATIVE GEOMETRIC DEFINITIONS (per-seed mean over 8 photos)")
    for _, r in alt_df.iterrows():
        say(f"{r['definition']:<38} n={int(r['n'])} bias={r['bias']:+.3f} "
            f"sd={r['bias_sd']:.3f} MAE={r['mae']:.3f} r={r['r']:.3f} ratio={r['ratio_mean']:.4f}")
    say(f"branch hull wins: {100*seeds.merge(cm[['file_index','branch']], on='file_index')['branch'].eq('hull').mean():.1f}%")
    say(f"max |width_exported - min(hull,len,rect)| = {cm['width_match_err_px'].max():.6f} px")
    say(f"max |length_extent - length_exported| = {(cm['length_extent_px'] - cm['length_px']).abs().max():.6f} px")

    # --- save report ---
    (RESULTS / "analysis_report.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"\nWrote {RESULTS / 'analysis_report.txt'}")


if __name__ == "__main__":
    main()
