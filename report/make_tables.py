# -*- coding: utf-8 -*-
"""Generate LaTeX table fragments for the validation paper from the saved CSVs.

Only reads results/tables/*.csv and results/dataset_seeds_wide.csv.
Writes results/tables/tex_*.tex.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent
T = R / "tables"


def n(x, nd=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "---"
    return r"\num{" + f"{x:.{nd}f}" + "}"


def sn(x, nd=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "---"
    s = r"\num{" + f"{abs(x):.{nd}f}" + "}"
    return ("+" if x >= 0 else "-") + s


def ci(lo, hi, nd=2):
    return "[" + n(lo, nd) + "; " + n(hi, nd) + "]"


def pv(p):
    if p is None or not math.isfinite(p):
        return "---"
    if p < 1e-4:
        return r"$<10^{-4}$"
    if p < 0.001:
        return r"$<0{,}001$"
    return r"$=$\num{" + f"{p:.3f}" + "}"


def sci(p):
    """Scientific p-value with comma decimal, e.g. 5,4·10^-73."""
    if p is None or not math.isfinite(p):
        return "---"
    if p >= 1e-3:
        return r"\num{" + f"{p:.3f}" + "}"
    e = int(math.floor(math.log10(p)))
    m = p / 10 ** e
    return f"${m:.1f}".replace(".", "{,}") + rf"\cdot10^{{{e}}}$"


def ph(s):
    return s.replace("_", r"\_")


def write(name, text):
    (T / name).write_text(text, encoding="utf-8")
    print("wrote", name)


# ---------------------------------------------------------------------------
run = pd.read_csv(T / "run_log.csv")
run["label"] = run["label"].replace({"1_2_second": "1_2"})
run = run.set_index("label")

final = pd.read_csv(R / "seeds_final.csv")
final["photo"] = final["file_index"].str.rsplit("_", n=1).str[0]
status = (final.groupby(["photo", "status"]).size().unstack(fill_value=0)
          .reindex(columns=["focused", "transition", "blurry"], fill_value=0))

order = ["1_1", "1_2", "1_3", "2_1", "2_1p", "2_2", "2_2p", "2_3p"]
P = {"2_1p", "2_2p", "2_3p"}
best = {"1_2", "2_1", "2_1p"}

rows = []
for photo in order:
    r = run.loc[photo]
    st = status.loc[photo]
    rows.append(
        f"{ph(photo)} & {'P' if photo in P else 'non-P'} & "
        f"{'да' if photo in best else 'нет'} & "
        f"{int(r['width_px'])}x{int(r['height_px'])} & "
        f"{n(r['mm_per_px'], 6)} & {int(r['found'])}/{int(r['saved'])} & "
        f"{int(st['focused'])}/{int(st['transition'])}/{int(st['blurry'])} & "
        f"{n(r['elapsed_s'], 2)} \\\\"
    )
write("tex_photos.tex", r"""\begin{tabular}{@{}lllrrrrr@{}}
\toprule
Снимок & Ориентация & Фокус$^{a}$ & Размер, px & Масштаб, мм/px &
Найдено/сохранено & focused/transition/blurry & $t$, с \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------------------
ov = pd.read_csv(T / "overall_agreement_full.csv").set_index("label")
rows = []
for label, name in [("length", "Длина"), ("width", "Ширина")]:
    r = ov.loc[label]
    rows.append(
        f"{name} & {int(r['n'])} & "
        f"${n(r['manual_mean'], 3)} \\pm {n(r['manual_sd'], 3)}$ & "
        f"${n(r['prog_mean'], 3)} \\pm {n(r['prog_sd'], 3)}$ & "
        f"${sn(r['bias'], 3)}$ {ci(r['bias_ci_low'], r['bias_ci_high'], 3)} & "
        f"{n(r['mae'], 3)} {ci(r['mae_ci_low'], r['mae_ci_high'], 3)} & "
        f"{n(r['rmse'], 3)} & {n(r['r'], 3)} & {n(r['ccc'], 3)} \\\\"
    )
write("tex_overall.tex", r"""\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
Признак & $n$ & Вручную, мм & Программа, мм &
Смещение [95\% ДИ], мм & MAE [95\% ДИ], мм & RMSE, мм & $r$ & CCC \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------------------
pp = pd.read_csv(T / "per_photo_agreement.csv")
rows = []
for photo in order:
    rl = pp[(pp["photo"] == photo) & (pp["parameter"] == "length")].iloc[0]
    rw = pp[(pp["photo"] == photo) & (pp["parameter"] == "width")].iloc[0]
    rows.append(
        f"{ph(photo)} & {'P' if photo in P else 'non-P'} & "
        f"{'да' if photo in best else 'нет'} & "
        f"${sn(rl['bias'], 3)}$ & {n(rl['mae'], 3)} & {n(rl['r'], 3)} & "
        f"${sn(rw['bias'], 3)}$ & {n(rw['mae'], 3)} & {n(rw['r'], 3)} \\\\"
    )
write("tex_per_photo.tex", r"""\begin{tabular}{@{}lllrrrrrr@{}}
\toprule
Снимок & Ориентация & Фокус$^{a}$ & $\Delta L$, мм & MAE$_L$, мм & $r_L$ &
$\Delta W$, мм & MAE$_W$, мм & $r_W$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------------------
rep = pd.read_csv(T / "repeatability.csv")
rows = []
for grp in ["все 8 снимков", "P-снимки (3)", "non-P снимки (5)",
            "заявленный лучший фокус (3)", "остальные снимки (5)"]:
    rl = rep[(rep["group"] == grp) & (rep["parameter"] == "length")].iloc[0]
    rw = rep[(rep["group"] == grp) & (rep["parameter"] == "width")].iloc[0]
    rows.append(
        f"{grp} & {int(rl['k'])} & {n(rl['sw'], 3)} & "
        f"{n(rl['icc'], 3)} {ci(rl['icc_ci_low'], rl['icc_ci_high'], 3)} & "
        f"{n(rl['rc'], 3)} & {n(rw['sw'], 3)} & "
        f"{n(rw['icc'], 3)} {ci(rw['icc_ci_low'], rw['icc_ci_high'], 3)} & "
        f"{n(rw['rc'], 3)} \\\\"
    )
write("tex_repeatability.tex", r"""\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Группа & $k$ & $S_w^{L}$, мм & ICC$_L$ [95\% ДИ] & RC$_L$, мм &
$S_w^{W}$, мм & ICC$_W$ [95\% ДИ] & RC$_W$, мм \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------------------
pc = pd.read_csv(T / "paired_comparisons.csv")
rows = []
for comp in ["2_1p − 2_1", "2_2p − 2_2", "P-среднее − non-P-среднее"]:
    for par, name in [("length", "длина"), ("width", "ширина")]:
        r = pc[(pc["comparison"] == comp) & (pc["parameter"] == par)].iloc[0]
        rows.append(
            f"{comp.replace(' − ', ' $-$ ').replace('-', '--')} & {name} & "
            f"${sn(r['mean_diff'], 3)}$ {ci(r['ci_low'], r['ci_high'], 3)} & "
            f"{sci(r['p_paired_t'])} & {n(r['r_between'], 3)} \\\\"
        )
write("tex_paired.tex", r"""\begin{tabular}{@{}llrrr@{}}
\toprule
Сравнение & Параметр & $\Delta$ [95\% ДИ], мм & $p$ (парный $t$) & $r$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------------------
st = pd.read_csv(T / "status_effect.csv")
rows = []
for par, name in [("length", "Длина"), ("width", "Ширина")]:
    for s in ["focused", "transition", "blurry"]:
        r = st[(st["parameter"] == par) & (st["status"] == s)].iloc[0]
        rows.append(
            f"{name} & {s} & {int(r['n_obs'])} & {int(r['n_seeds'])} & "
            f"${sn(r['mean_dev'], 3)}$ & {n(r['sd_dev'], 3)} \\\\"
        )
write("tex_status.tex", r"""\begin{tabular}{@{}llrrrr@{}}
\toprule
Признак & Статус & Наблюдений & Зёрен & Среднее отклонение, мм & СКО, мм \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------------------
alt = pd.read_csv(T / "alternative_definitions.csv").set_index("label")
keep = ["экспортируемая длина (макс. хорда)", "длинная сторона minAreaRect",
        "экспортируемая ширина", "короткая сторона minAreaRect",
        "протяжённость поперёк оси"]
rows = []
for label in keep:
    r = alt.loc[label]
    rows.append(
        f"{label} & ${sn(r['bias'], 3)}$ {ci(r['bias_ci_low'], r['bias_ci_high'], 3)} & "
        f"{n(r['mae'], 3)} & {n(r['r'], 3)} & {n(r['ratio_mean'], 4)} \\\\"
    )
write("tex_alternatives.tex", r"""\begin{tabular}{@{}lrrrr@{}}
\toprule
Определение & Смещение [95\% ДИ], мм & MAE, мм & $r$ & Отношение \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------------------
gr = pd.read_csv(T / "groove_by_photo.csv").set_index("photo")
rows = []
for photo in order:
    r = gr.loc[photo]
    rows.append(
        f"{ph(photo)} & {r['orientation']} & {int(r['positive'])}/{int(r['n'])} & "
        f"{n(100 * r['rate'], 1)} & "
        f"[{n(100 * r['wilson_low'], 1)}; {n(100 * r['wilson_high'], 1)}] \\\\"
    )
write("tex_groove.tex", r"""\begin{tabular}{@{}llrrr@{}}
\toprule
Снимок & Ориентация & Положительно/всего & Доля, \% & 95\% ДИ, \% \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------------------
out = pd.read_csv(T / "outliers.csv")
rows = []
for _, r in out.iterrows():
    rows.append(
        f"{'Длина' if r['parameter'] == 'length' else 'Ширина'} & "
        f"{int(r['serial'])} & {int(r['row'])} & {int(r['pos'])} & "
        f"{n(r['manual'], 2)} & {n(r['program'], 2)} & "
        f"${sn(r['bias'], 2)}$ & {n(r['robust_z'], 2)} \\\\"
    )
write("tex_outliers.tex", r"""\begin{tabular}{@{}llrrrrrr@{}}
\toprule
Признак & № зерна & Ряд & Поз. & Вручную, мм & Программа, мм &
Смещение, мм & $z$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------------------
sg = pd.read_csv(T / "subgroup_agreement.csv")
sgl = sg.iloc[:7].reset_index(drop=True)
sgw = sg.iloc[7:].reset_index(drop=True)
groups = ["Все зёрна", "Без ряда 11", "Только ряд 11",
          "P-снимки", "non-P снимки", "Заявленный лучший фокус",
          "Остальные снимки"]
rows = []
for i, name in enumerate(groups):
    rl = sgl.iloc[i]
    rw = sgw.iloc[i]
    rows.append(
        f"{name} & {int(rl['n'])} & ${sn(rl['bias'], 3)}$ & {n(rl['mae'], 3)} & "
        f"{n(rl['r'], 3)} & ${sn(rw['bias'], 3)}$ & {n(rw['mae'], 3)} & "
        f"{n(rw['r'], 3)} \\\\"
    )
write("tex_subgroups.tex", r"""\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Подгруппа & $n$ & $\Delta L$, мм & MAE$_L$, мм & $r_L$ &
$\Delta W$, мм & MAE$_W$, мм & $r_W$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------------------
cal = pd.read_csv(T / "calibration.csv").set_index("photo")
rows = []
for photo in order:
    r = cal.loc[photo]
    rows.append(
        f"{ph(photo)} & {n(r['mm_per_px_program'], 6)} & "
        f"{n(r['mm_per_px_ruler'], 6)} & ${sn(r['rel_diff_pct'], 3)}$ \\\\"
    )
write("tex_calibration.tex", r"""\begin{tabular}{@{}lrrr@{}}
\toprule
Снимок & Программа, мм/px & Независимая оценка, мм/px & Отличие, \% \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------------------
man = pd.read_csv(T / "manual_summary.csv")
rows = []
for _, r in man.iterrows():
    unit = "г" if r["parameter"] == "масса" else "мм"
    rows.append(
        f"{r['parameter'].capitalize()} & {int(r['n'])} & "
        f"${n(r['mean'], 3)} \\pm {n(r['sd'], 3)}$ & {n(r['median'], 3)} & "
        f"{n(r['min'], 3)}--{n(r['max'], 3)} & {int(r['unique'])} & "
        f"{n(r['grid_0.05_pct'], 1)} & {n(r['grid_0.1_pct'], 1)} \\\\"
    )
write("tex_manual_summary.tex", r"""\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Признак & $n$ & Среднее $\pm$ СКО & Медиана & Мин--макс &
Уникальных & На сетке 0,05, \% & На сетке 0,1, \% \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------------------
byrow = pd.read_csv(T / "by_row.csv")
rows = []
for _, r in byrow.iterrows():
    rows.append(
        f"{int(r['row'])} & {int(r['n'])} & ${sn(r['length_bias'], 3)}$ & "
        f"${sn(r['width_bias'], 3)}$ \\\\"
    )
write("tex_by_row.tex", r"""\begin{tabular}{@{}rrrr@{}}
\toprule
Ряд & $n$ & $\Delta L$, мм & $\Delta W$, мм \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------------------
seg = pd.read_csv(T / "segmentation.csv").set_index("photo")
rows = []
for photo in order:
    r = seg.loc[photo]
    rows.append(
        f"{ph(photo)} & {int(r['n'])} & {int(r['rows'])} & "
        f"{int(r['min_per_row'])}--{int(r['max_per_row'])} & "
        f"{n(r['nn_min_px'], 1)} & {n(r['nn_median_px'], 1)} & "
        f"{int(r['bw_min'])}--{int(r['bw_max'])} & "
        f"{int(r['bh_min'])}--{int(r['bh_max'])} & {int(r['mapping_mismatches'])} \\\\"
    )
write("tex_segmentation.tex", r"""\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
Снимок & $n$ & Рядов & В ряду & NN$_{\min}$, px & NN$_{\text{med}}$, px &
$b_w$, px & $b_h$, px & Ошибок сопоставления \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------------------
det = pd.read_csv(T / "determinism.csv").iloc[0]
write("tex_determinism.tex", r"""\begin{tabular}{@{}rrrrrr@{}}
\toprule
Строк в 1-м прогоне & Строк во 2-м & $\max|\Delta|$ числовой &
Различий status & Различий color & Различий groove \\
\midrule
""" + f"{int(det['rows_first'])} & {int(det['rows_final'])} & {n(det['max_abs_diff_numeric'], 1)} & "
f"{int(det['status_differences'])} & {int(det['color_differences'])} & {int(det['groove_differences'])} \\\\" +
r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------------------
# Missing-seed details
ms = pd.read_csv(T / "missing_seed_details.csv")
rows = []
for _, r in ms.iterrows():
    rows.append(
        f"{ph(r['photo'])} & {int(r['index'])} & {n(r['cx'], 1)} & {n(r['cy'], 1)} & "
        f"{int(r['bw'])}x{int(r['bh'])} & {n(r['nn_px'], 1)} \\\\"
    )
write("tex_missing_seed.tex", r"""\begin{tabular}{@{}lrrrrr@{}}
\toprule
Снимок & Индекс & $c_x$, px & $c_y$, px & $b_w$ x $b_h$, px & NN, px \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------------------
mk = pd.read_csv(T / "missing_k.csv")
rows = []
for _, r in mk.iterrows():
    rows.append(f"{int(r['k'])} & {n(r['mae_length'], 3)} & {n(r['r_length'], 3)} \\\\")
write("tex_missing_k.tex", r"""\begin{tabular}{@{}rrr@{}}
\toprule
$k$ & MAE длины, мм & $r$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------------------
# Full per-seed appendix table.
wide = pd.read_csv(R / "dataset_seeds_wide.csv").sort_values("serial")
rows = []
for _, r in wide.iterrows():
    miss = bool(r["missing_manual"])
    flags = []
    if bool(r["flag_length_outlier"]):
        flags.append("L")
    if bool(r["flag_width_outlier"]):
        flags.append("W")
    note = ", ".join(flags)
    if miss:
        note = "нет ручного замера"
    lm = "---" if miss else n(r["manual_length_mm"], 2)
    wm = "---" if miss else n(r["manual_width_mm"], 2)
    tm = "---" if miss else n(r["manual_thickness_mm"], 2)
    mm = "---" if miss else n(r["manual_mass_g"], 5)
    dl = "---" if miss else "$" + sn(r["length_bias_mm"], 2) + "$"
    dw = "---" if miss else "$" + sn(r["width_bias_mm"], 2) + "$"
    rows.append(
        f"{int(r['serial'])} & {int(r['row'])} & {int(r['pos'])} & "
        f"{lm} & {wm} & {tm} & {mm} & "
        f"{n(r['prog_length_mean_mm'], 2)} ({n(r['prog_length_sd_mm'], 2)}) & "
        f"{n(r['prog_width_mean_mm'], 2)} ({n(r['prog_width_sd_mm'], 2)}) & "
        f"{dl} & {dw} & {note} \\\\"
    )

head = r"""\begin{longtable}{@{}rrrrrrrrrrrr@{}}
\caption{Полная таблица сопоставления по зёрнам (325 строк; программа — среднее
и внутризерновое СКО по 8 снимкам).}\label{tab:full}\\
\toprule
№ & Ряд & Поз. & $L_{\text{руч}}$ & $W_{\text{руч}}$ & $T_{\text{руч}}$ &
$m_{\text{руч}}$, г & $L_{\text{прог}}$ (СКО) & $W_{\text{прог}}$ (СКО) &
$\Delta L$ & $\Delta W$ & Примечание \\
\midrule
\endfirsthead
\toprule
№ & Ряд & Поз. & $L_{\text{руч}}$ & $W_{\text{руч}}$ & $T_{\text{руч}}$ &
$m_{\text{руч}}$, г & $L_{\text{прог}}$ (СКО) & $W_{\text{прог}}$ (СКО) &
$\Delta L$ & $\Delta W$ & Примечание \\
\midrule
\endhead
\bottomrule
\endfoot
"""
write("tex_full.tex", head + "\n".join(rows) + "\n\\end{longtable}\n")

# ---------------------------------------------------------------------------
# Run log (appendix)
import json
log = [json.loads(line) for line in (R / "batch_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
log_by_label = {r["label"]: r for r in log}
rows = []
for label in ["1_1", "1_2_second", "1_3", "2_1", "2_1p", "2_2", "2_2p", "2_3p"]:
    r = log_by_label[label]
    photo = "1_2" if label == "1_2_second" else label
    rows.append(
        f"{ph(photo)} & \\texttt{{{ph(r['file'].replace(chr(92), '/'))}}} & "
        f"{r['start'][11:]} & {n(r['elapsed_s'], 2)} & {int(r['returncode'])} & "
        f"325/325 \\\\"
    )
write("tex_runlog.tex", r"""\begin{tabular}{@{}llrrrr@{}}
\toprule
Снимок & Файл & Начало & $t$, с & RC & Найдено/сохранено \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------------------
mo = pd.read_csv(T / "manual_mass_outliers.csv")
rows = []
for _, r in mo.iterrows():
    rows.append(
        f"{int(r['serial'])} & {int(r['row'])} & {int(r['pos'])} & "
        f"{n(r['mass_g'], 5)} & {n(r['robust_z'], 1)} \\\\"
    )
write("tex_mass_outliers.tex", r"""\begin{tabular}{@{}rrrrr@{}}
\toprule
№ зерна & Ряд & Поз. & Масса, г & $z$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
""")

# ---------------------------------------------------------------------------
# Numbers file with macros for key prose values.
ovl = ov.loc["length"]
ovw = ov.loc["width"]
rep8l = rep[(rep["group"] == "все 8 снимков") & (rep["parameter"] == "length")].iloc[0]
rep8w = rep[(rep["group"] == "все 8 снимков") & (rep["parameter"] == "width")].iloc[0]
ratio = pd.read_csv(T / "width_ratio.csv").iloc[0]
cal = pd.read_csv(T / "calibration.csv")
macros = []
def mac(name, val):
    macros.append("\\newcommand{\\" + name + "}{" + val + "}")
mac("LenBias", f"${sn(ovl['bias'], 3)}$")
mac("LenBiasLo", n(ovl["bias_ci_low"], 3))
mac("LenBiasHi", n(ovl["bias_ci_high"], 3))
mac("LenMae", n(ovl["mae"], 3))
mac("LenRmse", n(ovl["rmse"], 3))
mac("LenR", n(ovl["r"], 3))
mac("LenCcc", n(ovl["ccc"], 3))
mac("LenSlope", n(ovl["slope"], 3))
mac("LenIntercept", sn(ovl["intercept"], 3))
mac("LenLoaLo", sn(ovl["loa_low"], 3))
mac("LenLoaHi", sn(ovl["loa_high"], 3))
mac("LenRatio", n(ovl["ratio_mean"], 4))
mac("LenBiasPct", n(ovl["bias_pct"], 1))
mac("WidBias", f"${sn(ovw['bias'], 3)}$")
mac("WidBiasLo", n(ovw["bias_ci_low"], 3))
mac("WidBiasHi", n(ovw["bias_ci_high"], 3))
mac("WidMae", n(ovw["mae"], 3))
mac("WidRmse", n(ovw["rmse"], 3))
mac("WidR", n(ovw["r"], 3))
mac("WidCcc", n(ovw["ccc"], 3))
mac("WidSlope", n(ovw["slope"], 3))
mac("WidIntercept", sn(ovw["intercept"], 3))
mac("WidLoaLo", sn(ovw["loa_low"], 3))
mac("WidLoaHi", sn(ovw["loa_high"], 3))
mac("WidRatio", n(ovw["ratio_mean"], 4))
mac("WidBiasPct", n(abs(ovw["bias_pct"]), 1))
mac("RatioPi", n(ratio["p_vs_pi4"], 3))
mac("RatioMean", n(ratio["mean"], 4))
mac("RatioLo", n(ratio["ci_low"], 4))
mac("RatioHi", n(ratio["ci_high"], 4))
mac("PiPred", n(ratio["predicted_width_mean_pi4"], 4))
mac("PiActual", n(ratio["actual_width_mean"], 4))
mac("LenSw", n(rep8l["sw"], 3))
mac("LenIcc", n(rep8l["icc"], 3))
mac("LenIccLo", n(rep8l["icc_ci_low"], 3))
mac("LenIccHi", n(rep8l["icc_ci_high"], 3))
mac("LenRc", n(rep8l["rc"], 3))
mac("WidSw", n(rep8w["sw"], 3))
mac("WidIcc", n(rep8w["icc"], 3))
mac("WidIccLo", n(rep8w["icc_ci_low"], 3))
mac("WidIccHi", n(rep8w["icc_ci_high"], 3))
mac("WidRc", n(rep8w["rc"], 3))
mac("CalMean", n(cal["rel_diff_pct"].mean(), 3))
mac("CalSd", n(cal["rel_diff_pct"].std(ddof=1), 3))
mac("CalMin", n(cal["rel_diff_pct"].min(), 3))
mac("CalMax", n(cal["rel_diff_pct"].max(), 3))
mac("CalT", n(-0.81, 2))  # from report, t-test vs 0
mac("CalP", n(0.444, 3))
write("tex_numbers.tex", "\n".join(macros) + "\n")
print("done")
