"""
augment.py - аугментация нарезанного датасета семечек.

Кладётся рядом с utils.py. Запуск:
    python augment.py
    python augment.py out seeds_edited.csv out_aug

Что делает:
- берёт out/*.jpg + out/*_contour.json + CSV с разметкой (seeds_edited.csv, иначе seeds.csv);
- делит ФОТО (не семечки) на train/val, val не аугментирует;
- для каждой train-семечки делает N_AUG копий: поворот на 90/180/270 и отражения
  (без интерполяции и без заливки фона), яркость/контраст/гамма/баланс белого,
  градиент света, шум, JPEG;
- размер НЕ меняет, поэтому length/width/area/mm/thickness/mass/shape копируются как есть;
  пересчитывается только angle_deg, контур и размер кропа;
- пишет всё в out_aug/ в том же формате (jpg + _contour.json + seeds.csv),
  так что inspec.py открывает эту папку как обычно.
"""

import os
import re
import sys
import csv
import json
import math
import shutil
import random

import cv2
import numpy as np

from utils import imread_unicode, imwrite_unicode

# ---------- настройки ----------
SRC_DIR = "out"
CSV_NAME = "seeds_edited.csv"   # если файла нет, берётся seeds.csv
DST_DIR = "out_aug"

N_AUG = 7             # копий на семечку (вместе с оригиналом получится x8)
VAL_FRACTION = 0.2    # доля фото для val (они не аугментируются), 0 = выключить
SEED = 42
PAD_DEFAULT = 25      # отступ вокруг семечки, как в cut.py

# Геометрия: только повороты на 90 градусов и отражения (8 вариантов, оригинал пропускаем).
# Так фон остаётся настоящим, края не мылятся, размер в пикселях не меняется.
# Если N_AUG больше 7, геометрия повторяется, но фотометрия у копий всё равно разная.

# фотометрия (фон тёмный, поэтому диапазоны небольшие)
ALPHA_RANGE = (0.9, 1.1)      # контраст
BETA_RANGE = (-6, 6)          # яркость
GAMMA_RANGE = (0.9, 1.1)
WB_RANGE = 0.03               # сдвиг каждого канала +-3%
GRADIENT_MAX = 0.05           # неравномерность освещения
NOISE_MAX = 2.0               # макс. сигма гауссова шума
JPEG_QUALITY = (85, 95)

EXTRA_FIELDS = ["source", "split", "is_aug", "aug_of"]


# ---------- ввод/вывод ----------

def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except Exception:
            dialect = csv.excel

        reader = csv.DictReader(f, dialect=dialect)
        fields = list(reader.fieldnames or [])
        rows = []

        for r in reader:
            rows.append({k: (v or "") for k, v in r.items() if k is not None})

    return fields, rows


def load_contour(path):
    """Возвращает (points Nx2 float32, meta dict) или (None, None)."""
    if not os.path.isfile(path):
        return None, None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            data = {"points": data}

        pts = np.array(data.get("points", []), dtype=np.float32).reshape(-1, 2)
    except Exception:
        return None, None

    if len(pts) < 3:
        return None, None

    return pts, data


def source_of(file_index):
    """'2_1_001' -> '2_1' (имя исходного фото)."""
    return re.sub(r"_\d+$", "", file_index)


# ---------- геометрия ----------

# 8 симметрий прямоугольника: поворот на k*90 градусов + отражение.
# (0, False) - это сам оригинал, его пропускаем.
DIHEDRAL = [(k, f) for k in range(4) for f in (False, True) if (k, f) != (0, False)]


def geometric(img, pts, k90, flip):
    """
    Поворот на k90*90 градусов против часовой, затем (если flip) отражение слева-направо.
    Только операции над массивом: без интерполяции и без заливки углов.
    Возвращает новое изображение, новый контур и матрицу A (2x2) для пересчёта угла.
    """
    out = img
    p = pts.copy()
    A = np.eye(2)

    for _ in range(k90):
        w = out.shape[1]
        out = np.rot90(out)
        p = np.column_stack([p[:, 1], (w - 1) - p[:, 0]])
        A = np.array([[0.0, 1.0], [-1.0, 0.0]]) @ A

    if flip:
        w = out.shape[1]
        out = out[:, ::-1]
        p[:, 0] = (w - 1) - p[:, 0]
        A = np.diag([-1.0, 1.0]) @ A

    return np.ascontiguousarray(out), p, A


def transform_angle(angle_deg, A):
    """Угол длинной оси семечки после преобразования A (как в cut.py: 0..180)."""
    a = math.radians(angle_deg)
    d = A @ np.array([math.cos(a), math.sin(a)])
    return math.degrees(math.atan2(d[1], d[0])) % 180.0


# ---------- фотометрия ----------

def photometric(img):
    x = img.astype(np.float32)
    h, w = x.shape[:2]

    # контраст, яркость, баланс белого
    gain = 1.0 + np.random.uniform(-WB_RANGE, WB_RANGE, 3)
    alpha = random.uniform(*ALPHA_RANGE)
    beta = random.uniform(*BETA_RANGE)
    x = np.clip(x * alpha * gain + beta, 0, 255)

    # гамма
    g = random.uniform(*GAMMA_RANGE)
    x = 255.0 * (x / 255.0) ** g

    # линейный градиент освещения
    a = random.uniform(0, 2 * math.pi)
    s = random.uniform(0, GRADIENT_MAX)
    xs = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    ys = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    grad = 1.0 + s * (xs * math.cos(a) + ys * math.sin(a))
    x = x * grad[:, :, None]

    # шум
    sigma = random.uniform(0, NOISE_MAX)
    if sigma > 0.1:
        x = x + np.random.normal(0, sigma, x.shape)

    return np.clip(x, 0, 255).astype(np.uint8)


# ---------- основной код ----------

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else SRC_DIR
    csv_name = sys.argv[2] if len(sys.argv) > 2 else CSV_NAME
    dst = sys.argv[3] if len(sys.argv) > 3 else DST_DIR

    csv_path = os.path.join(src, csv_name)

    if not os.path.isfile(csv_path):
        csv_path = os.path.join(src, "seeds.csv")

    if not os.path.isfile(csv_path):
        print("CSV не найден в", src)
        return

    random.seed(SEED)
    np.random.seed(SEED)

    fields, rows = read_csv(csv_path)
    os.makedirs(dst, exist_ok=True)

    # делим по фото, чтобы семечки одного снимка не попали и в train, и в val
    sources = sorted({source_of(r["file_index"]) for r in rows})
    random.shuffle(sources)

    n_val = 0
    if VAL_FRACTION > 0 and len(sources) > 1:
        n_val = max(1, int(round(len(sources) * VAL_FRACTION)))

    val_sources = set(sources[:n_val])

    out_fields = list(fields)
    for name in EXTRA_FIELDS:
        if name not in out_fields:
            out_fields.append(name)

    out_rows = []
    n_orig = 0
    n_aug = 0
    n_skip = 0

    for row in rows:
        fi = row["file_index"]
        img_path = os.path.join(src, fi + ".jpg")
        json_path = os.path.join(src, fi + "_contour.json")

        img = imread_unicode(img_path) if os.path.isfile(img_path) else None
        pts, meta = load_contour(json_path)

        if img is None or pts is None:
            n_skip += 1
            continue

        src_name = source_of(fi)
        split = "val" if src_name in val_sources else "train"

        # оригинал копируем как есть
        shutil.copy2(img_path, os.path.join(dst, fi + ".jpg"))
        shutil.copy2(json_path, os.path.join(dst, fi + "_contour.json"))

        r0 = dict(row)
        r0.update({"source": src_name, "split": split, "is_aug": 0, "aug_of": ""})
        out_rows.append(r0)
        n_orig += 1

        if split == "val":
            continue

        pad = int(meta.get("padding", PAD_DEFAULT))

        angle_text = row.get("angle_deg", "").replace(",", ".").strip()
        try:
            angle_old = float(angle_text)
        except ValueError:
            angle_old = None

        ops = list(DIHEDRAL)
        random.shuffle(ops)

        for k in range(1, N_AUG + 1):
            k90, flip = ops[(k - 1) % len(ops)]

            im2, pts2, A = geometric(img, pts, k90, flip)
            im2 = photometric(im2)

            new_fi = f"{fi}_aug{k:02d}"
            q = random.randint(*JPEG_QUALITY)

            ok = imwrite_unicode(
                os.path.join(dst, new_fi + ".jpg"),
                im2,
                [int(cv2.IMWRITE_JPEG_QUALITY), q],
            )

            if not ok:
                continue

            h2, w2 = im2.shape[:2]

            payload = {
                "file_index": new_fi,
                "index": meta.get("index", row.get("index", "")),
                "points_space": "crop",
                "image_width": int(w2),
                "image_height": int(h2),
                "padding": pad,
                "source_name": src_name,
                "aug_of": fi,
                "aug": {"rot90": k90, "flip_h": flip},
                "points": pts2.astype(int).tolist(),
            }

            with open(os.path.join(dst, new_fi + "_contour.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)

            # Размерные поля (length/width/area/mm/thickness/mass/shape)
            # и фокус (edge_width_px/status) копируются без изменений.
            r1 = dict(row)
            r1["file_index"] = new_fi

            if angle_old is not None:
                r1["angle_deg"] = f"{transform_angle(angle_old, A):.3f}"

            r1.update({"source": src_name, "split": "train", "is_aug": 1, "aug_of": fi})
            out_rows.append(r1)
            n_aug += 1

    out_csv = os.path.join(dst, "seeds.csv")

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Оригиналов: {n_orig}, аугментаций: {n_aug}, пропущено: {n_skip}")
    print(f"Фото в val: {sorted(val_sources)}")
    print(f"Результат: {dst}")


if __name__ == "__main__":
    main()
