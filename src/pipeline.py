"""Main augmentation pipeline: load, split, augment, save."""
from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import sys
from typing import Any

import cv2
import numpy as np

from augmentation.geometric import (
    estimate_background_stats,
    rotate_any_angle,
    transform_angle,
)
from augmentation.photometric import JPEG_QUALITY, photometric
from core.contour_io import load_contour
from core.csv_utils import read_csv_robust, write_csv
from core.io_utils import imread_unicode, imwrite_unicode
from core.models import AUG_EXTRA_FIELDS

log = logging.getLogger(__name__)

IMG_EXT = ".jpg"
CONTOUR_SUFFIX = "_contour.json"
AUG_FMT = "aug{k:02d}"
PAD_DEFAULT = 25
SEED = 42


def source_of(file_index: str) -> str:
    """'2_1_001' → '2_1'."""
    return re.sub(r"_\d+$", "", file_index)


def load_dataset(src: str, csv_name: str) -> tuple[str, list[str], list[dict[str, str]]]:
    csv_path = os.path.join(src, csv_name)
    if not os.path.isfile(csv_path):
        csv_path = os.path.join(src, "seeds.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV не найден в {src}")
    fields, rows = read_csv_robust(csv_path)
    if "file_index" not in fields:
        raise ValueError("В CSV отсутствует обязательное поле: file_index")
    log.info("Загружен CSV: %s (%d строк)", csv_path, len(rows))
    return csv_path, fields, rows


def split_sources(rows: list[dict[str, str]], val_fraction: float, rng: random.Random) -> set[str]:
    sources = sorted({source_of(r["file_index"]) for r in rows})
    rng.shuffle(sources)
    n_val = max(1, int(round(len(sources) * val_fraction))) if val_fraction > 0 and len(sources) > 1 else 0
    val_sources = set(sources[:n_val])
    log.info("Split: %d фото, %d val", len(sources), len(val_sources))
    return val_sources


def augment_seed(
    img: np.ndarray, pts: np.ndarray, meta: dict[str, Any], row: dict[str, str],
    fi: str, src_name: str, dst: str, n_aug: int,
    rng_np: np.random.Generator, rng_std: random.Random,
) -> list[dict[str, str]]:
    pad = int(meta.get("padding", PAD_DEFAULT))
    angle_text = row.get("angle_deg", "").replace(",", ".").strip()
    try:
        angle_old = float(angle_text)
    except ValueError:
        angle_old = None
    bg_median, bg_std = estimate_background_stats(img, pts)
    aug_rows: list[dict[str, str]] = []
    sector = 360.0 / max(1, n_aug)

    for k in range(1, n_aug + 1):
        base_angle = (k - 1) * sector
        jitter = rng_std.uniform(-sector * 0.45, sector * 0.45)
        angle_rot = (base_angle + jitter) % 360.0
        flip = bool(rng_std.choice([False, True]))
        im2, pts2, A = rotate_any_angle(img, pts, angle_rot, flip, bg_median, bg_std, rng_np)
        im2 = photometric(im2, rng_np)
        new_fi = f"{fi}_{AUG_FMT.format(k=k)}"
        q = rng_std.randint(*JPEG_QUALITY)
        if not imwrite_unicode(os.path.join(dst, new_fi + IMG_EXT), im2, [int(cv2.IMWRITE_JPEG_QUALITY), q]):
            log.warning("Не удалось сохранить %s", new_fi)
            continue
        h2, w2 = im2.shape[:2]
        payload = {
            "file_index": new_fi, "index": meta.get("index", row.get("index", "")),
            "points_space": "crop", "image_width": int(w2), "image_height": int(h2),
            "padding": pad, "source_name": src_name, "aug_of": fi,
            "aug": {"rot_angle_deg": round(float(angle_rot), 2), "flip_h": bool(flip), "jpeg_q": int(q)},
            "points": np.round(pts2).astype(int).tolist(),
        }
        with open(os.path.join(dst, new_fi + CONTOUR_SUFFIX), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        r = dict(row)
        r["file_index"] = new_fi
        if angle_old is not None:
            r["angle_deg"] = f"{transform_angle(angle_old, A):.3f}"
        r.update({"source": src_name, "split": "train", "is_aug": 1, "aug_of": fi})
        aug_rows.append(r)
    return aug_rows


def run_augmentation(
    src: str = "out",
    csv_name: str = "seeds_edited.csv",
    dst: str = "out_aug",
    n_aug: int = 11,
    val_fraction: float = 0.2,
) -> None:
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
    rng_std = random.Random(SEED)
    rng_np = np.random.default_rng(SEED)
    csv_path, fields, rows = load_dataset(src, csv_name)
    os.makedirs(dst, exist_ok=True)
    val_sources = split_sources(rows, val_fraction, rng_std)

    out_fields = list(fields)
    for name in AUG_EXTRA_FIELDS:
        if name not in out_fields:
            out_fields.append(name)
    out_rows: list[dict[str, str]] = []
    n_orig = n_aug_total = n_skip = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(rows, desc="Аугментация", total=len(rows))
    except ImportError:
        iterator = rows

    for row in iterator:
        fi = row["file_index"]
        img_path = os.path.join(src, fi + IMG_EXT)
        json_path = os.path.join(src, fi + CONTOUR_SUFFIX)
        img = imread_unicode(img_path) if os.path.isfile(img_path) else None
        pts, meta = load_contour(json_path)
        if img is None or pts is None:
            n_skip += 1
            continue
        src_name = source_of(fi)
        split = "val" if src_name in val_sources else "train"
        shutil.copy2(img_path, os.path.join(dst, fi + IMG_EXT))
        shutil.copy2(json_path, os.path.join(dst, fi + CONTOUR_SUFFIX))
        r0 = dict(row)
        r0.update({"source": src_name, "split": split, "is_aug": 0, "aug_of": ""})
        out_rows.append(r0)
        n_orig += 1
        if split == "val":
            continue
        new_rows = augment_seed(img, pts, meta, row, fi, src_name, dst, n_aug, rng_np, rng_std)
        out_rows.extend(new_rows)
        n_aug_total += len(new_rows)

    write_csv(os.path.join(dst, "seeds.csv"), out_fields, out_rows)
    log.info("Оригиналов: %d, аугментаций: %d, пропущено: %d", n_orig, n_aug_total, n_skip)
    log.info("Результат: %s", dst)


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "out"
    csv_name = sys.argv[2] if len(sys.argv) > 2 else "seeds_edited.csv"
    dst = sys.argv[3] if len(sys.argv) > 3 else "out_aug"
    run_augmentation(src, csv_name, dst)


if __name__ == "__main__":
    main()