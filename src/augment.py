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

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import shutil
import sys
from typing import Any

import cv2
import numpy as np

from collections import Counter, defaultdict

try:
    from utils import imread_unicode, imwrite_unicode
except ImportError:
    from src.utils import imread_unicode, imwrite_unicode

# ---------- логирование ----------

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ---------- progress-bar ----------

try:
    from tqdm import tqdm
except ImportError:
    # fallback: минимальная замена без внешних зависимостей
    class tqdm:                                     # type: ignore[no-redef]
        """Простой fallback для tqdm."""
        def __init__(self, iterable=None, **kwargs):
            self._it = iterable
            self._total = kwargs.get("total", None)
            self._desc = kwargs.get("desc", "")
            self._n = 0

        def __iter__(self):
            for obj in self._it:
                yield obj
                self._n += 1
                if self._total and self._n % max(1, self._total // 20) == 0:
                    pct = self._n * 100 // self._total
                    print(f"\r{self._desc}: {pct}%", end="", flush=True)
            if self._total:
                print()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass


# ---------- настройки ----------

SRC_DIR = "out"
CSV_NAME = "seeds_edited.csv"   # если файла нет, берётся seeds.csv
DST_DIR = "out_aug"

N_AUG = 11             # копий на семечку (вместе с оригиналом получится x12)
VAL_FRACTION = 0.2    # доля для val (не аугментируются), 0 = выключить
SPLIT_MODE = "auto"   # "auto" (по фото если >=4 фото, иначе стратифицированно по семенам), "seeds", "photos"
SEED = 42
PAD_DEFAULT = 25      # отступ вокруг семечки, как в cut.py

# Суффиксы и паттерны имён файлов
IMG_EXT = ".jpg"
CONTOUR_SUFFIX = "_contour.json"
AUG_FMT = "aug{k:02d}"

# Геометрия: только повороты на 90 градусов и отражения (8 вариантов, оригинал пропускаем).
# Так фон остаётся настоящим, края не мылятся, размер в пикселях не меняется.
# Если N_AUG больше 7, геометрия повторяется, но фотометрия у копий всё равно разная.

# ── Фотометрия: яркость / контраст / гамма ──
ALPHA_RANGE = (0.85, 1.15)     # контраст (множитель)
BETA_RANGE = (-8, 8)           # яркость (смещение)
GAMMA_RANGE = (0.85, 1.15)     # гамма-коррекция

# ── Баланс белого / оттенок / насыщенность (HSV) ──
WB_RANGE = 0.04                # сдвиг каждого BGR-канала ±4 %
HUE_SHIFT_RANGE = (-4, 4)      # сдвиг Hue (из 180), ±4 — очень лёгкий
SAT_RANGE = (0.90, 1.10)       # множитель насыщенности

# ── Градиент освещения ──
GRADIENT_MAX = 0.06             # неравномерность освещения

# ── Шум ──
GAUSS_NOISE_MAX = 2.5           # макс. σ гауссова шума
ISO_NOISE_PROB = 0.3            # вероятность ISO-шума (Poisson-подобный)
ISO_NOISE_SCALE = (0.005, 0.02) # масштаб ISO-шума

# ── Blur / Defocus / Motion blur ──
BLUR_PROB = 0.25                # вероятность лёгкого гауссова размытия
BLUR_KSIZE_RANGE = (3, 5)       # размер ядра (нечётный, 3 или 5)
DEFOCUS_PROB = 0.15             # вероятность дискового defocus
DEFOCUS_RADIUS_RANGE = (1, 2)   # радиус диска (пикселей)
MOTION_BLUR_PROB = 0.15         # вероятность motion blur
MOTION_BLUR_KSIZE = (3, 7)      # длина ядра motion blur
MOTION_BLUR_ANGLE_RANGE = (0, 180)

# ── CLAHE / выравнивание гистограммы ──
CLAHE_PROB = 0.20               # вероятность CLAHE
CLAHE_CLIP_RANGE = (1.0, 3.0)   # clipLimit
CLAHE_GRID = (4, 4)             # tileGridSize
HISTEQ_PROB = 0.10              # вероятность полного выравнивания гистограммы

# ── JPEG-артефакты ──
JPEG_QUALITY = (85, 95)

EXTRA_FIELDS = ["source", "split", "is_aug", "aug_of"]


# ---------- ввод/вывод ----------

def read_csv(path: str) -> tuple[list[str], list[dict[str, str]]]:
    """Читает CSV с автоопределением разделителя, возвращает (поля, строки)."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except Exception:
            dialect = csv.excel

        reader = csv.DictReader(f, dialect=dialect)
        fields = list(reader.fieldnames or [])
        rows = [
            {k: (v or "") for k, v in r.items() if k is not None}
            for r in reader
        ]

    return fields, rows


def load_contour(path: str) -> tuple[np.ndarray | None, dict | None]:
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


def source_of(file_index: str) -> str:
    """'2_1_001' -> '2_1' (имя исходного фото)."""
    return re.sub(r"_\d+$", "", file_index)


# ---------- геометрия и фон ----------

def estimate_background_stats(
    img: np.ndarray,
    pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Оценивает медианный цвет фона и стандартное отклонение шума.
    Исключает область семечки (по расширенному контуру).
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [np.round(pts).astype(np.int32)], -1, 255, thickness=cv2.FILLED)

    # Дилатируем область семечки, чтобы гарантированно отсечь её границы
    d_k = max(3, min(w, h) // 12)
    if d_k % 2 == 0:
        d_k += 1
    dilated = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_k, d_k)))

    bg_pixels = img[dilated == 0]
    if len(bg_pixels) >= 50:
        bg_median = np.median(bg_pixels, axis=0).astype(np.float32)
        bg_std = np.std(bg_pixels, axis=0).astype(np.float32)
    else:
        # Fallback: берем полосу по периметру изображения
        border_mask = np.ones((h, w), dtype=bool)
        border_mask[2:-2, 2:-2] = False
        border_pixels = img[border_mask]
        if len(border_pixels) >= 10:
            bg_median = np.median(border_pixels, axis=0).astype(np.float32)
            bg_std = np.std(border_pixels, axis=0).astype(np.float32)
        else:
            bg_median = np.array([33.0, 34.0, 32.0], dtype=np.float32)
            bg_std = np.array([8.0, 8.0, 8.0], dtype=np.float32)

    # Ограничиваем шум снизу, чтобы фон не был плоским монотонным цветом
    bg_std = np.maximum(bg_std, 3.5)
    return bg_median, bg_std


def rotate_any_angle(
    img: np.ndarray,
    pts: np.ndarray,
    angle_deg: float,
    flip: bool,
    bg_median: np.ndarray,
    bg_std: np.ndarray,
    rng_np: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Поворот изображения и контура на произвольный угол angle_deg (в градусах).
    Образовавшаяся пустота за пределами исходного кадра заполняется
    медианным цветом фона с реалистичным гауссовым шумовым разбросом.

    Возвращает:
      - повернутое изображение (uint8)
      - новые координаты контура (float32)
      - матрицу линейного преобразования A (2x2) для пересчёта угла семечки
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Матрица поворота вокруг центра исходного изображения (против часовой)
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    # Вычисляем новые габариты холста, чтобы повернуть БЕЗ обрезания объекта
    rad = math.radians(angle_deg)
    cos_a = abs(math.cos(rad))
    sin_a = abs(math.sin(rad))
    new_w = int(math.ceil(h * sin_a + w * cos_a))
    new_h = int(math.ceil(h * cos_a + w * sin_a))

    # Смещаем центр вращения в центр нового расширенного холста
    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy

    # Поворот контура: (x, y) -> M @ [x, y, 1]
    pts_hom = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
    p_rot = (M @ pts_hom.T).T.astype(np.float32)

    # Генерация синтетического фона с медианой и шумом оригинала
    bg_noise = rng_np.normal(bg_median, bg_std, (new_h, new_w, 3))
    bg_canvas = np.clip(bg_noise, 0, 255).astype(np.float32)

    # Маска валидности исходного кадра
    src_mask = np.ones((h, w), dtype=np.float32)
    warped_mask = cv2.warpAffine(
        src_mask,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )

    # Поворачиваем само изображение с заполнением границ отражением
    warped_img = cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    ).astype(np.float32)

    # Мягкое наложение: исходные пиксели там, где была картинка,
    # и шумовой фон в образовавшихся пустых углах
    alpha = np.clip(warped_mask[:, :, None], 0.0, 1.0)
    out_f = warped_img * alpha + bg_canvas * (1.0 - alpha)
    out = np.clip(out_f, 0, 255).astype(np.uint8)

    # Линейная матрица направления для пересчёта угла длинной оси
    A = M[:2, :2].copy()

    # Отражение слева направо (flip)
    if flip:
        out = out[:, ::-1]
        p_rot[:, 0] = (new_w - 1) - p_rot[:, 0]
        A = np.diag([-1.0, 1.0]) @ A

    return np.ascontiguousarray(out), p_rot, A


def transform_angle(angle_deg: float, A: np.ndarray) -> float:
    """Угол длинной оси семечки после преобразования A (как в cut.py: 0..180)."""
    a = math.radians(angle_deg)
    d = A @ np.array([math.cos(a), math.sin(a)])
    return math.degrees(math.atan2(d[1], d[0])) % 180.0


# ---------- фотометрия ----------

def _make_disk_kernel(radius: int) -> np.ndarray:
    """Создаёт дисковое ядро для имитации defocus."""
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = (x * x + y * y <= radius * radius).astype(np.float32)
    total = mask.sum()
    if total > 0:
        mask /= total
    return mask


def _make_motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Создаёт ядро motion blur заданной длины и угла."""
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    kernel[center, :] = 1.0
    # Поворачиваем ядро на нужный угол
    M = cv2.getRotationMatrix2D((center, center), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (length, length), flags=cv2.INTER_LINEAR)
    total = kernel.sum()
    if total > 0:
        kernel /= total
    return kernel


def _apply_hue_sat_shift(
    img_uint8: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Небольшой сдвиг оттенка (Hue) и насыщенности (Saturation) в HSV."""
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Hue: циклический сдвиг (0..180 в OpenCV)
    h_shift = rng.uniform(*HUE_SHIFT_RANGE)
    hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180.0

    # Saturation: масштабирование
    s_scale = rng.uniform(*SAT_RANGE)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _apply_clahe(img_uint8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Применяет CLAHE (адаптивное выравнивание гистограммы) к L-каналу в LAB."""
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB)
    clip = rng.uniform(*CLAHE_CLIP_RANGE)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=CLAHE_GRID)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _apply_histeq(img_uint8: np.ndarray) -> np.ndarray:
    """Глобальное выравнивание гистограммы (по L-каналу в LAB)."""
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Применяет случайную комбинацию фотометрических преобразований.

    Пайплайн:
      1. Контраст / яркость / баланс белого (поканальный gain)
      2. Гамма-коррекция
      3. Сдвиг оттенка и насыщенности (HSV)
      4. Градиент освещения
      5. Гауссов шум / ISO-шум (Poisson-like)
      6. Лёгкий blur / defocus / motion blur
      7. CLAHE / выравнивание гистограммы
    """
    x = img.astype(np.float32)
    h, w = x.shape[:2]

    # ── 1. Контраст, яркость, баланс белого ──
    gain = 1.0 + rng.uniform(-WB_RANGE, WB_RANGE, 3).astype(np.float32)
    alpha = rng.uniform(*ALPHA_RANGE)
    beta = rng.uniform(*BETA_RANGE)
    x = x * alpha * gain + beta

    # ── 2. Гамма (до финального clip) ──
    g = rng.uniform(*GAMMA_RANGE)
    x_clipped = np.clip(x, 0, 255)
    x = 255.0 * (x_clipped / 255.0) ** g

    # ── 3. Сдвиг оттенка и насыщенности ──
    x = _apply_hue_sat_shift(np.clip(x, 0, 255).astype(np.uint8), rng).astype(np.float32)

    # ── 4. Линейный градиент освещения ──
    angle = rng.uniform(0, 2 * math.pi)
    strength = rng.uniform(0, GRADIENT_MAX)
    xs = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    ys = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    grad = 1.0 + strength * (xs * math.cos(angle) + ys * math.sin(angle))
    x = x * grad[:, :, None]

    # ── 5. Шум ──
    # Гауссов шум
    sigma = rng.uniform(0, GAUSS_NOISE_MAX)
    if sigma > 0.1:
        x = x + rng.normal(0, sigma, x.shape).astype(np.float32)

    # ISO-шум (Poisson-подобный, зависит от яркости пикселя)
    if rng.random() < ISO_NOISE_PROB:
        iso_scale = rng.uniform(*ISO_NOISE_SCALE)
        x_pos = np.clip(x, 0.1, 255)  # избегаем деления на 0
        # Пуассонов шум масштабируется с яркостью: σ ∝ √I
        poisson_sigma = iso_scale * np.sqrt(x_pos)
        x = x + rng.normal(0, 1, x.shape).astype(np.float32) * poisson_sigma

    # Промежуточный clip и конверсия в uint8 для blur/CLAHE
    result = np.clip(x, 0, 255).astype(np.uint8)

    # ── 6. Blur / Defocus / Motion blur (взаимоисключающие) ──
    blur_roll = rng.random()
    if blur_roll < BLUR_PROB:
        # Лёгкий гауссов blur
        ksize = int(rng.integers(BLUR_KSIZE_RANGE[0], BLUR_KSIZE_RANGE[1] + 1))
        if ksize % 2 == 0:
            ksize += 1
        result = cv2.GaussianBlur(result, (ksize, ksize), 0)
    elif blur_roll < BLUR_PROB + DEFOCUS_PROB:
        # Дисковый defocus
        radius = int(rng.integers(DEFOCUS_RADIUS_RANGE[0], DEFOCUS_RADIUS_RANGE[1] + 1))
        kernel = _make_disk_kernel(radius)
        result = cv2.filter2D(result, -1, kernel)
    elif blur_roll < BLUR_PROB + DEFOCUS_PROB + MOTION_BLUR_PROB:
        # Motion blur
        ksize = int(rng.integers(MOTION_BLUR_KSIZE[0], MOTION_BLUR_KSIZE[1] + 1))
        if ksize % 2 == 0:
            ksize += 1
        m_angle = rng.uniform(*MOTION_BLUR_ANGLE_RANGE)
        kernel = _make_motion_kernel(ksize, m_angle)
        result = cv2.filter2D(result, -1, kernel)

    # ── 7. CLAHE / выравнивание гистограммы (взаимоисключающие) ──
    hist_roll = rng.random()
    if hist_roll < CLAHE_PROB:
        result = _apply_clahe(result, rng)
    elif hist_roll < CLAHE_PROB + HISTEQ_PROB:
        result = _apply_histeq(result)

    return result


# ---------- загрузка и разделение данных ----------

def validate_csv_fields(fields: list[str]) -> None:
    """Проверяет наличие обязательных полей в CSV."""
    required = {"file_index"}
    missing = required - set(fields)
    if missing:
        raise ValueError(
            f"В CSV отсутствуют обязательные поля: {', '.join(sorted(missing))}"
        )


def load_dataset(
    src: str, csv_name: str
) -> tuple[str, list[str], list[dict[str, str]]]:
    """
    Загружает CSV-датасет. Пробует csv_name, затем seeds.csv.
    Возвращает (csv_path, fields, rows).
    """
    csv_path = os.path.join(src, csv_name)

    if not os.path.isfile(csv_path):
        csv_path = os.path.join(src, "seeds.csv")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV не найден в {src}")

    fields, rows = read_csv(csv_path)
    validate_csv_fields(fields)
    log.info("Загружен CSV: %s (%d строк, %d полей)", csv_path, len(rows), len(fields))

    return csv_path, fields, rows


def assign_splits(
    rows: list[dict[str, str]],
    val_fraction: float,
    rng_stdlib: Any,
    split_mode: str = SPLIT_MODE,
) -> dict[str, str]:
    """
    Разделяет набор данных на train и val.

    Стратегии:
    1. Если в CSV уже есть размеченные 'train' и 'val', сохраняет их.
    2. При split_mode == 'auto':
       - если фото >= 4, используется групповой сплит по фото (Group Split);
       - если фото < 4 (например, всего 1 фото), применяется стратифицированный
         сплит по семенам (по status и shape внутри каждого фото).
    3. При split_mode == 'seeds': всегда стратифицированный сплит по семенам.
    4. При split_mode == 'photos': групповой сплит по фото (если фото >= 2, иначе fallback на 'seeds').

    Возвращает:
      словарь {file_index: "train" | "val"}
    """
    if not rows:
        return {}

    existing_splits = {r.get("split", "").strip().lower() for r in rows if r.get("split")}
    if "train" in existing_splits and "val" in existing_splits:
        log.info("Обнаружена готовая разметка train/val в CSV — сохраняем её.")
        return {r["file_index"]: r.get("split", "train").strip().lower() for r in rows}

    n_total = len(rows)
    if val_fraction <= 0 or n_total < 2:
        log.info("VAL_FRACTION <= 0 или семян < 2: все %d семян назначены в train.", n_total)
        return {r["file_index"]: "train" for r in rows}

    sources_map: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        sources_map[source_of(r["file_index"])].append(r)

    unique_sources = sorted(sources_map.keys())

    use_photo_split = False
    if split_mode == "photos":
        if len(unique_sources) >= 2:
            use_photo_split = True
        else:
            log.warning(
                "Запрошен split_mode='photos', но источник всего 1 (%s). "
                "Автоматически переключаемся на поштучный сплит семян.",
                unique_sources[0],
            )
    elif split_mode == "auto":
        if len(unique_sources) >= 4:
            use_photo_split = True
        else:
            log.info(
                "Источников (%d) < 4. Применяется поштучный стратифицированный сплит семян.",
                len(unique_sources),
            )

    split_map: dict[str, str] = {}

    if use_photo_split:
        target_val = int(round(n_total * val_fraction))
        shuffled_srcs = list(unique_sources)
        rng_stdlib.shuffle(shuffled_srcs)

        selected_val_srcs: set[str] = set()
        current_val_seeds = 0

        for s in shuffled_srcs:
            s_len = len(sources_map[s])
            if current_val_seeds == 0 or (
                current_val_seeds + s_len <= target_val * 1.25
                and len(selected_val_srcs) < len(shuffled_srcs) - 1
            ):
                selected_val_srcs.add(s)
                current_val_seeds += s_len
                if current_val_seeds >= target_val:
                    break

        if not selected_val_srcs:
            min_src = min(shuffled_srcs, key=lambda s: len(sources_map[s]))
            selected_val_srcs.add(min_src)
            current_val_seeds = len(sources_map[min_src])

        for s, src_rows in sources_map.items():
            sp = "val" if s in selected_val_srcs else "train"
            for r in src_rows:
                split_map[r["file_index"]] = sp

        log.info(
            "Групповой сплит по фото: %d источников в val %s, всего семян в val: %d из %d (%.1f%%)",
            len(selected_val_srcs),
            sorted(selected_val_srcs),
            current_val_seeds,
            n_total,
            current_val_seeds / n_total * 100,
        )
    else:
        target_val = max(1, int(round(n_total * val_fraction)))
        val_indices: set[str] = set()

        for s, src_rows in sources_map.items():
            strata: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
            for r in src_rows:
                st = r.get("status", "unknown").strip().lower()
                sh = r.get("shape", "").strip().lower()
                strata[(st, sh)].append(r)

            for key, items in strata.items():
                items_copy = list(items)
                rng_stdlib.shuffle(items_copy)
                k_val = int(round(len(items_copy) * val_fraction))
                for item in items_copy[:k_val]:
                    val_indices.add(item["file_index"])

        all_fis = [r["file_index"] for r in rows]
        rng_stdlib.shuffle(all_fis)

        if len(val_indices) == 0 and target_val > 0:
            for fi in all_fis[:target_val]:
                val_indices.add(fi)
        elif abs(len(val_indices) - target_val) > max(2, int(target_val * 0.2)):
            if len(val_indices) < target_val:
                remain = [fi for fi in all_fis if fi not in val_indices]
                for fi in remain[:(target_val - len(val_indices))]:
                    val_indices.add(fi)
            elif len(val_indices) > target_val:
                val_list = list(val_indices)
                rng_stdlib.shuffle(val_list)
                val_indices = set(val_list[:target_val])

        for r in rows:
            fi = r["file_index"]
            split_map[fi] = "val" if fi in val_indices else "train"

        n_val_actual = len(val_indices)
        log.info(
            "Стратифицированный сплит по семенам: val: %d из %d (%.1f%%), train: %d (%.1f%%)",
            n_val_actual,
            n_total,
            n_val_actual / n_total * 100,
            n_total - n_val_actual,
            (n_total - n_val_actual) / n_total * 100,
        )

    train_statuses = Counter(
        r.get("status", "unknown") for r in rows if split_map[r["file_index"]] == "train"
    )
    val_statuses = Counter(
        r.get("status", "unknown") for r in rows if split_map[r["file_index"]] == "val"
    )
    log.info("Статусы в train: %s", dict(train_statuses))
    log.info("Статусы в val:   %s", dict(val_statuses))

    return split_map


def split_sources(
    rows: list[dict[str, str]],
    val_fraction: float,
    rng_stdlib: Any,
) -> set[str]:
    """Обратная совместимость: возвращает имена источников, семена которых попали в val."""
    s_map = assign_splits(rows, val_fraction, rng_stdlib)
    return {source_of(fi) for fi, sp in s_map.items() if sp == "val"}


# ---------- аугментация одной семечки ----------

def augment_seed(
    img: np.ndarray,
    pts: np.ndarray,
    meta: dict,
    row: dict[str, str],
    fi: str,
    src_name: str,
    dst: str,
    n_aug: int,
    rng_np: np.random.Generator,
    rng_stdlib: Any,
) -> list[dict[str, str]]:
    """
    Создаёт n_aug аугментированных копий одной семечки.
    Поворот на любой угол (0..360°), заполнение пустоты медианным фоном с шумом,
    случайный flip, фотометрия и сохранение контура и метаданных.
    Возвращает список новых CSV-строк.
    """
    pad = int(meta.get("padding", PAD_DEFAULT))

    angle_text = row.get("angle_deg", "").replace(",", ".").strip()
    try:
        angle_old = float(angle_text)
    except ValueError:
        angle_old = None

    # Оцениваем цвет и шум фона для синтеза пустых областей
    bg_median, bg_std = estimate_background_stats(img, pts)

    aug_rows: list[dict[str, str]] = []

    # Равномерно распределяем секторы по 360° со случайным дрожанием
    sector_size = 360.0 / max(1, n_aug)

    for k in range(1, n_aug + 1):
        # Случайный угол в пределах сектора (поворот на ЛЮБОЙ угол)
        base_angle = (k - 1) * sector_size
        jitter = rng_stdlib.uniform(-sector_size * 0.45, sector_size * 0.45)
        angle_rot = (base_angle + jitter) % 360.0

        # Случайное отражение (flip)
        flip = bool(rng_stdlib.choice([False, True]))

        im2, pts2, A = rotate_any_angle(
            img=img,
            pts=pts,
            angle_deg=angle_rot,
            flip=flip,
            bg_median=bg_median,
            bg_std=bg_std,
            rng_np=rng_np,
        )
        im2 = photometric(im2, rng_np)

        new_fi = f"{fi}_{AUG_FMT.format(k=k)}"
        q = rng_stdlib.randint(*JPEG_QUALITY)

        out_path = os.path.join(dst, new_fi + IMG_EXT)
        ok = imwrite_unicode(out_path, im2, [int(cv2.IMWRITE_JPEG_QUALITY), q])

        if not ok:
            log.warning("Не удалось сохранить %s", out_path)
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
            "aug": {
                "rot_angle_deg": round(float(angle_rot), 2),
                "flip_h": bool(flip),
                "jpeg_q": int(q),
            },
            "points": np.round(pts2).astype(int).tolist(),
        }

        json_path = os.path.join(dst, new_fi + CONTOUR_SUFFIX)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        json_path = os.path.join(dst, new_fi + CONTOUR_SUFFIX)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        # Размерные поля (length/width/area/mm/thickness/mass/shape)
        # и фокус (edge_width_px/status) копируются без изменений.
        r1 = dict(row)
        r1["file_index"] = new_fi

        if angle_old is not None:
            r1["angle_deg"] = f"{transform_angle(angle_old, A):.3f}"

        r1.update({"source": src_name, "split": "train", "is_aug": 1, "aug_of": fi})
        aug_rows.append(r1)

    return aug_rows


# ---------- сохранение результатов ----------

def save_results(
    dst: str,
    out_fields: list[str],
    out_rows: list[dict[str, str]],
) -> None:
    """Записывает итоговый CSV."""
    out_csv = os.path.join(dst, "seeds.csv")

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    log.info("CSV сохранён: %s (%d строк)", out_csv, len(out_rows))


# ---------- основной код ----------

def resolve_path(d: str) -> str:
    if os.path.isabs(d) or os.path.isdir(d):
        return d
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        d,
        os.path.join(script_dir, d),
        os.path.join(os.path.dirname(script_dir), d),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return d


def main() -> None:
    src = resolve_path(sys.argv[1] if len(sys.argv) > 1 else SRC_DIR)
    csv_name = sys.argv[2] if len(sys.argv) > 2 else CSV_NAME
    dst = resolve_path(sys.argv[3] if len(sys.argv) > 3 else DST_DIR)

    # Воспроизводимые генераторы случайных чисел
    import random
    rng_stdlib = random.Random(SEED)
    rng_np = np.random.default_rng(SEED)

    # Загрузка данных
    csv_path, fields, rows = load_dataset(src, csv_name)
    os.makedirs(dst, exist_ok=True)

    # Разделение на train/val по фотографиям
    split_map = assign_splits(rows, VAL_FRACTION, rng_stdlib, SPLIT_MODE)

    # Подготовка выходных полей
    out_fields = list(fields)
    for name in EXTRA_FIELDS:
        if name not in out_fields:
            out_fields.append(name)

    out_rows: list[dict[str, str]] = []
    n_orig = 0
    n_aug = 0
    n_skip = 0

    for row in tqdm(rows, desc="Аугментация", total=len(rows)):
        fi = row["file_index"]
        img_path = os.path.join(src, fi + IMG_EXT)
        json_path = os.path.join(src, fi + CONTOUR_SUFFIX)

        img = imread_unicode(img_path) if os.path.isfile(img_path) else None
        pts, meta = load_contour(json_path)

        if img is None or pts is None:
            n_skip += 1
            log.debug("Пропущено: %s (изображение/контур не найдены)", fi)
            continue

        src_name = source_of(fi)
        split = split_map.get(fi, "train")

        # Оригинал копируем как есть
        shutil.copy2(img_path, os.path.join(dst, fi + IMG_EXT))
        shutil.copy2(json_path, os.path.join(dst, fi + CONTOUR_SUFFIX))

        r0 = dict(row)
        r0.update({"source": src_name, "split": split, "is_aug": 0, "aug_of": ""})
        out_rows.append(r0)
        n_orig += 1

        if split == "val":
            continue

        # Аугментация train-семечек
        new_rows = augment_seed(
            img=img,
            pts=pts,
            meta=meta,
            row=row,
            fi=fi,
            src_name=src_name,
            dst=dst,
            n_aug=N_AUG,
            rng_np=rng_np,
            rng_stdlib=rng_stdlib,
        )
        out_rows.extend(new_rows)
        n_aug += len(new_rows)

    # Сохранение
    save_results(dst, out_fields, out_rows)

    n_val_final = sum(1 for r in out_rows if r.get("split") == "val")
    n_train_orig = sum(1 for r in out_rows if r.get("split") == "train" and str(r.get("is_aug")) == "0")
    log.info(
        "Итог: оригиналов %d (в train: %d, в val: %d), аугментаций: %d, пропущено: %d, всего записей: %d",
        n_orig, n_train_orig, n_val_final, n_aug, n_skip, len(out_rows)
    )
    log.info("Результат: %s", dst)


if __name__ == "__main__":
    main()
