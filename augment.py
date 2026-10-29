#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
augment.py — аугментация нарезанного датасета семян.

Формат входа (по умолчанию папка ``out``)::

    <file_index>.jpg
    <file_index>_contour.json
    seeds_edited.csv        # или seeds.csv

Формат выхода (по умолчанию папка ``out_aug``)::

    <file_index>.<ext>                 # оригинал, скопированный без изменений
    <file_index>_contour.json          # контур оригинала без изменений
    <file_index>_augNN.jpg             # аугментированная копия
    <file_index>_augNN_contour.json    # контур в координатах аугментированного кропа
    seeds.csv                          # исходные + новые строки

Ключевые принципы:

* train/val делится по исходным фотографиям, а не по отдельным семечкам;
  val-фотографии не аугментируются;
* геометрия — только 8 симметрий прямоугольника (повороты на 90/180/270
  и отражения). Интерполяции нет, поэтому размеры семечки в пикселях
  и физические поля (``length``, ``width``, ``area``, ``mm``, ``mass``,
  ``shape``) остаются корректными;
* ``angle_deg`` пересчитывается через матрицу преобразования;
* фотометрия — яркость, контраст, гамма, баланс белого, градиент
  освещения, шум и JPEG quality;
* генератор случайных чисел детерминирован для каждого ``file_index``:
  результат не зависит от порядка обхода строк и от пропущенных файлов;
* запись изображения, JSON и CSV выполняется через временный файл
  с последующей атомарной заменой, чтобы не оставлять битые артефакты.

Использование::

    python augment.py
    python augment.py out seeds_edited.csv out_aug
    python augment.py --src out --csv seeds_edited.csv --dst out_aug --n-aug 7
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Если utils рядом, используем его. Если нет — работаем через cv2 напрямую.
try:
    from utils import imread_unicode, imwrite_unicode
except ImportError:
    def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
        try:
            data = np.fromfile(path, dtype=np.uint8)
            return cv2.imdecode(data, flags)
        except Exception:
            return None

    def imwrite_unicode(
        path: str,
        img: np.ndarray,
        params: Optional[Sequence[int]] = None,
    ) -> bool:
        ext = os.path.splitext(path)[1] or ".jpg"
        try:
            ok, buf = cv2.imencode(ext, img, list(params or []))
            if not ok:
                return False
            buf.tofile(path)
            return True
        except Exception:
            return False


# ---------------------------- настройки по умолчанию ----------------------------

SRC_DIR = "out"
CSV_NAME = "seeds_edited.csv"   # если файла нет, будет использован seeds.csv
DST_DIR = "out_aug"

N_AUG = 7              # копий на семечку; вместе с оригиналом получится x8
VAL_FRACTION = 0.2     # доля исходных фото под val; 0 — выключить
SEED = 42
PAD_DEFAULT = 25       # padding вокруг семечки, как в main.py

# Фотометрия. Диапазоны небольшие, так как фон тёмный.
ALPHA_RANGE = (0.9, 1.1)       # контраст
BETA_RANGE = (-6.0, 6.0)       # яркость
GAMMA_RANGE = (0.9, 1.1)       # гамма
WB_RANGE = 0.03                # баланс белого, +-3% на канал
GRADIENT_MAX = 0.05            # неравномерность освещения
NOISE_MAX = 2.0                # макс. сигма гауссова шума
JPEG_QUALITY = (85, 95)

# Дополнительные поля в CSV.
EXTRA_FIELDS = ["source", "split", "is_aug", "aug_of"]

# Если в CSV есть поля с размером кропа/изображения — обновим их для аугментаций.
# Важные поля семечки (length/width/area/mm/thickness/mass/shape) не трогаем.
IMAGE_SIZE_FIELDS = (
    ("image_width", "image_height"),
    ("img_width", "img_height"),
    ("crop_width", "crop_height"),
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
CONTOUR_SUFFIX = "_contour.json"

# Поля исходного JSON, которые описывают положение кропа в исходном фото.
# После поворота/отражения они неверны, поэтому в аугментированный JSON
# не переносятся.
STALE_META_FIELDS = ("crop_origin_in_source", "bbox_in_source")

CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1251", "latin-1")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"

# <stem>_augNN.jpg / <stem>_augNN_contour.json
AUG_FILE_RE = re.compile(
    r"^(?P<stem>.+)_aug\d+(?:_contour\.json|\.[A-Za-z0-9]+)$",
    re.IGNORECASE,
)

log = logging.getLogger("augment")


# ---------------------------- модели данных ----------------------------

@dataclass(frozen=True, slots=True)
class Settings:
    """Параметры запуска, полученные из командной строки."""

    src: str
    csv_name: str
    dst: str
    csv_explicit: bool = False

    n_aug: int = N_AUG
    val_fraction: float = VAL_FRACTION
    seed: int = SEED
    pad_default: int = PAD_DEFAULT

    jpeg_quality: Tuple[int, int] = JPEG_QUALITY

    alpha_range: Tuple[float, float] = ALPHA_RANGE
    beta_range: Tuple[float, float] = BETA_RANGE
    gamma_range: Tuple[float, float] = GAMMA_RANGE
    wb_range: float = WB_RANGE
    gradient_max: float = GRADIENT_MAX
    noise_max: float = NOISE_MAX

    no_photometric: bool = False
    clean_dst: bool = False
    clean_stale: bool = True
    log_level: str = "INFO"


@dataclass(frozen=True, slots=True)
class AugOp:
    """Геометрическая операция: поворот на k90*90° и отражение по горизонтали."""

    k90: int
    flip: bool

    @property
    def name(self) -> str:
        if self.k90 == 0 and not self.flip:
            return "identity"
        parts = []
        if self.k90:
            parts.append(f"rot{self.k90 * 90}")
        if self.flip:
            parts.append("flip_h")
        return "+".join(parts)


IDENTITY_OP = AugOp(0, False)

# Семь неединичных симметрий. Именно они используются по умолчанию (n_aug=7).
NON_IDENTITY_GEOMETRIES: Tuple[AugOp, ...] = tuple(
    AugOp(k, flip)
    for k in range(4)
    for flip in (False, True)
    if (k, flip) != (0, False)
)

# Полный набор из восьми симметрий: используется, когда n_aug >= 8.
ALL_GEOMETRIES: Tuple[AugOp, ...] = NON_IDENTITY_GEOMETRIES + (IDENTITY_OP,)


@dataclass(slots=True)
class SeedTask:
    """Одна исходная семечка, подготовленная к обработке."""

    row: Dict[str, str]
    file_index: str
    source: str
    image_path: str
    contour_path: str
    points: np.ndarray
    meta: Dict[str, Any]


@dataclass(slots=True)
class PlanningResult:
    tasks: List[SeedTask]
    skipped: int = 0
    duplicates: int = 0


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """Результат разделения по исходным фотографиям."""

    val_sources: frozenset[str]
    source_count: int

    @property
    def train_count(self) -> int:
        return self.source_count - len(self.val_sources)

    def is_val(self, source: str) -> bool:
        return source in self.val_sources


@dataclass(slots=True)
class RunStats:
    originals: int = 0
    augmentations: int = 0
    skipped: int = 0
    errors: int = 0
    duplicates: int = 0

    def summary(self) -> str:
        return (
            f"Оригиналов: {self.originals}, аугментаций: {self.augmentations}, "
            f"пропущено: {self.skipped}, дубликатов: {self.duplicates}, "
            f"ошибок: {self.errors}"
        )


# ---------------------------- вспомогательные функции ----------------------------

def ordered_float_pair(values: Sequence[float]) -> Tuple[float, float]:
    a, b = float(values[0]), float(values[1])
    return (a, b) if a <= b else (b, a)


def ordered_int_pair(values: Sequence[int]) -> Tuple[int, int]:
    a, b = int(values[0]), int(values[1])
    return (a, b) if a <= b else (b, a)


def parse_float(value: Any) -> Optional[float]:
    """Возвращает конечное число или None (пустая строка, NaN, inf, мусор)."""
    if value is None:
        return None

    text = str(value).replace("\u00a0", "").replace(",", ".").strip()
    if not text:
        return None

    try:
        number = float(text)
    except ValueError:
        return None

    return number if math.isfinite(number) else None


def parse_int(value: Any) -> Optional[int]:
    number = parse_float(value)
    if number is None:
        return None
    try:
        return int(round(number))
    except (ValueError, OverflowError):
        return None


def derive_rng(seed: int, *parts: str) -> np.random.Generator:
    """
    Детерминированный RNG, зависящий от seed и ключа (обычно file_index).

    hashlib вместо hash() нужен, чтобы результат не зависел от
    PYTHONHASHSEED и порядка обработки строк.
    """
    key = "\x1f".join((str(seed), *parts)).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def preview(items: Sequence[str], limit: int = 20) -> str:
    shown = ", ".join(items[:limit])
    if len(items) > limit:
        shown += ", ..."
    return shown


def normalize_file_index(value: str) -> str:
    """
    Убирает слэши назад и расширение изображения, если оно попало в file_index.
    Пример: '2_1_001.jpg' -> '2_1_001'.
    """
    fi = (value or "").strip().replace("\\", "/")
    root, ext = os.path.splitext(fi)
    if ext.lower() in IMAGE_EXTENSIONS:
        return root
    return fi


def is_safe_file_index(fi: str) -> bool:
    """
    Разрешает только относительный путь внутри src/dst.

    Защищает от абсолютных путей, дисков Windows и '..'. Это важно:
    os.path.join(base, 'C:/secret') на Windows игнорирует base.
    """
    if not fi or fi.startswith("/"):
        return False

    if os.path.isabs(fi):
        return False

    drive, _ = os.path.splitdrive(fi)
    if drive:
        return False

    parts = fi.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return False

    for char in fi:
        if char in '<>:"|?*' or ord(char) < 32:
            return False

    return True


def resolve_under(base: str, relative: str, suffix: str = "") -> str:
    """
    Собирает путь base/relative+suffix и проверяет, что он остался внутри base.
    """
    base_abs = os.path.abspath(base)
    candidate = os.path.abspath(os.path.join(base_abs, relative + suffix))

    try:
        inside = os.path.commonpath(
            [os.path.realpath(base_abs), os.path.realpath(candidate)]
        )
    except ValueError as exc:
        raise ValueError(f"путь вне рабочей папки: {relative!r}") from exc

    if os.path.normcase(inside) != os.path.normcase(os.path.realpath(base_abs)):
        raise ValueError(f"путь вне рабочей папки: {relative!r}")

    return candidate


def source_of(file_index: str) -> str:
    """
    Возвращает имя исходного фото по индексу семечки.
    '2_1_001' -> '2_1'; 'sub/2_1_001_aug03' -> 'sub/2_1'.
    """
    parent, _, name = file_index.rpartition("/")
    name = re.sub(r"_aug\d+$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"_\d+$", "", name)
    return f"{parent}/{name}" if parent else name


def sanitize_source_name(value: Any) -> Optional[str]:
    """Приводит source_name из JSON к безопасному виду (как utils.sanitize_filename)."""
    if not isinstance(value, str):
        return None

    name = value.strip().replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r'[\\/:*?"<>|\s]+', "_", name).strip("._ ")

    return name or None


def source_from_meta(meta: Dict[str, Any], file_index: str) -> str:
    """
    Берёт source_name из contour JSON (его пишет main.py), иначе — эвристика
    по file_index. Это надёжнее для разделения train/val.
    """
    name = sanitize_source_name(meta.get("source_name"))
    if not name:
        return source_of(file_index)

    parent, _, _ = file_index.rpartition("/")
    return f"{parent}/{name}" if parent else name


def row_is_augmented(row: Dict[str, str]) -> bool:
    """Не даёт повторно аугментировать уже аугментированные строки."""
    if str(row.get("aug_of", "")).strip():
        return True

    flag = str(row.get("is_aug", "")).strip().lower()
    if flag in {"1", "true", "yes", "y", "да"}:
        return True

    file_index = normalize_file_index(row.get("file_index", ""))
    return bool(re.search(r"_aug\d+$", file_index, re.IGNORECASE))


def find_csv(src: str, csv_name: str, explicit: bool) -> str:
    """
    Ищет CSV.

    Без явного --csv приоритет у файла внутри src: запуск из другого
    каталога не должен подхватывать посторонний seeds_edited.csv из cwd.
    Явно указанный относительный путь, наоборот, сначала проверяется в cwd.
    """
    in_src = os.path.join(src, csv_name)
    candidates: List[str] = (
        [csv_name, in_src] if explicit else [in_src, csv_name]
    )

    if not explicit:
        for name in (CSV_NAME, "seeds.csv"):
            path = os.path.join(src, name)
            if path not in candidates:
                candidates.append(path)

    # Убираем повторы, сохраняя порядок (важно для абсолютного csv_name).
    unique: List[str] = []
    for path in candidates:
        if path not in unique:
            unique.append(path)

    for path in unique:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "CSV не найден. Проверены пути: " + ", ".join(unique)
    )


def _detect_delimiter(sample: str) -> str:
    """Определяет разделитель CSV, отдавая приоритет файлам с file_index."""
    sniffed: Optional[str] = None
    try:
        sniffed = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        pass

    first_line = sample.splitlines()[0] if sample.splitlines() else ""
    for delimiter in ([sniffed] if sniffed else []) + [",", ";", "\t"]:
        if not delimiter:
            continue
        try:
            header = next(csv.reader([first_line], delimiter=delimiter))
        except csv.Error:
            continue
        names = {name.strip().lstrip("\ufeff").lower() for name in header}
        if "file_index" in names:
            return delimiter

    return sniffed or ","


def read_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Читает CSV с несколькими кодировками, определяет разделитель,
    нормализует имена колонок и проверяет обязательный file_index.
    """
    last_error: Optional[Exception] = None

    for encoding in CSV_ENCODINGS:
        try:
            with open(path, "r", encoding=encoding, newline="") as handle:
                sample = handle.read(8192)
                handle.seek(0)

                delimiter = _detect_delimiter(sample)
                reader = csv.reader(handle, delimiter=delimiter)
                raw_header = next(reader, None)

                if not raw_header:
                    raise ValueError(f"CSV пуст или без заголовка: {path}")

                fields: List[str] = []
                positions: List[Tuple[str, int]] = []
                seen = set()

                for position, raw_name in enumerate(raw_header):
                    name = (raw_name or "").strip().lstrip("\ufeff")
                    if not name:
                        continue
                    if name in seen:
                        raise ValueError(f"Дублирующаяся колонка {name!r}: {path}")
                    seen.add(name)
                    fields.append(name)
                    positions.append((name, position))

                if "file_index" not in seen:
                    raise ValueError(f"В CSV нет обязательной колонки file_index: {path}")

                rows: List[Dict[str, str]] = []
                for raw_row in reader:
                    if not raw_row or all(not value.strip() for value in raw_row):
                        continue

                    row: Dict[str, str] = {}
                    for name, position in positions:
                        value = raw_row[position] if position < len(raw_row) else ""
                        row[name] = value.strip()
                    rows.append(row)

                return fields, rows

        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except csv.Error as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise ValueError(f"Не удалось прочитать CSV: {path}")


def _points_to_array(raw_points: Any) -> Optional[np.ndarray]:
    """Преобразует points из JSON ([[x, y], ...] или [{x, y}, ...]) в Nx2 float32."""
    if not isinstance(raw_points, (list, tuple)) or not raw_points:
        return None

    rows: List[Tuple[float, float]] = []

    if isinstance(raw_points[0], dict):
        for item in raw_points:
            if not isinstance(item, dict):
                continue
            try:
                x = float(item.get("x"))
                y = float(item.get("y"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                rows.append((x, y))
    else:
        try:
            array = np.asarray(raw_points, dtype=np.float32)
        except (TypeError, ValueError):
            return None

        if array.ndim != 2 or array.shape[1] < 2:
            return None

        array = array[:, :2]
        array = array[np.isfinite(array).all(axis=1)]
        rows = [(float(x), float(y)) for x, y in array]

    if len(rows) < 3:
        return None

    return np.asarray(rows, dtype=np.float32).reshape(-1, 2)


def load_contour(path: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Возвращает (points Nx2 float32, meta без поля points) или (None, None).
    """
    if not os.path.isfile(path):
        return None, None

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if isinstance(data, list):
            data = {"points": data}

        if not isinstance(data, dict):
            return None, None

        points = _points_to_array(data.get("points", []))
    except Exception as exc:
        log.debug("Не удалось прочитать контур %s: %s", path, exc)
        return None, None

    if points is None:
        return None, None

    meta = dict(data)
    meta.pop("points", None)

    return points, meta


def ensure_bgr(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Приводит изображение к 8-bit, 3 канала BGR."""
    if img is None or img.size == 0:
        return None

    if img.dtype != np.uint8:
        try:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        except Exception:
            img = cv2.convertScaleAbs(img)

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3:
        channels = img.shape[2]
        if channels == 1:
            return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        if channels == 3:
            return img
        if channels == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return None


def sanitize_contour(points: np.ndarray) -> np.ndarray:
    """
    Удаляет повторяющиеся соседние точки и замыкающую точку.
    Если после чистки осталось меньше 3 точек, возвращает то, что осталось:
    вызывающий код должен считать такой контур вырожденным.
    """
    if len(points) < 2:
        return points

    mask = np.ones(len(points), dtype=bool)
    mask[1:] = np.any(points[1:] != points[:-1], axis=1)

    if np.array_equal(points[0], points[-1]):
        mask[-1] = False

    return points[mask]


def safe_remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


# ---------------------------- геометрия ----------------------------

def geometric(
    img: np.ndarray,
    points: np.ndarray,
    op: AugOp,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Поворот на op.k90*90° против часовой, затем отражение слева-направо.

    Возвращает:
      - новое изображение;
      - контур в целых координатах нового кропа;
      - матрицу A 2x2 для пересчёта направления длинной оси.
    """
    out = img
    transformed = points.astype(np.float32, copy=True)
    matrix = np.eye(2, dtype=np.float32)

    for _ in range(int(op.k90) % 4):
        width = out.shape[1]
        out = np.rot90(out)
        transformed = np.column_stack(
            (transformed[:, 1], (width - 1) - transformed[:, 0])
        )
        matrix = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32) @ matrix

    if op.flip:
        width = out.shape[1]
        out = out[:, ::-1]
        transformed[:, 0] = (width - 1) - transformed[:, 0]
        matrix = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float32) @ matrix

    out = np.ascontiguousarray(out)

    height, width = out.shape[:2]
    transformed = np.rint(transformed).astype(np.int32)

    if len(transformed):
        transformed[:, 0] = np.clip(transformed[:, 0], 0, max(0, width - 1))
        transformed[:, 1] = np.clip(transformed[:, 1], 0, max(0, height - 1))

    transformed = sanitize_contour(transformed)

    if len(transformed) < 3 or abs(float(cv2.contourArea(transformed))) < 1.0:
        raise ValueError("контур вырожден после геометрического преобразования")

    return out, transformed, matrix


def transform_angle(angle_deg: Optional[float], matrix: np.ndarray) -> Optional[float]:
    """Пересчитывает угол длинной оси семечки; результат в диапазоне [0, 180)."""
    if angle_deg is None or not math.isfinite(float(angle_deg)):
        return None

    radians = math.radians(float(angle_deg))
    direction = matrix @ np.array(
        [math.cos(radians), math.sin(radians)],
        dtype=np.float32,
    )
    angle = math.degrees(
        math.atan2(float(direction[1]), float(direction[0]))
    ) % 180.0

    return angle if math.isfinite(angle) else None


def build_ops(n_aug: int) -> List[AugOp]:
    """
    Детерминированный список операций для n_aug копий.

    При n_aug <= 7 используются разные неединичные симметрии. Восьмая копия
    получает identity (отличается только фотометрией), дальнейшие — цикл.
    Фиксированный порядок делает геометрию одинаковой для всех семечек
    и не зависит от порядка обработки.
    """
    if n_aug <= 0:
        return []

    return [
        ALL_GEOMETRIES[index % len(ALL_GEOMETRIES)]
        for index in range(n_aug)
    ]


# ---------------------------- фотометрия ----------------------------

def photometric(
    img: np.ndarray,
    settings: Settings,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Фотометрическая аугментация:
      - баланс белого;
      - контраст и яркость;
      - гамма;
      - линейный градиент освещения;
      - гауссов шум.
    """
    was_gray = img.ndim == 2
    x = img.astype(np.float32)

    if was_gray:
        x = x[..., None]

    channels = x.shape[2]

    gain = 1.0 + rng.uniform(
        -settings.wb_range,
        settings.wb_range,
        size=channels,
    ).astype(np.float32)

    alpha = float(rng.uniform(*settings.alpha_range))
    beta = float(rng.uniform(*settings.beta_range))
    x = np.clip(x * alpha * gain + beta, 0.0, 255.0)

    gamma = float(rng.uniform(*settings.gamma_range))
    if abs(gamma - 1.0) > 1e-6:
        x = 255.0 * np.power(x / 255.0, gamma)

    grad_strength = float(rng.uniform(0.0, settings.gradient_max))
    if grad_strength > 1e-6:
        height, width = x.shape[:2]
        grad_angle = float(rng.uniform(0.0, 2.0 * math.pi))
        xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
        gradient = 1.0 + grad_strength * (
            xs * math.cos(grad_angle) + ys * math.sin(grad_angle)
        )
        x *= gradient[:, :, None]

    sigma = float(rng.uniform(0.0, settings.noise_max))
    if sigma > 1e-6:
        noise = rng.standard_normal(x.shape, dtype=np.float32)
        x += noise * sigma

    x = np.clip(x, 0.0, 255.0).astype(np.uint8)
    return x[:, :, 0] if was_gray else x


# ---------------------------- запись результатов ----------------------------

def update_image_size_fields(
    row: Dict[str, Any],
    fields: Sequence[str],
    img: np.ndarray,
) -> None:
    """Обновляет поля размера изображения/кропа, если они есть в CSV."""
    height, width = img.shape[:2]

    for width_field, height_field in IMAGE_SIZE_FIELDS:
        if width_field in fields:
            row[width_field] = str(int(width))
        if height_field in fields:
            row[height_field] = str(int(height))


def build_contour_payload(
    meta: Dict[str, Any],
    row: Dict[str, str],
    new_file_index: str,
    original_file_index: str,
    source: str,
    points: np.ndarray,
    img: np.ndarray,
    padding: int,
    op: AugOp,
) -> Dict[str, Any]:
    """Собирает JSON аугментированного контура без устаревших полей кропа."""
    payload = {
        key: value
        for key, value in meta.items()
        if key not in STALE_META_FIELDS
    }

    height, width = img.shape[:2]

    index_value: Any = row.get("index", "")
    if not str(index_value).strip():
        index_value = payload.get("index", "")

    # В исходных JSON index — число; сохраняем тип, если значение корректно.
    index_number = parse_int(index_value)
    if index_number is not None:
        index_value = index_number

    payload.update({
        "file_index": new_file_index,
        "index": index_value,
        "points_space": "crop",
        "image_width": int(width),
        "image_height": int(height),
        "padding": int(padding),
        "source_name": source,
        "aug_of": original_file_index,
        "aug": {
            "rot90": int(op.k90),
            "flip_h": bool(op.flip),
        },
        "points": points.astype(int).tolist(),
    })

    return payload


def _temp_path(target: str) -> str:
    """Создаёт временный файл рядом с target (в той же файловой системе)."""
    directory = os.path.dirname(target) or "."
    os.makedirs(directory, exist_ok=True)

    suffix = os.path.splitext(target)[1]
    descriptor, path = tempfile.mkstemp(
        prefix=".augment-",
        suffix=suffix,
        dir=directory,
    )
    os.close(descriptor)
    return path


def atomic_copy(src: str, dst: str) -> None:
    """Копирует файл через временный, чтобы не оставить недописанный dst."""
    if os.path.normcase(os.path.abspath(src)) == os.path.normcase(os.path.abspath(dst)):
        return

    temp = _temp_path(dst)
    try:
        shutil.copy2(src, temp)
        os.replace(temp, dst)
    finally:
        safe_remove(temp)


def write_augmentation(
    img_path: str,
    json_path: str,
    img: np.ndarray,
    payload: Dict[str, Any],
    jpeg_quality: int,
) -> None:
    """Записывает пару jpg + json; оба файла сначала готовятся во временных."""
    img_temp = _temp_path(img_path)
    json_temp = _temp_path(json_path)

    try:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        if not imwrite_unicode(img_temp, img, params):
            raise OSError(f"не удалось записать изображение: {img_path}")

        with open(json_temp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)

        os.replace(img_temp, img_path)
        os.replace(json_temp, json_path)
    finally:
        safe_remove(img_temp)
        safe_remove(json_temp)


def write_csv(path: str, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    """Атомарная запись итогового CSV в utf-8-sig."""
    temp = _temp_path(path)
    try:
        with open(temp, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fieldnames),
                extrasaction="ignore",
                restval="",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
    finally:
        safe_remove(temp)


def clean_stale_outputs(
    keep_by_stem: Dict[Tuple[str, str], set[str]],
) -> None:
    """
    Удаляет старые <stem>_augNN.*, которых нет в текущем плане.

    keep_by_stem: {(папка, stem): множество имён файлов, которые нужно оставить}.

    Сканируем каждую папку один раз, а не для каждой строки CSV: на большом
    плоском датасете это принципиально быстрее.
    """
    by_directory: Dict[str, Dict[str, set[str]]] = {}

    for (directory, stem), keep in keep_by_stem.items():
        by_directory.setdefault(directory, {})[stem] = keep

    for directory, stems in by_directory.items():
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            log.debug("Не удалось прочитать папку для очистки %s: %s", directory, exc)
            continue

        for entry in entries:
            if not entry.is_file():
                continue

            match = AUG_FILE_RE.match(entry.name)
            if match is None:
                continue

            keep = stems.get(match.group("stem"))
            if keep is None or entry.name in keep:
                continue

            try:
                os.remove(entry.path)
                log.debug("Удалён устаревший файл: %s", entry.path)
            except OSError as exc:
                log.warning("Не удалось удалить %s: %s", entry.path, exc)


# ---------------------------- подготовка данных ----------------------------

def find_source_image(src: str, file_index: str) -> Optional[str]:
    """Ищет оригинал с любым поддерживаемым расширением, начиная с .jpg."""
    for extension in IMAGE_EXTENSIONS:
        try:
            path = resolve_under(src, file_index, extension)
        except ValueError:
            return None
        if os.path.isfile(path):
            return path
    return None


def plan_tasks(src: str, rows: Sequence[Dict[str, str]]) -> PlanningResult:
    """
    Проверяет строки CSV и заранее загружает контуры.

    Предварительная проверка позволяет:
    - не начинать запись, если данные битые;
    - корректно посчитать val-разделение по source_name;
    - не читать contour JSON повторно во время аугментации.
    """
    tasks: List[SeedTask] = []
    seen: set[str] = set()
    skipped = 0
    duplicates = 0

    for number, raw_row in enumerate(rows, 1):
        raw_index = raw_row.get("file_index", "")
        file_index = normalize_file_index(raw_index)

        if not file_index:
            log.debug("Строка %d: пустой file_index", number)
            skipped += 1
            continue

        if not is_safe_file_index(file_index):
            log.warning(
                "Строка %d: недопустимый file_index %r — строка пропущена",
                number,
                raw_index,
            )
            skipped += 1
            continue

        if file_index in seen:
            log.warning(
                "Строка %d: дубликат file_index %r — строка пропущена",
                number,
                file_index,
            )
            duplicates += 1
            skipped += 1
            continue

        seen.add(file_index)

        if row_is_augmented(raw_row):
            log.warning(
                "Строка %d: %r уже является аугментацией — строка пропущена",
                number,
                raw_index,
            )
            skipped += 1
            continue

        image_path = find_source_image(src, file_index)
        if image_path is None:
            log.warning("Строка %d: изображение не найдено: %s.*", number, file_index)
            skipped += 1
            continue

        try:
            contour_path = resolve_under(src, file_index, CONTOUR_SUFFIX)
        except ValueError as exc:
            log.warning("Строка %d: %s", number, exc)
            skipped += 1
            continue

        if not os.path.isfile(contour_path):
            log.warning("Строка %d: contour JSON не найден: %s", number, contour_path)
            skipped += 1
            continue

        points, meta = load_contour(contour_path)
        if points is None or meta is None:
            log.warning("Строка %d: не удалось прочитать контур: %s", number, contour_path)
            skipped += 1
            continue

        row = dict(raw_row)
        row["file_index"] = file_index
        source = source_from_meta(meta, file_index)

        tasks.append(SeedTask(
            row=row,
            file_index=file_index,
            source=source,
            image_path=image_path,
            contour_path=contour_path,
            points=points,
            meta=meta,
        ))

    return PlanningResult(tasks=tasks, skipped=skipped, duplicates=duplicates)


def build_split(sources: Sequence[str], fraction: float, seed: int) -> SplitPlan:
    """Делит исходные фотографии на train/val детерминированно."""
    unique = sorted(set(sources))
    total = len(unique)

    if total == 0 or fraction <= 0.0:
        return SplitPlan(frozenset(), total)

    n_val = int(round(total * fraction))
    if total > 1:
        n_val = max(1, n_val)
        n_val = min(n_val, total - 1)
    else:
        n_val = 0

    rng = derive_rng(seed, "split")
    order = rng.permutation(total)
    val_sources = frozenset(unique[int(index)] for index in order[:n_val])

    return SplitPlan(val_sources, total)


# ---------------------------- командная строка ----------------------------

def _finite_pair(
    ap: argparse.ArgumentParser,
    values: Sequence[float],
    name: str,
) -> Tuple[float, float]:
    pair = ordered_float_pair(values)
    if not all(math.isfinite(value) for value in pair):
        ap.error(f"{name}: границы должны быть конечными числами")
    return pair


def parse_args(argv: Optional[Sequence[str]] = None) -> Settings:
    ap = argparse.ArgumentParser(
        description=(
            "Аугментация нарезанных семян с сохранением формата: "
            "jpg + _contour.json + seeds.csv"
        )
    )

    # Совместимость со старым вариантом запуска: [src] [csv_name] [dst]
    ap.add_argument("positionals", nargs="*", metavar="PATH", help="src [csv_name] [dst]")

    ap.add_argument("--src", help="Исходная папка с оригиналами (по умолчанию out)")
    ap.add_argument("--csv", dest="csv_name", help="Имя CSV внутри src или прямой путь к CSV")
    ap.add_argument("--dst", help="Целевая папка (по умолчанию out_aug)")

    ap.add_argument(
        "--n-aug", type=int, default=N_AUG,
        help="Сколько копий на семечку делать; 0 — только оригиналы",
    )
    ap.add_argument(
        "--val-fraction", type=float, default=VAL_FRACTION,
        help="Доля исходных фото под val, 0..1; 0 — без val",
    )
    ap.add_argument("--seed", type=int, default=SEED, help="Random seed")
    ap.add_argument(
        "--pad-default", type=int, default=PAD_DEFAULT,
        help="Padding для JSON, если его нет в исходном contour",
    )

    ap.add_argument(
        "--jpeg-quality", type=int, nargs=2, default=list(JPEG_QUALITY),
        metavar=("MIN", "MAX"), help="Диапазон JPEG quality",
    )
    ap.add_argument(
        "--alpha", type=float, nargs=2, default=list(ALPHA_RANGE),
        metavar=("MIN", "MAX"), help="Диапазон контраста",
    )
    ap.add_argument(
        "--beta", type=float, nargs=2, default=list(BETA_RANGE),
        metavar=("MIN", "MAX"), help="Диапазон яркости",
    )
    ap.add_argument(
        "--gamma", type=float, nargs=2, default=list(GAMMA_RANGE),
        metavar=("MIN", "MAX"), help="Диапазон гаммы",
    )
    ap.add_argument(
        "--wb-range", type=float, default=WB_RANGE,
        help="Плюс-минус сдвиг баланса белого по каналам, 0..1",
    )
    ap.add_argument(
        "--gradient-max", type=float, default=GRADIENT_MAX,
        help="Максимальная амплитуда градиента освещения, 0..1",
    )
    ap.add_argument(
        "--noise-max", type=float, default=NOISE_MAX,
        help="Максимальная сигма гауссова шума",
    )

    ap.add_argument(
        "--no-photometric", action="store_true",
        help="Только геометрия, без фотометрических искажений",
    )
    ap.add_argument(
        "--clean-dst", action="store_true",
        help="Удалить целевую папку перед запуском",
    )
    ap.add_argument(
        "--no-clean-stale", dest="clean_stale", action="store_false",
        help="Не удалять старые <file_index>_augNN.* в целевой папке",
    )
    ap.add_argument(
        "--log-level", type=str.upper, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Уровень логирования",
    )

    ns = ap.parse_args(argv)
    positionals = ns.positionals

    if len(positionals) > 3:
        ap.error("Допустимо не более трёх позиционных аргументов: src [csv_name] [dst]")

    src = ns.src if ns.src is not None else (positionals[0] if positionals else SRC_DIR)
    csv_explicit = ns.csv_name is not None or len(positionals) > 1
    csv_name = (
        ns.csv_name
        if ns.csv_name is not None
        else (positionals[1] if len(positionals) > 1 else CSV_NAME)
    )
    dst = ns.dst if ns.dst is not None else (positionals[2] if len(positionals) > 2 else DST_DIR)

    if not src.strip():
        ap.error("Не задана исходная папка")
    if not dst.strip():
        ap.error("Не задана целевая папка")

    if ns.n_aug < 0:
        ap.error("--n-aug должен быть >= 0")
    if not math.isfinite(ns.val_fraction) or not 0.0 <= ns.val_fraction <= 1.0:
        ap.error("--val-fraction должен быть в диапазоне [0, 1]")
    if ns.pad_default < 0:
        ap.error("--pad-default должен быть >= 0")

    jpeg = ordered_int_pair(ns.jpeg_quality)
    if jpeg[0] < 1 or jpeg[1] > 100:
        ap.error("--jpeg-quality должен быть в диапазоне [1, 100]")

    alpha = _finite_pair(ap, ns.alpha, "--alpha")
    if alpha[0] <= 0.0:
        ap.error("Нижняя граница --alpha должна быть > 0")

    beta = _finite_pair(ap, ns.beta, "--beta")
    gamma = _finite_pair(ap, ns.gamma, "--gamma")
    if gamma[0] <= 0.0:
        ap.error("Нижняя граница --gamma должна быть > 0")

    wb_range = abs(float(ns.wb_range))
    if not math.isfinite(wb_range) or wb_range > 1.0:
        ap.error("--wb-range должен быть в диапазоне [0, 1]")

    gradient_max = abs(float(ns.gradient_max))
    if not math.isfinite(gradient_max) or gradient_max > 1.0:
        ap.error("--gradient-max должен быть в диапазоне [0, 1]")

    noise_max = abs(float(ns.noise_max))
    if not math.isfinite(noise_max):
        ap.error("--noise-max должен быть конечным числом")

    return Settings(
        src=src,
        csv_name=csv_name,
        dst=dst,
        csv_explicit=csv_explicit,
        n_aug=ns.n_aug,
        val_fraction=float(ns.val_fraction),
        seed=ns.seed,
        pad_default=ns.pad_default,
        jpeg_quality=jpeg,
        alpha_range=alpha,
        beta_range=beta,
        gamma_range=gamma,
        wb_range=wb_range,
        gradient_max=gradient_max,
        noise_max=noise_max,
        no_photometric=ns.no_photometric,
        clean_dst=ns.clean_dst,
        clean_stale=ns.clean_stale,
        log_level=ns.log_level,
    )


# ---------------------------- основной код ----------------------------

def _same_path(first: str, second: str) -> bool:
    return os.path.normcase(os.path.abspath(first)) == os.path.normcase(
        os.path.abspath(second)
    )


def _is_within(path: str, parent: str) -> bool:
    """True, если path находится внутри parent (или совпадает с ним)."""
    path_real = os.path.realpath(path)
    parent_real = os.path.realpath(parent)
    try:
        common = os.path.commonpath([path_real, parent_real])
    except ValueError:
        return False
    return os.path.normcase(common) == os.path.normcase(parent_real)


def _is_dangerous_clean_target(path: str) -> bool:
    """Не позволяем --clean-dst удалить корень, домашнюю папку или cwd."""
    real = os.path.realpath(path)
    if os.path.dirname(real) == real:
        return True
    if _same_path(real, os.path.expanduser("~")):
        return True
    if _same_path(real, os.getcwd()):
        return True
    return False


def _validate_destination(src: str, dst: str, clean_dst: bool) -> Optional[str]:
    """Проверяет безопасность dst до чтения CSV и удаления папки."""
    if clean_dst:
        if _same_path(src, dst):
            return "--clean-dst нельзя использовать, если src и dst совпадают"
        if _is_within(src, dst):
            return (
                f"--clean-dst удалил бы исходную папку: "
                f"src={src} находится внутри dst={dst}"
            )
        if _is_dangerous_clean_target(dst):
            return (
                "--clean-dst запрещён для корня, домашней папки "
                f"и текущего каталога: {dst}"
            )
        if os.path.exists(dst) and not os.path.isdir(dst):
            return f"Целевой путь существует и не является папкой: {dst}"
    elif _same_path(src, dst):
        log.warning(
            "src и dst совпадают: аугментации будут записаны рядом с оригиналами"
        )

    return None


def _prepare_destination(dst: str, clean_dst: bool) -> bool:
    """Удаляет dst при --clean-dst (уже после проверки CSV) и создаёт папку."""
    if clean_dst and os.path.exists(dst):
        log.info("Удаляю целевую папку: %s", dst)
        try:
            shutil.rmtree(dst)
        except Exception as exc:
            log.error("Не удалось удалить %s: %s", dst, exc)
            return False

    try:
        os.makedirs(dst, exist_ok=True)
    except Exception as exc:
        log.error("Не удалось создать целевую папку %s: %s", dst, exc)
        return False

    return True


def run(settings: Settings) -> int:
    """Выполняет аугментацию. Возвращает код завершения процесса."""
    started = time.perf_counter()

    src = os.path.abspath(settings.src)
    dst = os.path.abspath(settings.dst)

    if not os.path.isdir(src):
        log.error("Исходная папка не найдена: %s", src)
        return 1

    destination_error = _validate_destination(src, dst, settings.clean_dst)
    if destination_error is not None:
        log.error("%s", destination_error)
        return 1

    try:
        csv_path = find_csv(src, settings.csv_name, settings.csv_explicit)
    except Exception as exc:
        log.error("%s", exc)
        return 1

    log.info("CSV: %s", csv_path)

    out_csv = os.path.join(dst, "seeds.csv")
    if _same_path(out_csv, csv_path):
        log.error(
            "Выходной CSV %s перезаписал бы исходный CSV %s. "
            "Укажите другую целевую папку или используйте исходный файл не 'seeds.csv'.",
            out_csv,
            csv_path,
        )
        return 1

    if os.path.isfile(out_csv):
        log.warning("Выходной CSV %s уже существует и будет перезаписан", out_csv)

    try:
        fields, rows = read_csv(csv_path)
    except Exception as exc:
        log.error("Ошибка чтения CSV: %s", exc)
        return 1

    if not rows:
        log.error("CSV не содержит строк: %s", csv_path)
        return 1

    planning = plan_tasks(src, rows)
    stats = RunStats(skipped=planning.skipped, duplicates=planning.duplicates)

    log.info(
        "Строк в CSV: %d, пригодно: %d, пропущено при проверке: %d, дубликатов: %d",
        len(rows),
        len(planning.tasks),
        planning.skipped,
        planning.duplicates,
    )

    if not planning.tasks:
        log.error("Нет ни одной пригодной строки для обработки")
        return 1

    split = build_split(
        [task.source for task in planning.tasks],
        settings.val_fraction,
        settings.seed,
    )

    log.info(
        "Источников фото: %d, train: %d, val: %d",
        split.source_count,
        split.train_count,
        len(split.val_sources),
    )

    if settings.val_fraction > 0.0 and split.source_count > 1 and not split.val_sources:
        log.warning("val не сформирован: проверьте --val-fraction")
    if settings.val_fraction >= 1.0 and split.source_count > 1:
        log.warning(
            "--val-fraction=1.0 ограничен: оставлен хотя бы один train-источник"
        )
    if settings.no_photometric and settings.n_aug > len(ALL_GEOMETRIES):
        log.warning(
            "--no-photometric и n_aug > %d дадут одинаковые копии: "
            "геометрия начнёт повторяться",
            len(ALL_GEOMETRIES),
        )

    # Папку удаляем только после успешной проверки CSV, контуров и плана.
    if not _prepare_destination(dst, settings.clean_dst):
        return 1

    out_fields = list(fields)
    for name in EXTRA_FIELDS:
        if name not in out_fields:
            out_fields.append(name)

    out_rows: List[Dict[str, Any]] = []
    total = len(planning.tasks)
    cleanup_plan: Dict[Tuple[str, str], set[str]] = {}

    for number, task in enumerate(planning.tasks, 1):
        row = task.row
        file_index = task.file_index
        source = task.source
        split_name = "val" if split.is_val(source) else "train"

        source_ext = os.path.splitext(task.image_path)[1].lower() or ".jpg"

        try:
            dst_image = resolve_under(dst, file_index, source_ext)
            dst_json = resolve_under(dst, file_index, CONTOUR_SUFFIX)

            atomic_copy(task.image_path, dst_image)
            atomic_copy(task.contour_path, dst_json)
        except Exception as exc:
            log.error(
                "Строка %d (%s): не удалось скопировать оригинал: %s",
                number,
                file_index,
                exc,
            )
            stats.errors += 1
            continue

        row_original = dict(row)
        row_original.update({
            "source": source,
            "split": split_name,
            "is_aug": 0,
            "aug_of": "",
        })

        out_rows.append(row_original)
        stats.originals += 1

        target_dir = os.path.dirname(dst_image)
        stem = file_index.rsplit("/", 1)[-1]

        # Val не аугментируем. При n_aug=0 чистим старые аугментации,
        # чтобы папка соответствовала текущему запуску.
        if split_name == "val" or settings.n_aug == 0:
            if settings.clean_stale:
                cleanup_plan[(target_dir, stem)] = set()
            if number % 100 == 0 or number == total:
                log.info("Обработано строк: %d/%d", number, total)
            continue

        img = ensure_bgr(imread_unicode(task.image_path))
        if img is None:
            log.warning(
                "Строка %d (%s): не удалось прочитать изображение для аугментации",
                number,
                file_index,
            )
            stats.errors += 1
            continue

        padding = parse_int(task.meta.get("padding"))
        padding = settings.pad_default if padding is None else max(0, padding)

        angle_old = parse_float(row.get("angle_deg"))
        rng = derive_rng(settings.seed, "aug", file_index)
        ops = build_ops(settings.n_aug)

        base_name = stem
        keep_names: set[str] = set()
        for index in range(1, len(ops) + 1):
            base = f"{base_name}_aug{index:02d}"
            keep_names.add(f"{base}.jpg")
            keep_names.add(f"{base}{CONTOUR_SUFFIX}")

        if settings.clean_stale:
            cleanup_plan[(target_dir, stem)] = keep_names

        for index, op in enumerate(ops, 1):
            new_file_index = f"{file_index}_aug{index:02d}"

            try:
                aug_img, aug_points, matrix = geometric(img, task.points, op)

                if not settings.no_photometric:
                    aug_img = photometric(aug_img, settings, rng)

                if settings.no_photometric:
                    quality = settings.jpeg_quality[1]
                else:
                    quality = int(rng.integers(
                        settings.jpeg_quality[0],
                        settings.jpeg_quality[1] + 1,
                    ))

                aug_image_path = resolve_under(dst, new_file_index, ".jpg")
                aug_json_path = resolve_under(dst, new_file_index, CONTOUR_SUFFIX)

                payload = build_contour_payload(
                    meta=task.meta,
                    row=row,
                    new_file_index=new_file_index,
                    original_file_index=file_index,
                    source=source,
                    points=aug_points,
                    img=aug_img,
                    padding=padding,
                    op=op,
                )

                write_augmentation(
                    aug_image_path,
                    aug_json_path,
                    aug_img,
                    payload,
                    quality,
                )
            except Exception as exc:
                log.warning(
                    "Строка %d (%s): аугментация %s не создана: %s",
                    number,
                    file_index,
                    new_file_index,
                    exc,
                )
                stats.errors += 1
                continue

            row_aug = dict(row)
            row_aug["file_index"] = new_file_index

            if angle_old is not None:
                new_angle = transform_angle(angle_old, matrix)
                if new_angle is not None:
                    row_aug["angle_deg"] = f"{new_angle:.3f}"

            update_image_size_fields(row_aug, out_fields, aug_img)

            row_aug.update({
                "source": source,
                "split": "train",
                "is_aug": 1,
                "aug_of": file_index,
            })

            out_rows.append(row_aug)
            stats.augmentations += 1

        if number % 100 == 0 or number == total:
            log.info("Обработано строк: %d/%d", number, total)

    if not out_rows:
        log.error("Не удалось обработать ни одной строки; CSV не записан")
        return 1

    if settings.clean_stale and cleanup_plan:
        clean_stale_outputs(cleanup_plan)

    try:
        write_csv(out_csv, out_fields, out_rows)
    except Exception as exc:
        log.error("Не удалось записать CSV %s: %s", out_csv, exc)
        return 1

    elapsed = time.perf_counter() - started
    val_preview = preview(sorted(split.val_sources)) or "нет"

    log.info("%s", stats.summary())
    log.info("Фото в val (%d): %s", len(split.val_sources), val_preview)
    log.info("Результат: %s (%.1f c)", dst, elapsed)

    print(stats.summary())
    print(f"Фото в val ({len(split.val_sources)}): {val_preview}")
    print(f"Результат: {dst}")

    return 0 if stats.originals > 0 else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    settings = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format=LOG_FORMAT,
        force=True,
    )

    try:
        return run(settings)
    except KeyboardInterrupt:
        raise
    except Exception:
        log.exception("Непредвиденная ошибка")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Прервано пользователем")
        sys.exit(130)
