"""Type definitions and constants shared across the pipeline."""
from __future__ import annotations

from enum import Enum
from typing import Any, TypedDict

import numpy as np


class SeedStatus(str, Enum):
    FOCUSED = "focused"
    TRANSITION = "transition"
    BLURRY = "blurry"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


STATUS_COLORS: dict[SeedStatus, tuple[int, int, int]] = {
    SeedStatus.FOCUSED: (0, 200, 0),
    SeedStatus.TRANSITION: (0, 220, 255),
    SeedStatus.BLURRY: (0, 0, 255),
    SeedStatus.UNKNOWN: (80, 160, 255),
}

STATUS_NAMES_RU: dict[SeedStatus, str] = {
    SeedStatus.FOCUSED: "фокус",
    SeedStatus.TRANSITION: "транзит",
    SeedStatus.BLURRY: "размыто",
    SeedStatus.UNKNOWN: "неизвестно",
}

IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

DEFAULT_CSV_FIELDS: list[str] = [
    "file_index", "index", "status", "color_hex", "edge_width_px",
    "length_px", "width_px", "angle_deg", "area_px",
    "has_groove",
    "mm_per_px",
]

EXTRA_EDITABLE_FIELDS: list[str] = ["thickness_mm", "mass_g", "shape"]

SKIP_CSV_FIELDS: frozenset[str] = frozenset({
    "filename", "path", "source_path", "length_mm", "width_mm",
})

AUG_EXTRA_FIELDS: list[str] = ["source", "split", "is_aug", "aug_of"]


def is_augmentation(record: dict[str, str]) -> bool:
    return str(record.get("is_aug", "0")).strip() in ("1", "True", "true")


class SeedCandidate(TypedDict, total=False):
    """Intermediate detection result before measurement."""
    bbox: tuple[int, int, int, int]
    area: float
    contour: np.ndarray
    solidity: float
    extent: float
    contrast: float
    compactness: float


class SeedItem(SeedCandidate, total=False):
    """Fully measured seed with geometry and focus status."""
    rect: Any
    box: np.ndarray
    center_x: float
    center_y: float
    length_px: float
    width_px: float
    angle_deg: float
    edge_width: float | None
    status: str
    color_hex: str
    groove: int | None
    groove_score: float | None
    index: int


class ProcessResult(TypedDict):
    items: list[SeedItem]
    annotated: np.ndarray
    mask: np.ndarray


class ExportSummary(TypedDict):
    saved: int
    selected: int
    contour_errors: int
    saved_file_indexes: list[str]
    csv_error: str | None


class SizeStats(TypedDict):
    count: int
    unit: str
    length: tuple[float, float, float]
    width: tuple[float, float, float]


class StatusCounts(TypedDict):
    total: int
    focused: int
    blurry: int
    unknown: int


class ContourPayload(TypedDict, total=False):
    file_index: str
    index: int | str
    points_space: str
    image_width: int
    image_height: int
    crop_origin_in_source: list[int]
    padding: int
    points: list[list[int]]
    bbox_in_source: list[int]
    source_name: str
    aug_of: str
    aug: dict[str, Any]