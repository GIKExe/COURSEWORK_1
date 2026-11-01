"""Robust CSV reading and writing with encoding auto-detection."""
from __future__ import annotations

import csv
import os
import re

from core.models import DEFAULT_CSV_FIELDS, SKIP_CSV_FIELDS


def read_csv_robust(path: str) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV file, auto-detecting encoding and delimiter.

    Returns (fieldnames, rows).
    """
    encodings = ("utf-8-sig", "utf-8", "cp1251", "latin-1")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding, newline="") as fh:
                sample = fh.read(8192)
                fh.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                except Exception:
                    dialect = csv.excel
                reader = csv.DictReader(fh, dialect=dialect)
                fieldnames = list(reader.fieldnames or [])
                rows = [
                    {k: ("" if v is None else str(v).strip()) for k, v in row.items() if k is not None}
                    for row in reader
                ]
                return fieldnames, rows
        except (UnicodeDecodeError, csv.Error) as exc:
            last_error = exc
    raise last_error or ValueError(f"Не удалось прочитать CSV: {path}")


def write_csv(path: str, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write rows to a CSV file with UTF-8 BOM."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def merge_fieldnames(
    existing: list[str],
    new_rows: list[dict[str, str]],
    base_fields: list[str] | None = None,
) -> list[str]:
    """Build a unified column list: base fields first, then existing, then new."""
    names: list[str] = []
    for source in (base_fields or DEFAULT_CSV_FIELDS, existing):
        for name in source:
            if name not in names and name not in SKIP_CSV_FIELDS:
                names.append(name)
    for row in new_rows:
        for name in row:
            if name not in names and name not in SKIP_CSV_FIELDS:
                names.append(name)
    return names


def update_seeds_csv(
    path: str,
    new_rows: list[dict[str, str]],
    group_prefix: str,
) -> None:
    """Update seeds.csv: remove old rows for *group_prefix*, append *new_rows*."""
    existing_rows: list[dict[str, str]] = []
    existing_fields: list[str] = []
    if os.path.isfile(path):
        try:
            existing_fields, existing_rows = read_csv_robust(path)
        except Exception:
            pass

    pattern = re.compile(r"^" + re.escape(group_prefix) + r"_\d+$")
    kept = [r for r in existing_rows if not pattern.match(str(r.get("file_index", "")).strip())]

    fieldnames = merge_fieldnames(existing_fields, new_rows)
    write_csv(path, fieldnames, kept + new_rows)