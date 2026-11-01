"""Batch processing: segment multiple images without GUI."""
from __future__ import annotations

import os
import sys

from core.calibration import format_calibration_message, try_calibrate_ruler
from core.export import export_seed_items, make_source_name, select_items_for_export
from core.io_utils import imread_unicode
from core.measurement import count_seed_statuses
from core.params import get_default_params, normalize_params
from core.segmentation import process_image


def collect_image_paths(args: list[str]) -> list[str]:
    """Collect .jpg/.jpeg files from file/directory arguments."""
    paths: list[str] = []
    for arg in args:
        arg = os.path.abspath(arg)
        if os.path.isfile(arg):
            if os.path.splitext(arg)[1].lower() in (".jpg", ".jpeg"):
                paths.append(arg)
        elif os.path.isdir(arg):
            for name in sorted(os.listdir(arg)):
                if os.path.splitext(name)[1].lower() in (".jpg", ".jpeg"):
                    paths.append(os.path.join(arg, name))
        else:
            print(f"[ПРЕДУПРЕЖДЕНИЕ] Путь не найден, пропущен: {arg}")
    return paths


def batch_process_file(image_path: str, params: dict, out_dir: str) -> None:
    img = imread_unicode(image_path)
    if img is None:
        print(f"[ОШИБКА] Не удалось прочитать: {image_path}")
        return
    source_name = make_source_name(image_path)
    h, w = img.shape[:2]
    print(f"  Обработка: {os.path.basename(image_path)} ({w}x{h})")

    mm_per_px, calibration_error = try_calibrate_ruler(img)
    if mm_per_px is not None:
        print(f"    Калибровка: {format_calibration_message(mm_per_px)}")
    else:
        print(f"    [ПРЕДУПРЕЖДЕНИЕ] Калибровка не удалась: {calibration_error}")
        print("    Продолжаю без масштаба (замеры в px).")

    p = dict(params)
    p["mm_per_px"] = mm_per_px
    p = normalize_params(p)
    try:
        result = process_image(img, p)
    except Exception as exc:
        print(f"    [ОШИБКА] Обработка не удалась: {exc}")
        return

    items = result["items"]
    print(f"    Найдено объектов: {len(items)}")
    selected = select_items_for_export(items, save_only_focused=p["save_only_focused"], skip_blurry=False)
    if not selected:
        print("    Нет объектов для сохранения после фильтрации.")
        return
    try:
        summary = export_seed_items(img, selected, out_dir, source_name,
                                    padding=int(p["padding"]), mm_per_px=mm_per_px)
    except Exception as exc:
        print(f"    [ОШИБКА] Не удалось сохранить фрагменты: {exc}")
        return
    if summary["csv_error"]:
        print(f"    [ОШИБКА] Не удалось обновить CSV: {summary['csv_error']}")
    counts = count_seed_statuses(items)
    print(f"    Сохранено: {summary['saved']} шт. | фокус: {counts['focused']}, "
          f"размыто: {counts['blurry']}, всего: {counts['total']}")
    if summary["contour_errors"]:
        print(f"    Ошибки контуров: {summary['contour_errors']}")


def batch_main(paths: list[str], out_dir: str) -> None:
    image_files = collect_image_paths(paths)
    if not image_files:
        print("Не найдено ни одного .jpg файла по указанным путям.")
        sys.exit(1)
    print(f"Найдено файлов для обработки: {len(image_files)}")
    print(f"Папка результатов: {out_dir}")
    print("-" * 60)
    os.makedirs(out_dir, exist_ok=True)
    params = get_default_params()
    for i, path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}]")
        batch_process_file(path, params, out_dir)
    print("-" * 60)
    print(f"Пакетная обработка завершена. Результат в: {out_dir}")