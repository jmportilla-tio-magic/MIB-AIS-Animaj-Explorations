from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np


RELATIVE_ROOT_TRANSLATION_INDICES = (0, 1, 2)
RANGE_FILTER_COLUMNS = (
    "x_main_CTRL:translateX",
    "x_main_CTRL:translateY",
    "x_main_CTRL:translateZ",
)


def apply_relative_root_translation(
    vectors: np.ndarray,
    *,
    indices: Iterable[int] = RELATIVE_ROOT_TRANSLATION_INDICES,
) -> np.ndarray:
    """Match upstream relative preprocessing for x_main_CTRL translation."""

    out = np.asarray(vectors, dtype=np.float32).copy()
    if out.ndim != 2 or len(out) == 0:
        return out
    cols = [int(i) for i in indices if 0 <= int(i) < out.shape[1]]
    if cols:
        out[:, cols] = out[:, cols] - out[0:1, cols]
    return out


def dataset_root_from_controller_h5(controller_h5: str | Path) -> Path | None:
    path = Path(controller_h5)
    for parent in path.parents:
        if parent.name in {"in_house_dataset", "prod_test_dataset"}:
            return parent
    return None


def invalid_scene_keys_from_ranges(
    ranges_csv: str | Path | None,
    *,
    threshold: float | None,
    columns: Iterable[str] = RANGE_FILTER_COLUMNS,
) -> set[tuple[str, str]]:
    if ranges_csv is None or threshold is None:
        return set()
    path = Path(ranges_csv)
    if not path.exists():
        return set()

    invalid: set[tuple[str, str]] = set()
    requested = tuple(columns)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for column in requested:
                raw = row.get(column)
                if raw is None or raw == "":
                    continue
                if float(raw) > float(threshold):
                    invalid.add((str(row.get("episode_id", "")), str(row.get("scene_id", ""))))
                    break
    return invalid
