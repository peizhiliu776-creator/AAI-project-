from __future__ import annotations

from pathlib import Path
import random
from typing import Iterable

import pandas as pd

from .config import RAW_DATA_DIR, RAW_EXCLUDE_PARTS
from .progress import log_step, progress


EXPECTED_COLUMNS = {
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "start_station_name",
    "start_station_id",
    "end_station_name",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "member_casual",
}


def discover_csv_files(raw_dir: Path = RAW_DATA_DIR) -> list[Path]:
    """Find raw Citi Bike CSVs, ignoring generated outputs and macOS sidecar files."""
    files: list[Path] = []
    for path in raw_dir.rglob("*.csv"):
        if not path.is_file():
            continue
        parts = set(path.relative_to(raw_dir).parts)
        if parts & RAW_EXCLUDE_PARTS:
            continue
        if path.name.startswith("._") or "__MACOSX" in path.parts:
            continue
        files.append(path)
    return sorted(files)


def _normalize_columns(columns: Iterable[str]) -> list[str]:
    return [c.strip().lower().replace(" ", "_").replace("-", "_") for c in columns]


def _count_csv_data_rows(path: Path) -> int:
    with path.open("rb") as f:
        return max(sum(1 for _ in f) - 1, 0)


def _read_csv_random_sample(path: Path, sample_rows_per_file: int, random_state: int = 42) -> pd.DataFrame:
    total_rows = _count_csv_data_rows(path)
    if total_rows <= sample_rows_per_file:
        return pd.read_csv(path, low_memory=False)

    rng = random.Random(random_state + abs(hash(str(path))) % 1_000_000)
    keep_data_rows = set(rng.sample(range(1, total_rows + 1), sample_rows_per_file))
    skiprows = lambda idx: idx != 0 and idx not in keep_data_rows
    return pd.read_csv(path, skiprows=skiprows, low_memory=False)


def load_trip_csvs(
    paths: list[Path] | None = None,
    sample_rows: int | None = None,
    sample_rows_per_file: int | None = None,
    total_sample_rows: int | None = None,
) -> pd.DataFrame:
    paths = paths or discover_csv_files()
    if not paths:
        raise FileNotFoundError(
            f"No raw Citi Bike CSV files found under {RAW_DATA_DIR}. Put CSV files in this workspace."
        )

    frames: list[pd.DataFrame] = []
    log_step(f"Loading {len(paths)} Citi Bike CSV files")
    per_file_samples: dict[Path, int] = {}
    if total_sample_rows:
        base = max(total_sample_rows // len(paths), 1)
        remainder = total_sample_rows % len(paths)
        for i, path in enumerate(paths):
            per_file_samples[path] = base + (1 if i < remainder else 0)
        log_step(f"Balanced sampling target: {total_sample_rows} total rows, about {base} per CSV")

    for path in progress(paths, total=len(paths), desc="Loading CSV files"):
        rows_for_file = per_file_samples.get(path, sample_rows_per_file)
        if rows_for_file:
            df = _read_csv_random_sample(path, rows_for_file)
        else:
            df = pd.read_csv(path, nrows=sample_rows, low_memory=False)
        df.columns = _normalize_columns(df.columns)
        missing = EXPECTED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing expected columns: {sorted(missing)}")
        df["source_file"] = path.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
