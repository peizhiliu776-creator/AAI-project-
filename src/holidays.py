from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PROCESSED_DIR, ensure_dirs


NYC_HOLIDAYS = [
    ("2025-01-01", "New Year's Day"),
    ("2025-01-20", "Martin Luther King Jr. Day"),
    ("2025-02-17", "Presidents' Day"),
    ("2025-05-26", "Memorial Day"),
    ("2025-06-19", "Juneteenth"),
    ("2025-07-04", "Independence Day"),
    ("2025-09-01", "Labor Day"),
    ("2025-10-13", "Italian Heritage/Indigenous Peoples' Day"),
    ("2025-11-04", "Election Day"),
    ("2025-11-11", "Veterans Day"),
    ("2025-11-27", "Thanksgiving Day"),
    ("2025-12-25", "Christmas Day"),
    ("2026-01-01", "New Year's Day"),
    ("2026-01-19", "Martin Luther King Jr. Day"),
    ("2026-02-16", "Presidents' Day"),
    ("2026-05-25", "Memorial Day"),
    ("2026-06-19", "Juneteenth"),
    ("2026-07-03", "Independence Day observed"),
    ("2026-07-04", "Independence Day"),
    ("2026-09-07", "Labor Day"),
    ("2026-10-12", "Italian Heritage/Indigenous Peoples' Day"),
    ("2026-11-03", "Election Day"),
    ("2026-11-11", "Veterans Day"),
    ("2026-11-26", "Thanksgiving Day"),
    ("2026-12-25", "Christmas Day"),
]


def build_nyc_holiday_table() -> pd.DataFrame:
    holidays = pd.DataFrame(NYC_HOLIDAYS, columns=["date", "holiday_name"])
    holidays["date"] = pd.to_datetime(holidays["date"]).dt.date
    holidays["source"] = "NYC official holiday calendar"
    return holidays


def save_nyc_holiday_table(path: Path | None = None) -> Path:
    ensure_dirs()
    path = path or PROCESSED_DIR / "nyc_holidays.csv"
    holidays = build_nyc_holiday_table()
    holidays.to_csv(path, index=False)
    return path
