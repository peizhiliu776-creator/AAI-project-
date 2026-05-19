from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .config import EXTERNAL_DIR, ensure_dirs


NOAA_CDO_DATA_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
DEFAULT_STATION_ID = "GHCND:USW00094728"
DEFAULT_DATATYPES = ["TAVG", "TMAX", "TMIN", "PRCP", "SNOW", "AWND"]


def fetch_noaa_daily_weather(
    start_date: str,
    end_date: str,
    token: str | None = None,
    station_id: str = DEFAULT_STATION_ID,
    datatypes: Iterable[str] = DEFAULT_DATATYPES,
) -> pd.DataFrame:
    """Fetch daily NOAA CDO data for NYC.

    Default station is Central Park / NY City. NOAA CDO requires an API token in
    the `token` argument or `NOAA_TOKEN` environment variable.
    """
    token = token or os.getenv("NOAA_TOKEN")
    if not token:
        raise ValueError("NOAA API token is required. Set NOAA_TOKEN or pass --token.")

    params: list[tuple[str, str | int]] = [
        ("datasetid", "GHCND"),
        ("stationid", station_id),
        ("startdate", start_date),
        ("enddate", end_date),
        ("limit", 1000),
        ("units", "metric"),
    ]
    params.extend(("datatypeid", item) for item in datatypes)
    rows: list[dict] = []
    offset = 1
    while True:
        response = requests.get(
            NOAA_CDO_DATA_URL,
            headers={"token": token},
            params=params + [("offset", offset)],
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("results", [])
        rows.extend(batch)
        if not batch or len(batch) < 1000:
            break
        offset += 1000

    if not rows:
        return pd.DataFrame(columns=["date", "temperature", "precipitation", "snow", "wind_speed"])

    raw = pd.DataFrame(rows)
    raw["date"] = pd.to_datetime(raw["date"]).dt.date
    wide = raw.pivot_table(index="date", columns="datatype", values="value", aggfunc="mean").reset_index()
    if "TAVG" not in wide.columns and {"TMAX", "TMIN"}.issubset(set(wide.columns)):
        wide["TAVG"] = (wide["TMAX"] + wide["TMIN"]) / 2
    rename = {"TAVG": "temperature", "PRCP": "precipitation", "SNOW": "snow", "AWND": "wind_speed"}
    wide = wide.rename(columns=rename)
    for col in rename.values():
        if col not in wide.columns:
            wide[col] = pd.NA
    return wide[["date", "temperature", "precipitation", "snow", "wind_speed"]]


def save_noaa_daily_weather(
    start_date: str,
    end_date: str,
    token: str | None = None,
    station_id: str = DEFAULT_STATION_ID,
    output_path: Path | None = None,
) -> Path:
    ensure_dirs()
    output_path = output_path or EXTERNAL_DIR / "noaa_nyc_daily_weather.csv"
    weather = fetch_noaa_daily_weather(start_date, end_date, token=token, station_id=station_id)
    weather.to_csv(output_path, index=False)
    return output_path
