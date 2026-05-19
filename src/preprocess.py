from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MAX_TRIP_HOURS, MIN_TRIP_MINUTES


def haversine_miles(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    radius_miles = 3958.8
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * radius_miles * np.arcsin(np.sqrt(a))


def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Citi Bike trips and preserve fields needed for modeling and maps."""
    data = df.copy()
    data["started_at"] = pd.to_datetime(data["started_at"], errors="coerce")
    data["ended_at"] = pd.to_datetime(data["ended_at"], errors="coerce")

    required = [
        "ride_id",
        "started_at",
        "ended_at",
        "start_station_id",
        "start_station_name",
        "start_lat",
        "start_lng",
        "member_casual",
        "rideable_type",
    ]
    data = data.dropna(subset=required)
    data = data.drop_duplicates(subset=["ride_id"])

    for col in ["start_lat", "start_lng", "end_lat", "end_lng"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["start_lat", "start_lng"])

    coord_mask = (
        data["start_lat"].between(40.0, 41.2)
        & data["start_lng"].between(-75.0, -73.0)
        & (data["end_lat"].isna() | data["end_lat"].between(40.0, 41.2))
        & (data["end_lng"].isna() | data["end_lng"].between(-75.0, -73.0))
    )
    data = data.loc[coord_mask].copy()

    data["duration_min"] = (data["ended_at"] - data["started_at"]).dt.total_seconds() / 60
    data = data[data["duration_min"].between(MIN_TRIP_MINUTES, MAX_TRIP_HOURS * 60)].copy()

    data["rider_type"] = data["member_casual"].str.lower().str.strip()
    data["rideable_type"] = data["rideable_type"].str.lower().str.strip()
    for col in [
        "ride_id",
        "start_station_id",
        "end_station_id",
        "start_station_name",
        "end_station_name",
        "source_file",
        "member_casual",
    ]:
        if col in data.columns:
            data[col] = data[col].astype("string")
    data["trip_distance_miles"] = haversine_miles(
        data["start_lat"], data["start_lng"], data["end_lat"], data["end_lng"]
    )
    data["hour_start"] = data["started_at"].dt.floor("h")
    return data.reset_index(drop=True)
