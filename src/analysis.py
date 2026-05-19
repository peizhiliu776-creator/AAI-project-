from __future__ import annotations

import pandas as pd


def overview_metrics(trips: pd.DataFrame, station_hour: pd.DataFrame) -> dict[str, object]:
    peak = station_hour.groupby("hour")["demand"].sum().idxmax()
    top_station = station_hour.groupby("start_station_name")["demand"].sum().idxmax()
    return {
        "total_trips": int(len(trips)),
        "stations": int(station_hour["start_station_id"].nunique()),
        "date_min": trips["started_at"].min(),
        "date_max": trips["started_at"].max(),
        "peak_hour": int(peak),
        "top_station": str(top_station),
    }


def rider_type_summary(trips: pd.DataFrame) -> pd.DataFrame:
    return (
        trips.groupby("rider_type", as_index=False)
        .agg(
            trips=("ride_id", "count"),
            avg_duration_min=("duration_min", "mean"),
            median_duration_min=("duration_min", "median"),
            avg_distance_miles=("trip_distance_miles", "mean"),
        )
        .sort_values("trips", ascending=False)
    )


def station_hotspots(station_hour: pd.DataFrame, rider_type: str = "all", top_k: int = 20) -> pd.DataFrame:
    data = station_hour if rider_type == "all" else station_hour[station_hour["rider_type"] == rider_type]
    return (
        data.groupby(["start_station_id", "start_station_name", "station_lat", "station_lng"], as_index=False)["demand"]
        .sum()
        .sort_values("demand", ascending=False)
        .head(top_k)
    )


def area_hotspots(area_hour: pd.DataFrame, rider_type: str = "all", top_k: int = 20) -> pd.DataFrame:
    data = area_hour if rider_type == "all" else area_hour[area_hour["rider_type"] == rider_type]
    return data.groupby(["grid_cell", "region_cluster"], as_index=False)["demand"].sum().sort_values("demand", ascending=False).head(top_k)

