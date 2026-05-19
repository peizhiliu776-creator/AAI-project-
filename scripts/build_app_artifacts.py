from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import PROCESSED_DIR, ensure_dirs
from src.feature_engineering import infer_area_label
from src.pipeline import build_app_aggregates, build_app_station_context


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lightweight Streamlit app artifacts from processed data.")
    parser.add_argument("--recent-hours", type=int, default=168, help="Recent hours per station/rider kept for app prediction context. Use 0 to keep all rows.")
    args = parser.parse_args()
    recent_hours = None if args.recent_hours == 0 else args.recent_hours
    ensure_dirs()
    station_hour_path = PROCESSED_DIR / "station_hour_demand.parquet"
    area_hour_path = PROCESSED_DIR / "area_hour_demand.parquet"
    trips_path = PROCESSED_DIR / "clean_trips.parquet"
    if station_hour_path.exists():
        station_hour = pd.read_parquet(station_hour_path)
        if "area_label" not in station_hour.columns:
            station_hour["area_label"] = infer_area_label(station_hour["station_lat"], station_hour["station_lng"])
        build_app_station_context(station_hour, recent_hours=recent_hours).to_parquet(PROCESSED_DIR / "app_station_context.parquet", index=False)
        rider_summary = station_hour.groupby("rider_type", as_index=False).agg(
            trips=("demand", "sum"),
            avg_duration_min=("avg_duration_min", "mean"),
            avg_distance_miles=("avg_trip_distance_miles", "mean"),
        )
        rider_summary.to_csv(PROCESSED_DIR / "app_rider_summary.csv", index=False)
    if area_hour_path.exists():
        area_hour = pd.read_parquet(area_hour_path)
        if "area_label" not in area_hour.columns:
            area_hour["area_label"] = infer_area_label(area_hour["station_lat"], area_hour["station_lng"])
        area_hour.to_parquet(PROCESSED_DIR / "app_area_hour.parquet", index=False)
    if station_hour_path.exists() and area_hour_path.exists():
        for filename, frame in build_app_aggregates(station_hour, area_hour).items():
            frame.to_parquet(PROCESSED_DIR / filename, index=False)
    if trips_path.exists():
        trips = pd.read_parquet(
            trips_path,
            columns=[
                "ride_id",
                "started_at",
                "start_station_name",
                "end_station_name",
                "start_lat",
                "start_lng",
                "end_lat",
                "end_lng",
                "rider_type",
            ],
        )
        trips["hour"] = trips["started_at"].dt.hour
        trips["day_of_week"] = trips["started_at"].dt.dayofweek
        flow = (
            trips.dropna(subset=["end_station_name", "end_lat", "end_lng"])
            .groupby(["day_of_week", "hour", "rider_type", "start_station_name", "end_station_name"], as_index=False)
            .agg(
                flow_count=("ride_id", "count"),
                start_lat=("start_lat", "median"),
                start_lng=("start_lng", "median"),
                end_lat=("end_lat", "median"),
                end_lng=("end_lng", "median"),
            )
        )
        flow["rank"] = flow.groupby(["day_of_week", "hour", "rider_type"])["flow_count"].rank(method="first", ascending=False)
        flow.query("rank <= 30").drop(columns=["rank"]).to_parquet(PROCESSED_DIR / "app_flow_summary.parquet", index=False)
    print("Built app artifacts in processed/")


if __name__ == "__main__":
    main()
