from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import OUTPUTS_DIR, PROCESSED_DIR, ensure_dirs
from .data_loader import discover_csv_files, load_trip_csvs
from .feature_engineering import add_holiday_features, add_lag_features, aggregate_area_hour, aggregate_station_hour, build_station_metadata, infer_area_label, merge_weather
from .holidays import save_nyc_holiday_table
from .modeling import train_models
from .preprocess import clean_trips
from .progress import log_step
from .visualization import hourly_pattern, save_chart, top_stations, weekday_weekend_pattern


def build_flow_summary(trips: pd.DataFrame, top_per_group: int = 30) -> pd.DataFrame:
    flow = (
        trips.dropna(subset=["end_station_id", "end_station_name", "end_lat", "end_lng"])
        .assign(hour=trips["started_at"].dt.hour, day_of_week=trips["started_at"].dt.dayofweek)
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
    all_flow = (
        trips.dropna(subset=["end_station_id", "end_station_name", "end_lat", "end_lng"])
        .assign(hour=trips["started_at"].dt.hour, day_of_week=trips["started_at"].dt.dayofweek, rider_type="all")
        .groupby(["day_of_week", "hour", "rider_type", "start_station_name", "end_station_name"], as_index=False)
        .agg(
            flow_count=("ride_id", "count"),
            start_lat=("start_lat", "median"),
            start_lng=("start_lng", "median"),
            end_lat=("end_lat", "median"),
            end_lng=("end_lng", "median"),
        )
    )
    all_flow["rank"] = all_flow.groupby(["day_of_week", "hour", "rider_type"])["flow_count"].rank(method="first", ascending=False)
    return pd.concat([flow, all_flow], ignore_index=True).query("rank <= @top_per_group").drop(columns=["rank"])


def build_app_aggregates(station_hour: pd.DataFrame, area_hour: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if "area_label" not in station_hour.columns:
        station_hour = station_hour.copy()
        station_hour["area_label"] = infer_area_label(station_hour["station_lat"], station_hour["station_lng"])
    if "area_label" not in area_hour.columns:
        area_hour = area_hour.copy()
        area_hour["area_label"] = infer_area_label(area_hour["station_lat"], area_hour["station_lng"])
    station_map_hour = (
        station_hour.groupby(
            [
                "hour",
                "rider_type",
                "area_label",
                "start_station_id",
                "start_station_name",
                "station_lat",
                "station_lng",
                "grid_cell",
            ],
            as_index=False,
        )
        .agg(predicted_demand=("demand", "mean"), demand=("demand", "sum"))
    )
    area_map_hour = (
        area_hour.groupby(["hour", "rider_type", "area_label", "grid_cell", "region_cluster"], as_index=False)
        .agg(predicted_demand=("demand", "mean"), demand=("demand", "sum"), station_lat=("station_lat", "mean"), station_lng=("station_lng", "mean"))
    )
    hourly_patterns = station_hour.groupby(["hour", "rider_type", "area_label"], as_index=False)["demand"].mean()
    weekday_patterns = station_hour.groupby(["is_weekend", "rider_type", "area_label"], as_index=False)["demand"].mean()
    holiday_patterns = station_hour.groupby(["is_holiday", "rider_type", "area_label"], as_index=False)["demand"].mean()
    top_station_summary = (
        station_hour.groupby(["rider_type", "area_label", "start_station_id", "start_station_name", "station_lat", "station_lng"], as_index=False)["demand"]
        .sum()
        .sort_values(["rider_type", "area_label", "demand"], ascending=[True, True, False])
    )
    return {
        "app_station_map_hour.parquet": station_map_hour,
        "app_area_map_hour.parquet": area_map_hour,
        "app_hourly_patterns.parquet": hourly_patterns,
        "app_weekday_patterns.parquet": weekday_patterns,
        "app_holiday_patterns.parquet": holiday_patterns,
        "app_top_station_summary.parquet": top_station_summary,
    }


def build_app_station_context(station_hour: pd.DataFrame, recent_hours: int | None = 168) -> pd.DataFrame:
    keep_cols = [
        "hour_start",
        "start_station_id",
        "start_station_name",
        "rider_type",
        "demand",
        "station_lat",
        "station_lng",
        "grid_cell",
        "area_label",
        "region_cluster",
        "avg_duration_min",
        "avg_trip_distance_miles",
        "day_of_week",
        "is_holiday",
        "is_pre_holiday",
        "hour",
        "temperature",
        "precipitation",
        "snow",
        "wind_speed",
        "rolling_mean_24h",
        "rolling_mean_3h",
        "rolling_mean_6h",
        "rolling_mean_168h",
        "rolling_max_24h",
        "rolling_min_24h",
    ]
    context = station_hour[[col for col in keep_cols if col in station_hour.columns]].sort_values("hour_start")
    if not recent_hours or recent_hours <= 0:
        return context.reset_index(drop=True)
    return context.groupby(["start_station_id", "rider_type"], as_index=False).tail(recent_hours).reset_index(drop=True)


def run_pipeline(
    raw_paths: list[Path] | None = None,
    sample_rows: int | None = None,
    sample_rows_per_file: int | None = None,
    total_sample_rows: int | None = None,
    weather_path: str | None = None,
    holiday_path: str | None = None,
    max_train_rows: int | None = None,
    fast_models: bool = False,
    selection_metric: str = "combined",
    train_rider_models: bool = True,
    train: bool = True,
    app_recent_hours: int | None = 168,
) -> dict[str, object]:
    ensure_dirs()
    log_step("Starting pipeline")
    holiday_path = holiday_path or str(save_nyc_holiday_table())
    paths = raw_paths or discover_csv_files()
    log_step(f"Discovered {len(paths)} raw CSV files")
    raw = load_trip_csvs(
        paths,
        sample_rows=sample_rows,
        sample_rows_per_file=sample_rows_per_file,
        total_sample_rows=total_sample_rows,
    )
    log_step("Cleaning trips")
    trips = clean_trips(raw)
    log_step("Building station metadata")
    station_meta = build_station_metadata(trips)
    log_step("Aggregating station-hour demand")
    station_hour = aggregate_station_hour(trips, station_meta)
    log_step("Adding holiday, lag, rolling, and weather features")
    station_hour = add_holiday_features(station_hour, holiday_path=holiday_path)
    station_hour = add_lag_features(station_hour)
    station_hour = merge_weather(station_hour, weather_path=weather_path)
    log_step("Aggregating area-hour demand")
    area_hour = aggregate_area_hour(station_hour)
    area_hour = add_holiday_features(area_hour, holiday_path=holiday_path)

    log_step("Saving processed artifacts")
    for frame in [trips, station_meta, station_hour, area_hour]:
        for col in ["ride_id", "start_station_id", "end_station_id", "start_station_name", "end_station_name", "source_file"]:
            if col in frame.columns:
                frame[col] = frame[col].astype("string")
    trips.to_parquet(PROCESSED_DIR / "clean_trips.parquet", index=False)
    station_meta.to_csv(PROCESSED_DIR / "station_metadata.csv", index=False)
    station_hour.to_parquet(PROCESSED_DIR / "station_hour_demand.parquet", index=False)
    area_hour.to_parquet(PROCESSED_DIR / "area_hour_demand.parquet", index=False)
    area_hour.to_parquet(PROCESSED_DIR / "app_area_hour.parquet", index=False)
    build_app_station_context(station_hour, recent_hours=app_recent_hours).to_parquet(PROCESSED_DIR / "app_station_context.parquet", index=False)
    for filename, frame in build_app_aggregates(station_hour, area_hour).items():
        frame.to_parquet(PROCESSED_DIR / filename, index=False)
    build_flow_summary(trips).to_parquet(PROCESSED_DIR / "app_flow_summary.parquet", index=False)
    trips.groupby("rider_type", as_index=False).agg(
        trips=("ride_id", "count"),
        avg_duration_min=("duration_min", "mean"),
        median_duration_min=("duration_min", "median"),
        avg_distance_miles=("trip_distance_miles", "mean"),
    ).to_csv(PROCESSED_DIR / "app_rider_summary.csv", index=False)
    station_hour.to_csv(PROCESSED_DIR / "station_hour_demand_sample.csv", index=False)
    area_hour.to_csv(PROCESSED_DIR / "area_hour_demand.csv", index=False)
    save_chart(hourly_pattern(station_hour), OUTPUTS_DIR / "figures" / "hourly_pattern.html")
    save_chart(weekday_weekend_pattern(station_hour), OUTPUTS_DIR / "figures" / "weekday_weekend_pattern.html")
    save_chart(top_stations(station_hour, n=25), OUTPUTS_DIR / "figures" / "top_stations.html")

    metrics = None
    if train:
        log_step("Training and evaluating models")
        metrics = train_models(
            station_hour,
            max_train_rows=max_train_rows,
            fast_models=fast_models,
            selection_metric=selection_metric,
            train_rider_models=train_rider_models,
        )

    summary = pd.DataFrame(
        [
            {
                "raw_files": len(paths),
                "clean_trips": len(trips),
                "stations": station_meta["start_station_id"].nunique(),
                "station_hour_rows": len(station_hour),
                "area_hour_rows": len(area_hour),
            }
        ]
    )
    summary.to_csv(OUTPUTS_DIR / "pipeline_summary.csv", index=False)
    log_step("Pipeline complete")
    return {"trips": trips, "station_hour": station_hour, "area_hour": area_hour, "metrics": metrics, "summary": summary}
