from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .config import GRID_SIZE_DEGREES
from .weather_features import add_weather_categories


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8, 9):
        return "summer"
    return "fall"


def demand_tier(value: float | int | None) -> str:
    if pd.isna(value):
        return "unknown"
    if value < 1:
        return "low"
    if value < 3:
        return "medium"
    if value < 8:
        return "high"
    return "very_high"


def add_time_features(df: pd.DataFrame, time_col: str = "hour_start") -> pd.DataFrame:
    data = df.copy()
    t = pd.to_datetime(data[time_col])
    data["year"] = t.dt.year
    data["month"] = t.dt.month
    data["day"] = t.dt.day
    data["hour"] = t.dt.hour
    data["minute"] = t.dt.minute
    data["day_of_week"] = t.dt.dayofweek
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)
    data["week_of_year"] = t.dt.isocalendar().week.astype(int)
    data["season"] = data["month"].map(season_from_month)
    data["peak_hour"] = data["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    data["morning_peak"] = data["hour"].isin([7, 8, 9]).astype(int)
    data["evening_peak"] = data["hour"].isin([16, 17, 18, 19]).astype(int)
    data["weekend_peak"] = ((data["is_weekend"] == 1) & data["hour"].between(11, 17)).astype(int)
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
    data["daypart"] = pd.cut(
        data["hour"],
        bins=[-1, 5, 10, 15, 20, 23],
        labels=["overnight", "morning", "midday", "evening", "night"],
    ).astype(str)
    return data


def add_holiday_features(df: pd.DataFrame, time_col: str = "hour_start", holiday_path: str | None = None) -> pd.DataFrame:
    data = df.copy()
    dates = pd.to_datetime(data[time_col]).dt.date
    if "is_holiday" not in data.columns:
        data["is_holiday"] = 0
    if "is_pre_holiday" not in data.columns:
        data["is_pre_holiday"] = 0
    if holiday_path:
        try:
            holidays = pd.read_csv(holiday_path)
            holidays["date"] = pd.to_datetime(holidays["date"]).dt.date
            holiday_dates = set(holidays["date"])
            pre_dates = {d - pd.Timedelta(days=1) for d in holiday_dates}
            data["is_holiday"] = dates.isin(holiday_dates).astype(int)
            data["is_pre_holiday"] = dates.isin(pre_dates).astype(int)
        except Exception:
            pass
    return data


def assign_grid(lat: pd.Series, lng: pd.Series, grid_size: float = GRID_SIZE_DEGREES) -> pd.Series:
    lat_bin = np.floor(lat / grid_size).astype(int)
    lng_bin = np.floor(lng / grid_size).astype(int)
    return lat_bin.astype(str) + "_" + lng_bin.astype(str)


def infer_area_label(lat: pd.Series, lng: pd.Series) -> pd.Series:
    """Approximate borough/area labels from coordinates when official labels are absent."""
    area = pd.Series("Other", index=lat.index)
    area[(lat.between(40.68, 40.89)) & (lng.between(-74.03, -73.91))] = "Manhattan"
    area[(lat.between(40.56, 40.74)) & (lng.between(-74.05, -73.83))] = "Brooklyn"
    area[(lat.between(40.53, 40.82)) & (lng.between(-73.96, -73.70))] = "Queens"
    area[(lat.between(40.78, 40.92)) & (lng.between(-73.94, -73.76))] = "Bronx"
    area[(lat.between(40.68, 40.77)) & (lng.between(-74.10, -74.02))] = "Jersey City / Hoboken"
    return area


def build_station_metadata(trips: pd.DataFrame, n_clusters: int = 12) -> pd.DataFrame:
    station = (
        trips.groupby(["start_station_id", "start_station_name"], as_index=False)
        .agg(start_lat=("start_lat", "median"), start_lng=("start_lng", "median"), trips=("ride_id", "count"))
        .rename(columns={"start_lat": "station_lat", "start_lng": "station_lng"})
    )
    station["grid_cell"] = assign_grid(station["station_lat"], station["station_lng"])
    station["area_label"] = infer_area_label(station["station_lat"], station["station_lng"])
    cluster_count = min(n_clusters, len(station))
    if cluster_count >= 2:
        km = KMeans(n_clusters=cluster_count, random_state=42, n_init="auto")
        station["region_cluster"] = km.fit_predict(station[["station_lat", "station_lng"]])
    else:
        station["region_cluster"] = 0
    return station


def aggregate_station_hour(trips: pd.DataFrame, station_meta: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        trips.groupby(["hour_start", "start_station_id", "rider_type"], as_index=False)
        .agg(
            demand=("ride_id", "count"),
            avg_duration_min=("duration_min", "mean"),
            avg_trip_distance_miles=("trip_distance_miles", "mean"),
        )
    )
    all_riders = (
        trips.groupby(["hour_start", "start_station_id"], as_index=False)
        .agg(
            demand=("ride_id", "count"),
            avg_duration_min=("duration_min", "mean"),
            avg_trip_distance_miles=("trip_distance_miles", "mean"),
        )
    )
    all_riders["rider_type"] = "all"
    out = pd.concat([grouped, all_riders], ignore_index=True)
    out = out.merge(station_meta, on="start_station_id", how="left")
    out = add_time_features(out)
    return out


def add_lag_features(demand: pd.DataFrame) -> pd.DataFrame:
    data = demand.sort_values(["start_station_id", "rider_type", "hour_start"]).copy()
    keys = ["start_station_id", "rider_type"]
    grp = data.groupby(keys)["demand"]
    data["lag_1h"] = grp.shift(1)
    data["lag_24h"] = grp.shift(24)
    data["lag_168h"] = grp.shift(168)
    data["rolling_mean_24h"] = grp.transform(lambda s: s.shift(1).rolling(24, min_periods=1).mean())
    data["rolling_mean_3h"] = grp.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    data["rolling_mean_6h"] = grp.transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
    data["rolling_mean_168h"] = grp.transform(lambda s: s.shift(1).rolling(168, min_periods=1).mean())
    data["rolling_max_24h"] = grp.transform(lambda s: s.shift(1).rolling(24, min_periods=1).max())
    data["rolling_min_24h"] = grp.transform(lambda s: s.shift(1).rolling(24, min_periods=1).min())
    data["trend_24h"] = data["lag_1h"] - data["lag_24h"]
    profile = data.groupby(["start_station_id", "rider_type", "day_of_week", "hour"])["demand"].transform("mean")
    data["historical_profile_demand"] = profile
    data["station_hour_mean"] = data.groupby(["start_station_id", "rider_type", "hour"])["demand"].transform("mean")
    data["station_hour_median"] = data.groupby(["start_station_id", "rider_type", "hour"])["demand"].transform("median")
    data["area_hour_mean"] = data.groupby(["grid_cell", "rider_type", "hour"])["demand"].transform("mean")
    data["station_avg_demand"] = data.groupby(["start_station_id", "rider_type"])["demand"].transform("mean")
    data["station_total_demand"] = data.groupby(["start_station_id", "rider_type"])["demand"].transform("sum")
    data["area_avg_demand"] = data.groupby(["grid_cell", "rider_type"])["demand"].transform("mean")
    data["area_total_demand"] = data.groupby(["grid_cell", "rider_type"])["demand"].transform("sum")
    data["cluster_hour_mean"] = data.groupby(["region_cluster", "rider_type", "hour"])["demand"].transform("mean")
    data["station_demand_tier"] = data["station_avg_demand"].map(demand_tier)
    data["area_demand_tier"] = data["area_avg_demand"].map(demand_tier)
    area_hourly = (
        data.groupby(["grid_cell", "rider_type", "hour_start"], as_index=False)["demand"]
        .sum()
        .sort_values(["grid_cell", "rider_type", "hour_start"])
    )
    area_grp = area_hourly.groupby(["grid_cell", "rider_type"])["demand"]
    area_hourly["area_lag_1h"] = area_grp.shift(1)
    area_hourly["area_lag_24h"] = area_grp.shift(24)
    area_hourly["area_rolling_mean_24h"] = area_grp.transform(lambda s: s.shift(1).rolling(24, min_periods=1).mean())
    data = data.merge(
        area_hourly[["grid_cell", "rider_type", "hour_start", "area_lag_1h", "area_lag_24h", "area_rolling_mean_24h"]],
        on=["grid_cell", "rider_type", "hour_start"],
        how="left",
    )
    lag_cols = [
        "lag_1h",
        "lag_24h",
        "lag_168h",
        "rolling_mean_24h",
        "rolling_mean_3h",
        "rolling_mean_6h",
        "rolling_mean_168h",
        "rolling_max_24h",
        "rolling_min_24h",
        "trend_24h",
        "historical_profile_demand",
        "station_hour_mean",
        "station_hour_median",
        "area_hour_mean",
        "station_avg_demand",
        "station_total_demand",
        "area_avg_demand",
        "area_total_demand",
        "cluster_hour_mean",
        "area_lag_1h",
        "area_lag_24h",
        "area_rolling_mean_24h",
    ]
    data[lag_cols] = data[lag_cols].fillna(data["historical_profile_demand"]).fillna(data["demand"].median())
    return data


def aggregate_area_hour(station_hour: pd.DataFrame) -> pd.DataFrame:
    area = (
        station_hour.groupby(["hour_start", "grid_cell", "region_cluster", "area_label", "rider_type"], as_index=False)
        .agg(
            demand=("demand", "sum"),
            station_lat=("station_lat", "mean"),
            station_lng=("station_lng", "mean"),
            stations=("start_station_id", "nunique"),
        )
    )
    area = add_time_features(area)
    return area


def merge_weather(demand: pd.DataFrame, weather_path: str | None = None) -> pd.DataFrame:
    data = demand.copy()
    for col in ["temperature", "precipitation", "snow", "wind_speed"]:
        data[col] = np.nan
    if not weather_path:
        return data
    try:
        weather = pd.read_csv(weather_path)
        if "timestamp" in weather.columns:
            weather["hour_start"] = pd.to_datetime(weather["timestamp"], errors="coerce").dt.floor("h")
            keep = ["hour_start", "temperature", "precipitation", "snow", "wind_speed"]
            merged = data.drop(columns=keep[1:], errors="ignore").merge(weather[keep], on="hour_start", how="left")
            return add_weather_categories(merged)
        if "date" in weather.columns:
            weather["weather_date"] = pd.to_datetime(weather["date"], errors="coerce").dt.date
            data["weather_date"] = pd.to_datetime(data["hour_start"], errors="coerce").dt.date
            keep = ["weather_date", "temperature", "precipitation", "snow", "wind_speed"]
            merged = data.drop(columns=keep[1:], errors="ignore").merge(weather[keep], on="weather_date", how="left")
            return add_weather_categories(merged.drop(columns=["weather_date"], errors="ignore"))
    except Exception:
        return data
    return add_weather_categories(data)
