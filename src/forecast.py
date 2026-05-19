from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .feature_engineering import add_holiday_features, add_time_features, demand_tier, infer_area_label, season_from_month
from .modeling import NUMERIC_FEATURES, load_best_model
from .weather_features import add_weather_categories


@dataclass(frozen=True)
class Scenario:
    future_time: pd.Timestamp
    rider_type: str = "all"
    area_label: str | None = None
    station_id: str | None = None
    grid_cell: str | None = None
    region_cluster: int | None = None
    season: str | None = None
    is_holiday: int | None = None
    is_pre_holiday: int | None = None
    is_weekend: int | None = None
    temperature: float | None = None
    precipitation: float | None = None
    snow: float | None = None
    wind_speed: float | None = None


def build_scenario_frame(station_hour: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    if station_hour.empty:
        raise ValueError("No station-hour data is available for the selected scenario.")
    if "area_label" not in station_hour.columns:
        station_hour = station_hour.copy(deep=False)
        station_hour["area_label"] = infer_area_label(station_hour["station_lat"], station_hour["station_lng"])
    if scenario.station_id:
        station_hour = station_hour[station_hour["start_station_id"].astype(str) == str(scenario.station_id)]
    if scenario.grid_cell:
        station_hour = station_hour[station_hour["grid_cell"].astype(str) == str(scenario.grid_cell)]
    if scenario.area_label:
        station_hour = station_hour[station_hour["area_label"].astype(str) == str(scenario.area_label)]
    if scenario.region_cluster is not None:
        station_hour = station_hour[station_hour["region_cluster"].astype(int) == int(scenario.region_cluster)]
    if scenario.rider_type != "all":
        station_hour = station_hour[station_hour["rider_type"] == scenario.rider_type]
    if station_hour.empty:
        raise ValueError("No station or area matched the selected scenario.")
    meta_cols = [
        "start_station_id",
        "start_station_name",
        "station_lat",
        "station_lng",
        "grid_cell",
        "area_label",
        "region_cluster",
        "avg_duration_min",
        "avg_trip_distance_miles",
    ]
    meta = station_hour.sort_values("hour_start").groupby("start_station_id", as_index=False).tail(1)[meta_cols]
    meta = meta.drop_duplicates("start_station_id")
    if meta.empty:
        raise ValueError("No station or area matched the selected scenario.")

    out = meta.copy()
    out["hour_start"] = pd.Timestamp(scenario.future_time).floor("h")
    out["rider_type"] = scenario.rider_type
    out = add_time_features(out)
    out = add_holiday_features(out)
    out["season"] = scenario.season or out["month"].map(season_from_month)
    if scenario.is_weekend is not None:
        out["is_weekend"] = int(scenario.is_weekend)
    if scenario.is_holiday is not None:
        out["is_holiday"] = int(scenario.is_holiday)
    if scenario.is_pre_holiday is not None:
        out["is_pre_holiday"] = int(scenario.is_pre_holiday)

    hist = station_hour
    profile = (
        hist.groupby(["start_station_id", "rider_type", "day_of_week", "hour"])["demand"]
        .mean()
        .rename("historical_profile_demand")
        .reset_index()
    )
    out = out.merge(profile, on=["start_station_id", "rider_type", "day_of_week", "hour"], how="left")
    fallback_profile = (
        hist.groupby(["start_station_id", "day_of_week", "hour"])["demand"].mean().rename("fallback_profile").reset_index()
    )
    out = out.merge(fallback_profile, on=["start_station_id", "day_of_week", "hour"], how="left")
    out["historical_profile_demand"] = out["historical_profile_demand"].fillna(out["fallback_profile"]).fillna(hist["demand"].mean())
    for name, keys, agg in [
        ("station_hour_mean", ["start_station_id", "rider_type", "hour"], "mean"),
        ("station_hour_median", ["start_station_id", "rider_type", "hour"], "median"),
        ("area_hour_mean", ["grid_cell", "rider_type", "hour"], "mean"),
        ("station_avg_demand", ["start_station_id", "rider_type"], "mean"),
        ("station_total_demand", ["start_station_id", "rider_type"], "sum"),
        ("area_avg_demand", ["grid_cell", "rider_type"], "mean"),
        ("area_total_demand", ["grid_cell", "rider_type"], "sum"),
        ("cluster_hour_mean", ["region_cluster", "rider_type", "hour"], "mean"),
    ]:
        values = hist.groupby(keys)["demand"].agg(agg).rename(name).reset_index()
        out = out.merge(values, on=keys, how="left")
        out[name] = out[name].fillna(out["historical_profile_demand"])
    out["station_demand_tier"] = out["station_avg_demand"].map(demand_tier)
    out["area_demand_tier"] = out["area_avg_demand"].map(demand_tier)
    area_latest_source = (
        hist.groupby(["grid_cell", "rider_type", "hour_start"], as_index=False)["demand"]
        .sum()
        .sort_values("hour_start")
    )
    area_latest = (
        area_latest_source.groupby(["grid_cell", "rider_type"], as_index=False)
        .tail(1)[["grid_cell", "rider_type", "demand"]]
        .rename(columns={"demand": "area_lag_1h"})
    )
    area_profile = (
        hist.groupby(["grid_cell", "rider_type", "day_of_week", "hour"])["demand"]
        .sum()
        .rename("area_lag_24h")
        .reset_index()
    )
    area_roll = (
        hist.groupby(["grid_cell", "rider_type"])["demand"]
        .mean()
        .rename("area_rolling_mean_24h")
        .reset_index()
    )
    out = out.merge(area_latest, on=["grid_cell", "rider_type"], how="left")
    out = out.merge(area_profile, on=["grid_cell", "rider_type", "day_of_week", "hour"], how="left")
    out = out.merge(area_roll, on=["grid_cell", "rider_type"], how="left")
    for col in ["area_lag_1h", "area_lag_24h", "area_rolling_mean_24h"]:
        out[col] = out[col].fillna(out["area_hour_mean"]).fillna(out["historical_profile_demand"])

    latest_cols = [
        "start_station_id",
        "rider_type",
        "demand",
        "rolling_mean_24h",
        "rolling_mean_3h",
        "rolling_mean_6h",
        "rolling_mean_168h",
        "rolling_max_24h",
        "rolling_min_24h",
    ]
    latest_cols = [col for col in latest_cols if col in hist.columns]
    latest = (
        hist.sort_values("hour_start")
        .groupby(["start_station_id", "rider_type"], as_index=False)
        .tail(1)[latest_cols]
        .rename(columns={"demand": "lag_1h"})
    )
    out = out.merge(latest, on=["start_station_id", "rider_type"], how="left")
    out["lag_24h"] = out["historical_profile_demand"]
    out["lag_168h"] = out["historical_profile_demand"]
    out["rolling_mean_24h"] = out["rolling_mean_24h"].fillna(out["historical_profile_demand"])
    out["rolling_mean_3h"] = out.get("rolling_mean_3h", out["historical_profile_demand"]).fillna(out["historical_profile_demand"])
    out["rolling_mean_6h"] = out.get("rolling_mean_6h", out["historical_profile_demand"]).fillna(out["historical_profile_demand"])
    out["rolling_mean_168h"] = out.get("rolling_mean_168h", out["historical_profile_demand"]).fillna(out["historical_profile_demand"])
    out["rolling_max_24h"] = out["rolling_max_24h"].fillna(out["historical_profile_demand"])
    out["rolling_min_24h"] = out["rolling_min_24h"].fillna(out["historical_profile_demand"])
    out["lag_1h"] = out["lag_1h"].fillna(out["historical_profile_demand"])
    out["trend_24h"] = out["lag_1h"] - out["lag_24h"]

    for name, value in {
        "temperature": scenario.temperature,
        "precipitation": scenario.precipitation,
        "snow": scenario.snow,
        "wind_speed": scenario.wind_speed,
    }.items():
        out[name] = 0 if value is None else value
    out = add_weather_categories(out)

    for col in NUMERIC_FEATURES:
        if col not in out.columns:
            out[col] = 0
    out[NUMERIC_FEATURES] = out[NUMERIC_FEATURES].fillna(0)
    return out.drop(columns=["fallback_profile"], errors="ignore")


def predict_station_demand(station_hour: pd.DataFrame, scenario: Scenario, model=None) -> pd.DataFrame:
    model = model or load_best_model()
    frame = build_scenario_frame(station_hour, scenario)
    features = model.named_steps["preprocess"].feature_names_in_
    raw_pred = model.predict(frame[list(features)])
    if getattr(model, "target_transform_", None) == "log1p":
        raw_pred = np.expm1(raw_pred)
    blend_weight = getattr(model, "blend_weight_", 1.0)
    if "historical_profile_demand" in frame.columns and blend_weight < 1:
        raw_pred = blend_weight * raw_pred + (1 - blend_weight) * frame["historical_profile_demand"].to_numpy()
    frame["predicted_demand"] = np.clip(raw_pred, 0, getattr(model, "prediction_cap_", None))
    return frame.sort_values("predicted_demand", ascending=False).reset_index(drop=True)


def predict_area_hotspots(station_predictions: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["area_label"] if "area_label" in station_predictions.columns else ["grid_cell", "region_cluster"]
    return (
        station_predictions.groupby(group_cols, as_index=False)
        .agg(
            predicted_demand=("predicted_demand", "sum"),
            station_lat=("station_lat", "mean"),
            station_lng=("station_lng", "mean"),
            stations=("start_station_id", "nunique"),
        )
        .sort_values("predicted_demand", ascending=False)
        .reset_index(drop=True)
    )


def forecast_trend(station_hour: pd.DataFrame, scenario: Scenario, periods: int = 24, freq: str = "h", model=None) -> pd.DataFrame:
    rows = []
    for ts in pd.date_range(pd.Timestamp(scenario.future_time).floor("h"), periods=periods, freq=freq):
        s = Scenario(
            future_time=ts,
            rider_type=scenario.rider_type,
            area_label=scenario.area_label,
            station_id=scenario.station_id,
            grid_cell=scenario.grid_cell,
            region_cluster=scenario.region_cluster,
            season=scenario.season,
            is_holiday=scenario.is_holiday,
            is_pre_holiday=scenario.is_pre_holiday,
            is_weekend=scenario.is_weekend,
            temperature=scenario.temperature,
            precipitation=scenario.precipitation,
            snow=scenario.snow,
            wind_speed=scenario.wind_speed,
        )
        pred = predict_station_demand(station_hour, s, model=model)
        rows.append({"hour_start": ts, "predicted_demand": pred["predicted_demand"].sum()})
    return pd.DataFrame(rows)


def predict_flow_trends(trips: pd.DataFrame, scenario: Scenario, top_k: int = 30) -> pd.DataFrame:
    data = trips.copy()
    data["hour"] = data["started_at"].dt.hour
    data["day_of_week"] = data["started_at"].dt.dayofweek
    if scenario.area_label:
        if "area_label" not in data.columns:
            data["area_label"] = infer_area_label(data["start_lat"], data["start_lng"])
        data = data[data["area_label"] == scenario.area_label]
    if scenario.rider_type != "all":
        data = data[data["rider_type"] == scenario.rider_type]
    data = data[(data["hour"] == scenario.future_time.hour) & (data["day_of_week"] == scenario.future_time.dayofweek)]
    return (
        data.dropna(subset=["end_station_id"])
        .groupby(["start_station_name", "end_station_name"], as_index=False)
        .agg(
            flow_count=("ride_id", "count"),
            start_lat=("start_lat", "median"),
            start_lng=("start_lng", "median"),
            end_lat=("end_lat", "median"),
            end_lng=("end_lng", "median"),
        )
        .sort_values("flow_count", ascending=False)
        .head(top_k)
    )
