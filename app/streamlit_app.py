from __future__ import annotations

from pathlib import Path
import sys
import json

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import area_hotspots, station_hotspots
from src.config import OUTPUTS_DIR, PREDICTIONS_DIR, PROCESSED_DIR, ensure_dirs
from src.forecast import Scenario, forecast_trend, predict_area_hotspots, predict_station_demand
from src.feature_engineering import infer_area_label, season_from_month
from src.geocoding import get_cached_address, reverse_geocode_address
from src.map_utils import flow_map, station_prediction_map
from src.modeling import load_best_model_for_rider
from src.pipeline import run_pipeline
from src.visualization import demand_trend, holiday_pattern, hourly_pattern, top_stations, weather_binned_demand, weather_vs_demand, weekday_weekend_pattern
from src.weather_features import add_weather_categories, describe_weather_scenario


st.set_page_config(page_title="Citi Bike Demand Forecast", layout="wide")
ensure_dirs()


def render_chart(chart) -> None:
    if chart.__class__.__module__.startswith("altair"):
        st.altair_chart(chart, use_container_width=True)
    else:
        st.plotly_chart(chart, use_container_width=True)


def ensure_area_label(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "area_label" in df.columns:
        return df
    data = df.copy(deep=False)
    if "area_label" not in data.columns and {"station_lat", "station_lng"}.issubset(data.columns):
        data["area_label"] = infer_area_label(data["station_lat"], data["station_lng"])
    if "area_label" not in data.columns and {"start_lat", "start_lng"}.issubset(data.columns):
        data["area_label"] = infer_area_label(data["start_lat"], data["start_lng"])
    return data


def filter_area(df: pd.DataFrame, area_label: str) -> pd.DataFrame:
    if area_label == "All areas" or "area_label" not in df.columns:
        return df
    return df[df["area_label"] == area_label]


def limit_context_window(df: pd.DataFrame, recent_hours: int) -> pd.DataFrame:
    if df.empty or recent_hours <= 0 or "hour_start" not in df.columns:
        return df
    keys = [col for col in ["start_station_id", "rider_type"] if col in df.columns]
    if not keys:
        return df.sort_values("hour_start").tail(recent_hours)
    return df.sort_values("hour_start").groupby(keys, as_index=False).tail(recent_hours)


def station_display_options(station_data: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    meta = (
        station_data[["start_station_id", "start_station_name", "area_label", "station_lat", "station_lng"]]
        .drop_duplicates("start_station_id")
        .sort_values("start_station_name")
    )
    labels: list[str] = []
    mapping: dict[str, str] = {}
    for _, row in meta.iterrows():
        label = (
            f"{row['start_station_id']} | {row['start_station_name']} | "
            f"{row['area_label']} | {row['station_lat']:.5f}, {row['station_lng']:.5f}"
        )
        labels.append(label)
        mapping[label] = str(row["start_station_id"])
    return labels, mapping


def show_station_location(station_data: pd.DataFrame, station_id: str) -> None:
    meta = station_data[station_data["start_station_id"].astype(str) == str(station_id)]
    if meta.empty:
        return
    row = meta[["start_station_name", "area_label", "station_lat", "station_lng"]].drop_duplicates().iloc[0]
    st.caption(
        f"Station: {row['start_station_name']} | Area: {row['area_label']} | "
        f"Lat/Lng: {row['station_lat']:.6f}, {row['station_lng']:.6f}"
    )
    cached_address = get_cached_address(station_id)
    if cached_address:
        st.write(f"Street address: {cached_address}")
        return
    if st.button("Lookup street address", key=f"lookup_address_{station_id}"):
        with st.spinner("Looking up street address from coordinates..."):
            try:
                address = reverse_geocode_address(
                    station_id=str(station_id),
                    station_name=str(row["start_station_name"]),
                    lat=float(row["station_lat"]),
                    lng=float(row["station_lng"]),
                )
                st.write(f"Street address: {address}")
            except Exception as exc:
                st.warning(f"Could not look up a street address right now: {exc}")


@st.cache_data(show_spinner=False)
def load_holidays() -> pd.DataFrame:
    path = PROCESSED_DIR / "nyc_holidays.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date", "holiday_name"])
    holidays = pd.read_csv(path)
    holidays["date"] = pd.to_datetime(holidays["date"]).dt.date
    return holidays


def holiday_flag_for_date(date_value) -> tuple[int, str | None]:
    holidays = load_holidays()
    if holidays.empty:
        return 0, None
    target_date = pd.Timestamp(date_value).date()
    match = holidays[holidays["date"] == target_date]
    if match.empty:
        return 0, None
    return 1, str(match.iloc[0].get("holiday_name", "Holiday"))


def day_type_for_date(date_value) -> tuple[int, str]:
    is_weekend = int(pd.Timestamp(date_value).dayofweek in [5, 6])
    return is_weekend, "Weekend" if is_weekend else "Weekday"


def load_holiday_patterns(station_path: Path) -> pd.DataFrame:
    path = PROCESSED_DIR / "app_holiday_patterns.parquet"
    if path.exists():
        return pd.read_parquet(path)
    full_station_path = PROCESSED_DIR / "app_station_hour.parquet"
    source_path = full_station_path if full_station_path.exists() else station_path
    if not source_path.exists():
        return pd.DataFrame()
    try:
        data = pd.read_parquet(source_path, columns=["is_holiday", "rider_type", "area_label", "demand"])
    except Exception:
        return pd.DataFrame()
    if "area_label" not in data.columns:
        return pd.DataFrame()
    return data.groupby(["is_holiday", "rider_type", "area_label"], as_index=False)["demand"].mean()


@st.cache_resource(show_spinner=False)
def load_artifacts() -> dict[str, pd.DataFrame]:
    station_path = PROCESSED_DIR / "app_station_context.parquet"
    area_path = PROCESSED_DIR / "app_area_hour.parquet"
    flow_path = PROCESSED_DIR / "app_flow_summary.parquet"
    rider_summary_path = PROCESSED_DIR / "app_rider_summary.csv"
    if not station_path.exists():
        station_path = PROCESSED_DIR / "app_station_hour.parquet"
    if not area_path.exists():
        area_path = PROCESSED_DIR / "area_hour_demand.parquet"
    if not station_path.exists() or not area_path.exists():
        raise FileNotFoundError("Processed artifacts are missing.")
    return {
        "station_hour": pd.read_parquet(station_path),
        "area_hour": pd.read_parquet(area_path),
        "flow": pd.read_parquet(flow_path) if flow_path.exists() else pd.DataFrame(),
        "rider_summary": pd.read_csv(rider_summary_path) if rider_summary_path.exists() else pd.DataFrame(),
        "station_map_hour": pd.read_parquet(PROCESSED_DIR / "app_station_map_hour.parquet")
        if (PROCESSED_DIR / "app_station_map_hour.parquet").exists()
        else pd.DataFrame(),
        "hourly_patterns": pd.read_parquet(PROCESSED_DIR / "app_hourly_patterns.parquet")
        if (PROCESSED_DIR / "app_hourly_patterns.parquet").exists()
        else pd.DataFrame(),
        "weekday_patterns": pd.read_parquet(PROCESSED_DIR / "app_weekday_patterns.parquet")
        if (PROCESSED_DIR / "app_weekday_patterns.parquet").exists()
        else pd.DataFrame(),
        "holiday_patterns": load_holiday_patterns(station_path),
        "top_station_summary": pd.read_parquet(PROCESSED_DIR / "app_top_station_summary.parquet")
        if (PROCESSED_DIR / "app_top_station_summary.parquet").exists()
        else pd.DataFrame(),
    }


@st.cache_resource(show_spinner=False)
def cached_model(rider_type: str = "all"):
    return load_best_model_for_rider(rider_type)


st.title("NYC/Jersey City Citi Bike demand prediction")

try:
    artifacts = load_artifacts()
    station_hour = artifacts["station_hour"]
    area_hour = artifacts["area_hour"]
    flow_summary = artifacts["flow"]
    rider_summary = artifacts["rider_summary"]
    station_map_hour = artifacts["station_map_hour"]
    hourly_patterns = artifacts["hourly_patterns"]
    weekday_patterns = artifacts["weekday_patterns"]
    holiday_patterns = artifacts["holiday_patterns"]
    top_station_summary = artifacts["top_station_summary"]
    station_hour = ensure_area_label(station_hour)
    area_hour = ensure_area_label(area_hour)
except FileNotFoundError:
    st.warning("Processed data was not found. Run the full pipeline once before using the app.")
    if st.button("Run pipeline now"):
        with st.spinner("Processing local CSV files and training models..."):
            run_pipeline(train=True)
            st.cache_data.clear()
            st.cache_resource.clear()
        st.rerun()
    st.stop()

page = st.sidebar.radio(
    "Page",
    [
        "Overview",
        "Future Demand Prediction",
        "Weather Impact",
        "Station/Region Trend Viewer",
    ],
)

rider_options = ["all"] + sorted([x for x in station_hour["rider_type"].dropna().unique().tolist() if x != "all"])
area_options = ["All areas"] + sorted(station_hour["area_label"].dropna().unique().tolist())
selected_area = st.sidebar.selectbox("Area / borough", area_options)
context_options = {
    "Recent 168 hours": 168,
    "Recent 336 hours": 336,
    "Recent 720 hours": 720,
    "Recent 1440 hours": 1440,
    "All loaded context": 0,
}
context_label = st.sidebar.selectbox("Prediction context amount", list(context_options), index=0)
app_context_hours = context_options[context_label]
station_hour_context = limit_context_window(station_hour, app_context_hours)
st.sidebar.caption(f"Loaded context rows: {len(station_hour_context):,}")
station_hour_area = filter_area(station_hour_context, selected_area)
area_hour_area = filter_area(area_hour, selected_area)
station_map_hour_area = filter_area(ensure_area_label(station_map_hour), selected_area) if not station_map_hour.empty else pd.DataFrame()
hourly_patterns_area = filter_area(hourly_patterns, selected_area) if not hourly_patterns.empty else pd.DataFrame()
weekday_patterns_area = filter_area(weekday_patterns, selected_area) if not weekday_patterns.empty else pd.DataFrame()
holiday_patterns_area = filter_area(holiday_patterns, selected_area) if not holiday_patterns.empty else pd.DataFrame()
top_station_summary_area = filter_area(top_station_summary, selected_area) if not top_station_summary.empty else pd.DataFrame()

if page == "Overview":
    rider = st.selectbox("Rider type", rider_options)
    st.subheader("Rider behavior summary")
    if not rider_summary.empty:
        st.dataframe(rider_summary, use_container_width=True)
    else:
        st.dataframe(
            station_hour.groupby("rider_type", as_index=False).agg(
                trips=("demand", "sum"),
                avg_duration_min=("avg_duration_min", "mean"),
                avg_distance_miles=("avg_trip_distance_miles", "mean"),
            ),
            use_container_width=True,
        )
    chart_hourly = hourly_patterns_area if not hourly_patterns_area.empty else station_hour_area
    chart_weekday = weekday_patterns_area if not weekday_patterns_area.empty else station_hour_area
    chart_holiday = holiday_patterns_area if not holiday_patterns_area.empty else station_hour_area
    render_chart(hourly_pattern(chart_hourly if rider == "all" else chart_hourly[chart_hourly["rider_type"].isin([rider, "all"])]))
    render_chart(weekday_weekend_pattern(chart_weekday if rider == "all" else chart_weekday[chart_weekday["rider_type"].isin([rider, "all"])]))
    if "is_holiday" in chart_holiday.columns:
        render_chart(holiday_pattern(chart_holiday if rider == "all" else chart_holiday[chart_holiday["rider_type"].isin([rider, "all"])]))
    st.subheader("Hotspot tables")
    c1, c2 = st.columns(2)
    c1.dataframe(station_hotspots(station_hour_area, rider_type=rider), use_container_width=True)
    c2.dataframe(area_hotspots(area_hour_area, rider_type=rider), use_container_width=True)
    st.subheader("Member vs casual top stations")
    render_chart(top_stations(top_station_summary_area if not top_station_summary_area.empty else station_hour_area, rider_type="member", n=15))
    render_chart(top_stations(top_station_summary_area if not top_station_summary_area.empty else station_hour_area, rider_type="casual", n=15))
    metrics_path = OUTPUTS_DIR / "evaluation_metrics_top3.csv"
    if not metrics_path.exists():
        metrics_path = OUTPUTS_DIR / "evaluation_metrics.csv"
    if metrics_path.exists():
        st.subheader("Top 3 model comparison")
        st.dataframe(pd.read_csv(metrics_path, index_col=0).head(3), use_container_width=True)
        feature_config_path = PROJECT_ROOT / "models" / "feature_columns.json"
        if feature_config_path.exists():
            with feature_config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)
            st.success(f"Selected final model: {config.get('best_model')} ({config.get('selection_rule', 'best metric score')})")
        summary_path = OUTPUTS_DIR / "best_model_summary.csv"
        if summary_path.exists():
            st.caption("Best model summary")
            st.dataframe(pd.read_csv(summary_path), use_container_width=True)
        ranking_path = OUTPUTS_DIR / "model_selection_ranking.csv"
        if ranking_path.exists():
            st.caption("Top 3 combined model selection ranking")
            st.dataframe(pd.read_csv(ranking_path, index_col=0).head(3), use_container_width=True)
        explanation_path = OUTPUTS_DIR / "metric_explanations.csv"
        if explanation_path.exists():
            st.caption("Metric explanations")
            st.dataframe(pd.read_csv(explanation_path), use_container_width=True)
        rider_metrics_path = OUTPUTS_DIR / "rider_specific_model_metrics.csv"
        if rider_metrics_path.exists():
            st.caption("Rider-specific model metrics")
            st.dataframe(pd.read_csv(rider_metrics_path), use_container_width=True)
    hotspot_metrics_path = OUTPUTS_DIR / "hotspot_classification_metrics.csv"
    if hotspot_metrics_path.exists():
        st.subheader("Hotspot classification metrics")
        st.caption("Hotspot accuracy is computed by converting demand regression into a binary hotspot/not-hotspot task.")
        st.dataframe(pd.read_csv(hotspot_metrics_path), use_container_width=True)

elif page == "Future Demand Prediction":
    st.subheader("Custom future scenario")
    c1, c2, c3 = st.columns(3)
    future_date = c1.date_input("Future date", value=pd.Timestamp.now().date())
    future_hour = c2.slider("Future hour", 0, 23, 8)
    rider = c3.selectbox("Rider type", rider_options)
    model = cached_model(rider)
    c4, c5, c6, c10 = st.columns(4)
    auto_season = season_from_month(pd.Timestamp(future_date).month)
    c4.metric("Season", auto_season.title())
    holiday_auto, holiday_name = holiday_flag_for_date(future_date)
    c5.metric("Holiday", holiday_name if holiday_auto else "Non-holiday")
    weekend_auto, day_type_label = day_type_for_date(future_date)
    c6.metric("Day type", day_type_label)
    top_k = c10.slider("Top-k hotspots", 5, 100, 25)
    override_calendar = st.checkbox("Override season or holiday for scenario testing", value=False)
    season_value = auto_season
    holiday_value = holiday_auto
    weekend_value = weekend_auto
    if override_calendar:
        o1, o2, o3 = st.columns(3)
        season_value = o1.selectbox("Season override", ["winter", "spring", "summer", "fall"], index=["winter", "spring", "summer", "fall"].index(auto_season))
        holiday_choice = o2.selectbox("Holiday override", ["holiday", "non-holiday"], index=0 if holiday_auto else 1)
        holiday_value = int(holiday_choice == "holiday")
        day_type_choice = o3.selectbox("Day type override", ["weekday", "weekend"], index=1 if weekend_auto else 0)
        weekend_value = int(day_type_choice == "weekend")
    c7, c8, c9 = st.columns(3)
    station_labels, station_map = station_display_options(station_hour_area)
    stations = ["all"] + station_labels
    station_choice = c7.selectbox("Station", stations)
    station_id = "all" if station_choice == "all" else station_map[station_choice]
    weather_scenario = c8.selectbox("Weather scenario", ["Typical", "Clear", "Rain", "Snow", "Wind", "Temperature", "Custom"])
    weather_intensity = None
    if weather_scenario == "Rain":
        weather_intensity = c9.selectbox("Weather intensity", ["Light rain (0-2.5 mm)", "Moderate rain (2.5-10 mm)", "Heavy rain (>10 mm)"])
    elif weather_scenario == "Snow":
        weather_intensity = c9.selectbox("Weather intensity", ["Light snow (0-10 mm)", "Moderate snow (10-50 mm)", "Heavy snow (>50 mm)"])
    elif weather_scenario == "Wind":
        weather_intensity = c9.selectbox("Weather intensity", ["Light wind (<3 m/s)", "Moderate wind (3-8 m/s)", "Strong wind (>=8 m/s)"])
    elif weather_scenario == "Temperature":
        weather_intensity = c9.selectbox("Weather intensity", ["Freezing (<0 C)", "Cold (0-10 C)", "Mild (10-22 C)", "Hot (>=22 C)"])
    if station_id != "all":
        show_station_location(station_hour_area, station_id)
    weather_defaults = station_hour_area[["temperature", "precipitation", "snow", "wind_speed"]].median(numeric_only=True).fillna(0)
    weather_values = {
        "temperature": float(weather_defaults.get("temperature", 15.0)),
        "precipitation": float(weather_defaults.get("precipitation", 0.0)),
        "snow": float(weather_defaults.get("snow", 0.0)),
        "wind_speed": float(weather_defaults.get("wind_speed", 2.0)),
    }
    weather_profiles = {
        ("Clear", None): {"precipitation": 0.0, "snow": 0.0},
        ("Rain", "Light rain (0-2.5 mm)"): {"precipitation": 1.5, "snow": 0.0},
        ("Rain", "Moderate rain (2.5-10 mm)"): {"precipitation": 6.0, "snow": 0.0},
        ("Rain", "Heavy rain (>10 mm)"): {"precipitation": 15.0, "snow": 0.0},
        ("Snow", "Light snow (0-10 mm)"): {"precipitation": 0.0, "snow": 5.0, "temperature": -2.0},
        ("Snow", "Moderate snow (10-50 mm)"): {"precipitation": 0.0, "snow": 25.0, "temperature": -2.0},
        ("Snow", "Heavy snow (>50 mm)"): {"precipitation": 0.0, "snow": 75.0, "temperature": -5.0},
        ("Wind", "Light wind (<3 m/s)"): {"wind_speed": 2.0},
        ("Wind", "Moderate wind (3-8 m/s)"): {"wind_speed": 5.0},
        ("Wind", "Strong wind (>=8 m/s)"): {"wind_speed": 10.0},
        ("Temperature", "Freezing (<0 C)"): {"temperature": -5.0},
        ("Temperature", "Cold (0-10 C)"): {"temperature": 5.0},
        ("Temperature", "Mild (10-22 C)"): {"temperature": 15.0},
        ("Temperature", "Hot (>=22 C)"): {"temperature": 26.0},
    }
    weather_values.update(weather_profiles.get((weather_scenario, weather_intensity), {}))
    if weather_scenario == "Custom":
        w1, w2, w3, w4 = st.columns(4)
        weather_values["temperature"] = w1.number_input("Temperature C", value=weather_values["temperature"])
        weather_values["precipitation"] = w2.number_input("Rain mm", value=weather_values["precipitation"], min_value=0.0)
        weather_values["snow"] = w3.number_input("Snow mm", value=weather_values["snow"], min_value=0.0)
        weather_values["wind_speed"] = w4.number_input("Wind speed m/s", value=weather_values["wind_speed"], min_value=0.0)
    st.caption(
        "Weather inputs: "
        f"{weather_values['temperature']:.1f} C, "
        f"rain {weather_values['precipitation']:.1f} mm, "
        f"snow {weather_values['snow']:.1f} mm, "
        f"wind {weather_values['wind_speed']:.1f} m/s"
    )
    st.caption("Weather category: " + describe_weather_scenario(**weather_values))
    if "temperature" not in getattr(model.named_steps["preprocess"], "feature_names_in_", []):
        st.info("Weather scenario controls are visible, but the current saved model was trained before weather features were included. Rerun the pipeline with --weather to make weather affect predictions.")
    future_time = pd.Timestamp(future_date) + pd.Timedelta(hours=future_hour)
    scenario = Scenario(
        future_time=future_time,
        rider_type=rider,
        area_label=None if selected_area == "All areas" else selected_area,
        station_id=None if station_id == "all" else station_id,
        season=season_value,
        is_holiday=holiday_value,
        is_weekend=weekend_value,
        temperature=weather_values["temperature"],
        precipitation=weather_values["precipitation"],
        snow=weather_values["snow"],
        wind_speed=weather_values["wind_speed"],
    )
    prediction_source = station_hour_area
    if station_id != "all":
        prediction_source = prediction_source[prediction_source["start_station_id"].astype(str) == str(station_id)]
    elif selected_area == "All areas" and app_context_hours > 0 and len(prediction_source) > 1_000_000:
        st.info(f"All-area prediction is using the most recent {app_context_hours} station-hour context rows per station/rider to avoid loading the full history into memory. Select an area or station for a more detailed scenario.")
        prediction_source = prediction_source.sort_values("hour_start").groupby(["start_station_id", "rider_type"], as_index=False).tail(app_context_hours)
    predictions = predict_station_demand(prediction_source, scenario, model=model)
    predictions["is_weekend"] = weekend_value
    areas = predict_area_hotspots(predictions)
    predictions.to_csv(PREDICTIONS_DIR / "latest_station_predictions.csv", index=False)
    areas.to_csv(PREDICTIONS_DIR / "latest_area_predictions.csv", index=False)
    predicted_total = float(predictions["predicted_demand"].sum())
    st.metric("Predicted total demand", f"{predicted_total:.2f}")
    comparison_options = [
        (intensity, profile)
        for (scenario_name, intensity), profile in weather_profiles.items()
        if scenario_name == weather_scenario and intensity is not None
    ]
    if comparison_options:
        comparison_rows = []
        for intensity, profile in comparison_options:
            comparison_values = {
                "temperature": float(weather_defaults.get("temperature", 15.0)),
                "precipitation": float(weather_defaults.get("precipitation", 0.0)),
                "snow": float(weather_defaults.get("snow", 0.0)),
                "wind_speed": float(weather_defaults.get("wind_speed", 2.0)),
            }
            comparison_values.update(profile)
            comparison_scenario = Scenario(
                future_time=future_time,
                rider_type=rider,
                area_label=None if selected_area == "All areas" else selected_area,
                station_id=None if station_id == "all" else station_id,
                season=season_value,
                is_holiday=holiday_value,
                is_weekend=weekend_value,
                temperature=comparison_values["temperature"],
                precipitation=comparison_values["precipitation"],
                snow=comparison_values["snow"],
                wind_speed=comparison_values["wind_speed"],
            )
            comparison_pred = predict_station_demand(prediction_source, comparison_scenario, model=model)
            comparison_total = float(comparison_pred["predicted_demand"].sum())
            comparison_rows.append(
                {
                    "Weather intensity": intensity,
                    "Predicted demand": comparison_total,
                    "Difference from selected": comparison_total - predicted_total,
                }
            )
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df["Predicted demand"] = comparison_df["Predicted demand"].round(2)
        comparison_df["Difference from selected"] = comparison_df["Difference from selected"].round(2)
        st.subheader("Weather intensity comparison")
        st.dataframe(comparison_df, use_container_width=True)
    st_folium(
        station_prediction_map(predictions, top_k=top_k, selected_station_id=None if station_id == "all" else station_id),
        height=620,
        use_container_width=True,
    )
    c1, c2 = st.columns(2)
    c1.subheader("Future hotspot stations")
    c1.dataframe(predictions[["start_station_name", "start_station_id", "area_label", "predicted_demand"]].head(top_k), use_container_width=True)
    c2.subheader("Future hotspot areas")
    c2.dataframe(areas.head(top_k), use_container_width=True)
    st.subheader("Likely flow directions for matching hour and weekday")
    if flow_summary.empty:
        flows = pd.DataFrame()
    else:
        flow_rider = rider if rider in set(flow_summary["rider_type"].astype(str)) else "all"
        flows = flow_summary[
            (flow_summary["day_of_week"] == future_time.dayofweek)
            & (flow_summary["hour"] == future_time.hour)
            & (flow_summary["rider_type"] == flow_rider)
        ].sort_values("flow_count", ascending=False).head(30)
    st_folium(flow_map(flows), height=500, use_container_width=True)

elif page == "Weather Impact":
    st.subheader("Weather and demand")
    rider = st.selectbox("Rider type", rider_options)
    weather_cols = ["temperature", "precipitation", "snow", "wind_speed"]
    available = [col for col in weather_cols if col in station_hour.columns and station_hour[col].notna().any()]
    if not available:
        st.info("No real weather data is merged yet. Download NOAA data, then rerun the pipeline with --weather.")
        st.code(
            "python scripts/download_noaa_weather.py --start 2025-08-01 --end 2026-03-31 --token YOUR_NOAA_TOKEN\n"
            "python scripts/run_pipeline.py --weather outputs/external/noaa_nyc_daily_weather.csv",
            language="powershell",
        )
    else:
        categorized_weather = add_weather_categories(station_hour_area)
        col = st.selectbox("Weather variable", available, format_func=lambda x: "rain" if x == "precipitation" else x.replace("_", " "))
        render_chart(weather_vs_demand(categorized_weather, col, rider_type=rider))
        st.caption(
            "Hourly bar chart: x-axis is hour of day; y-axis is average hourly Citi Bike demand in rides; "
            "bar color is the selected weather range "
            "(temperature C, rain mm, snow mm, or wind speed m/s)."
        )
        render_chart(weather_binned_demand(categorized_weather, col, rider_type=rider))
        st.caption(
            "Binned chart: x-axis groups the weather values into ranges; "
            "y-axis is the average hourly demand in rides."
        )
        summary = (
            categorized_weather.groupby(pd.to_datetime(categorized_weather["hour_start"]).dt.date)
            .agg(demand=("demand", "sum"), temperature=("temperature", "mean"), precipitation=("precipitation", "mean"), snow=("snow", "mean"), wind_speed=("wind_speed", "mean"))
            .reset_index()
            .rename(columns={"hour_start": "date"})
        )
        st.dataframe(summary.rename(columns={"precipitation": "rain"}).tail(60), use_container_width=True)

elif page == "Station/Region Trend Viewer":
    mode = st.radio("Trend target", ["Station", "Area"], horizontal=True)
    rider = st.selectbox("Rider type", rider_options)
    model = cached_model(rider)
    future_date = st.date_input("Start date", value=pd.Timestamp.now().date())
    holiday_auto, holiday_name = holiday_flag_for_date(future_date)
    auto_season = season_from_month(pd.Timestamp(future_date).month)
    st.caption(f"Calendar: {auto_season.title()} | {holiday_name if holiday_auto else 'Non-holiday'}")
    future_hour = st.slider("Start hour", 0, 23, 8)
    horizon = st.selectbox("Forecast horizon", ["Next 24 hours", "Next 7 days"])
    periods = 24 if horizon == "Next 24 hours" else 24 * 7
    if mode == "Station":
        station_labels, station_map = station_display_options(station_hour_area)
        station_choice = st.selectbox("Station", station_labels)
        station_id = station_map[station_choice]
        show_station_location(station_hour_area, station_id)
        selected_station_map_data = (
            station_hour_area[station_hour_area["start_station_id"].astype(str) == str(station_id)]
            .drop_duplicates("start_station_id")
            .assign(predicted_demand=1.0)
        )
        st_folium(station_prediction_map(selected_station_map_data, top_k=1, heatmap=False, selected_station_id=station_id), height=360, use_container_width=True)
        scenario = Scenario(
            future_time=pd.Timestamp(future_date) + pd.Timedelta(hours=future_hour),
            rider_type=rider,
            area_label=None if selected_area == "All areas" else selected_area,
            season=auto_season,
            is_holiday=holiday_auto,
            station_id=station_id,
        )
    else:
        st.info("Area trend prediction may take more time because it forecasts all stations in the selected area.")
        available_areas = sorted(station_hour["area_label"].dropna().astype(str).unique().tolist())
        if selected_area != "All areas":
            available_areas = [selected_area]
        area_choice = st.selectbox("Area", available_areas)
        scenario = Scenario(
            future_time=pd.Timestamp(future_date) + pd.Timedelta(hours=future_hour),
            rider_type=rider,
            area_label=area_choice,
            season=auto_season,
            is_holiday=holiday_auto,
        )
    trend_source = station_hour_area if selected_area != "All areas" else station_hour
    if mode == "Station":
        trend_source = trend_source[trend_source["start_station_id"].astype(str) == str(station_id)]
    else:
        trend_source = trend_source[trend_source["area_label"].astype(str) == str(area_choice)]
    trend = forecast_trend(trend_source, scenario, periods=periods, model=model)
    render_chart(demand_trend(trend))
    st.dataframe(trend, use_container_width=True)
