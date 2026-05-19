from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd

try:
    import plotly.express as px
except ModuleNotFoundError:
    px = None


def hourly_pattern(station_hour: pd.DataFrame):
    data = station_hour.groupby(["hour", "rider_type"], as_index=False)["demand"].mean()
    if px:
        return px.line(data, x="hour", y="demand", color="rider_type", markers=True, title="Average hourly demand")
    return alt.Chart(data, title="Average hourly demand").mark_line(point=True).encode(x="hour:Q", y="demand:Q", color="rider_type:N")


def weekday_weekend_pattern(station_hour: pd.DataFrame):
    data = station_hour.groupby(["is_weekend", "rider_type"], as_index=False)["demand"].mean()
    data["day_type"] = data["is_weekend"].map({0: "Weekday", 1: "Weekend"})
    if px:
        return px.bar(data, x="day_type", y="demand", color="rider_type", barmode="group", title="Weekday vs weekend demand")
    return alt.Chart(data, title="Weekday vs weekend demand").mark_bar().encode(x="day_type:N", y="demand:Q", color="rider_type:N", xOffset="rider_type:N")


def holiday_pattern(station_hour: pd.DataFrame):
    data = station_hour.dropna(subset=["is_holiday"]).copy()
    data["is_holiday"] = data["is_holiday"].astype(int)
    data = data.groupby(["is_holiday", "rider_type"], as_index=False)["demand"].mean()
    data["calendar_type"] = data["is_holiday"].map({0: "Non-holiday", 1: "Holiday"})
    category_order = ["Non-holiday", "Holiday"]
    if px:
        return px.bar(
            data,
            x="calendar_type",
            y="demand",
            color="rider_type",
            barmode="group",
            title="Holiday vs non-holiday demand",
            labels={"calendar_type": "calendar type", "demand": "average hourly demand"},
            category_orders={"calendar_type": category_order},
        )
    return alt.Chart(data, title="Holiday vs non-holiday demand").mark_bar().encode(
        x=alt.X("calendar_type:N", sort=category_order),
        y=alt.Y("demand:Q", title="average hourly demand"),
        color="rider_type:N",
        xOffset="rider_type:N",
    )


def top_stations(station_hour: pd.DataFrame, rider_type: str = "all", n: int = 20):
    data = station_hour if rider_type == "all" else station_hour[station_hour["rider_type"] == rider_type]
    if {"start_station_name", "demand"}.issubset(data.columns):
        top = data.groupby("start_station_name", as_index=False)["demand"].sum().sort_values("demand", ascending=True).tail(n)
    else:
        top = data
    if px:
        return px.bar(top, x="demand", y="start_station_name", orientation="h", title="Top stations")
    return alt.Chart(top, title="Top stations").mark_bar().encode(x="demand:Q", y=alt.Y("start_station_name:N", sort="-x"))


def demand_trend(df: pd.DataFrame, y: str = "predicted_demand"):
    if px:
        return px.line(df, x="hour_start", y=y, markers=True, title="Predicted demand trend")
    return alt.Chart(df, title="Predicted demand trend").mark_line(point=True).encode(x="hour_start:T", y=f"{y}:Q")


def _weather_bins(weather_col: str):
    return {
        "temperature": [-30, -10, -5, 0, 5, 10, 15, 20, 25, 35],
        "precipitation": [-0.001, 0, 2.5, 10, 25, 100],
        "snow": [-0.001, 0, 10, 50, 100, 500],
        "wind_speed": [0, 3, 8, 12, 20, 50],
    }.get(weather_col)


def _weather_label(weather_col: str) -> str:
    return {"precipitation": "rain"}.get(weather_col, weather_col.replace("_", " "))


def _weather_bin_labels(weather_col: str) -> list[str] | None:
    return {
        "temperature": ["<-10 C", "-10--5 C", "-5-0 C", "0-5 C", "5-10 C", "10-15 C", "15-20 C", "20-25 C", "25-35 C"],
        "precipitation": ["0 mm", "0-2.5 mm", "2.5-10 mm", "10-25 mm", ">25 mm"],
        "snow": ["0 mm", "0-10 mm", "10-50 mm", "50-100 mm", ">100 mm"],
        "wind_speed": ["<3 m/s", "3-8 m/s", "8-12 m/s", "12-20 m/s", ">20 m/s"],
    }.get(weather_col)


def _assign_weather_bins(values: pd.Series, weather_col: str) -> pd.Series:
    bins = _weather_bins(weather_col)
    labels = _weather_bin_labels(weather_col)
    if bins is None:
        dynamic_bins = min(10, max(3, values.nunique()))
        return pd.cut(values, bins=dynamic_bins, include_lowest=True, duplicates="drop").astype(str)
    return pd.cut(values, bins=bins, labels=labels, include_lowest=True, duplicates="drop")


def weather_vs_demand(station_hour: pd.DataFrame, weather_col: str, rider_type: str = "all"):
    data = station_hour if rider_type == "all" else station_hour[station_hour["rider_type"] == rider_type]
    data = data.dropna(subset=[weather_col])
    data = data.groupby(["hour_start", "hour", weather_col], as_index=False)["demand"].sum()
    data["weather_bin"] = _assign_weather_bins(data[weather_col], weather_col)
    summary = (
        data.groupby(["hour", "weather_bin"], observed=True, as_index=False)
        .agg(avg_demand=("demand", "mean"), observations=("demand", "size"))
    )
    weather_label = _weather_label(weather_col)
    category_order = _weather_bin_labels(weather_col)
    if category_order:
        summary["weather_bin"] = pd.Categorical(summary["weather_bin"], categories=category_order, ordered=True)
    summary = summary.sort_values(["hour", "weather_bin"])
    summary["weather_bin"] = summary["weather_bin"].astype(str)
    title = f"Hourly demand by {weather_label} range"
    if px:
        return px.bar(
            summary,
            x="hour",
            y="avg_demand",
            color="weather_bin",
            barmode="group",
            title=title,
            labels={
                "hour": "hour of day",
                "avg_demand": "average hourly demand",
                "weather_bin": f"{weather_label} range",
            },
            category_orders={"weather_bin": category_order} if category_order else None,
        )
    return alt.Chart(summary, title=title).mark_bar().encode(
        x=alt.X("hour:O", title="hour of day"),
        y=alt.Y("avg_demand:Q", title="average hourly demand"),
        color=alt.Color("weather_bin:N", sort=category_order),
        tooltip=["hour:O", "weather_bin:N", "avg_demand:Q"],
    )


def weather_binned_demand(station_hour: pd.DataFrame, weather_col: str, rider_type: str = "all"):
    data = station_hour if rider_type == "all" else station_hour[station_hour["rider_type"] == rider_type]
    data = data.dropna(subset=[weather_col])
    if data.empty:
        return weather_vs_demand(station_hour, weather_col, rider_type=rider_type)
    hourly = data.groupby(["hour_start", weather_col], as_index=False)["demand"].sum()
    hourly["weather_bin"] = _assign_weather_bins(hourly[weather_col], weather_col)
    summary = (
        hourly.dropna(subset=["weather_bin"])
        .groupby("weather_bin", observed=True, as_index=False)
        .agg(avg_demand=("demand", "mean"), observations=("demand", "size"))
    )
    weather_label = _weather_label(weather_col)
    category_order = _weather_bin_labels(weather_col)
    if category_order:
        summary["weather_bin"] = pd.Categorical(summary["weather_bin"], categories=category_order, ordered=True)
        summary = summary.sort_values("weather_bin")
    summary["weather_bin"] = summary["weather_bin"].astype(str)
    title = f"Average demand by {weather_label} range"
    if px:
        fig = px.bar(
            summary,
            x="weather_bin",
            y="avg_demand",
            title=title,
            labels={"weather_bin": f"{weather_label} range", "avg_demand": "average hourly demand"},
            category_orders={"weather_bin": category_order} if category_order else None,
        )
        return fig
    return alt.Chart(summary, title=title).mark_bar().encode(
        x=alt.X("weather_bin:N", sort=category_order),
        y="avg_demand:Q",
        tooltip=["weather_bin:N", "avg_demand:Q"],
    )


def weather_category_demand(station_hour: pd.DataFrame, category_col: str, rider_type: str = "all"):
    data = station_hour if rider_type == "all" else station_hour[station_hour["rider_type"] == rider_type]
    data = data.dropna(subset=[category_col])
    summary = data.groupby([category_col, "rider_type"], as_index=False)["demand"].mean()
    title = f"Average demand by {category_col.replace('_', ' ')}"
    if px:
        return px.bar(summary, x=category_col, y="demand", color="rider_type", barmode="group", title=title)
    return alt.Chart(summary, title=title).mark_bar().encode(
        x=alt.X(f"{category_col}:N", sort=None),
        y="demand:Q",
        color="rider_type:N",
        xOffset="rider_type:N",
    )


def save_chart(chart, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(chart, "write_html"):
        chart.write_html(path)
    else:
        chart.save(str(path))
