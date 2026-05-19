from __future__ import annotations

import folium
import pandas as pd
from folium.plugins import HeatMap, MarkerCluster


def _center(df: pd.DataFrame) -> tuple[float, float]:
    if df.empty:
        return 40.73, -74.03
    return float(df["station_lat"].mean()), float(df["station_lng"].mean())


def station_prediction_map(
    predictions: pd.DataFrame,
    top_k: int = 50,
    heatmap: bool = True,
    selected_station_id: str | None = None,
) -> folium.Map:
    selected = pd.DataFrame()
    if selected_station_id is not None and "start_station_id" in predictions.columns:
        selected = predictions[predictions["start_station_id"].astype(str) == str(selected_station_id)]
    center = _center(selected if not selected.empty else predictions)
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB positron")
    if heatmap and not predictions.empty:
        heat_data = predictions[["station_lat", "station_lng", "predicted_demand"]].dropna().values.tolist()
        HeatMap(heat_data, radius=18, blur=24, min_opacity=0.25).add_to(m)
    cluster = MarkerCluster(name="Predicted station demand").add_to(m)
    top = predictions.sort_values("predicted_demand", ascending=False).head(top_k)
    max_demand = max(float(top["predicted_demand"].max()), 1.0) if not top.empty else 1.0
    for _, row in top.iterrows():
        demand = float(row["predicted_demand"])
        area = row["grid_cell"] if "grid_cell" in row.index else "unknown"
        folium.CircleMarker(
            location=[row["station_lat"], row["station_lng"]],
            radius=4 + 12 * demand / max_demand,
            color="#c62828",
            fill=True,
            fill_color="#ef5350",
            fill_opacity=0.75,
            popup=f"{row['start_station_name']}<br>Demand: {demand:.1f}<br>Area: {area}",
        ).add_to(cluster)
    if not selected.empty:
        row = selected.iloc[0]
        demand = float(row["predicted_demand"])
        folium.Marker(
            location=[row["station_lat"], row["station_lng"]],
            tooltip=f"Selected station: {row['start_station_name']}",
            popup=(
                f"<b>Selected station</b><br>{row['start_station_name']}<br>"
                f"Station ID: {row['start_station_id']}<br>"
                f"Predicted demand: {demand:.1f}<br>"
                f"Lat/Lng: {row['station_lat']:.6f}, {row['station_lng']:.6f}"
            ),
            icon=folium.Icon(color="red", icon="star"),
        ).add_to(m)
        folium.Circle(
            location=[row["station_lat"], row["station_lng"]],
            radius=180,
            color="#d62728",
            weight=3,
            fill=True,
            fill_color="#d62728",
            fill_opacity=0.12,
        ).add_to(m)
    folium.LayerControl().add_to(m)
    return m


def historical_heatmap(station_hour: pd.DataFrame, hour: int | None = None, rider_type: str = "all") -> folium.Map:
    data = station_hour.copy()
    if hour is not None:
        data = data[data["hour"] == hour]
    if rider_type != "all":
        data = data[data["rider_type"] == rider_type]
    grouped = (
        data.groupby(["start_station_id", "start_station_name", "station_lat", "station_lng", "grid_cell"], as_index=False)
        .agg(predicted_demand=("demand", "mean"))
    )
    return station_prediction_map(grouped, top_k=100, heatmap=True)


def flow_map(flows: pd.DataFrame) -> folium.Map:
    if flows.empty:
        return folium.Map(location=[40.73, -74.03], zoom_start=13, tiles="CartoDB positron")
    m = folium.Map(location=[float(flows["start_lat"].mean()), float(flows["start_lng"].mean())], zoom_start=13, tiles="CartoDB positron")
    max_flow = max(float(flows["flow_count"].max()), 1.0)
    for _, row in flows.iterrows():
        weight = 1 + 6 * float(row["flow_count"]) / max_flow
        folium.PolyLine(
            locations=[[row["start_lat"], row["start_lng"]], [row["end_lat"], row["end_lng"]]],
            color="#1565c0",
            weight=weight,
            opacity=0.65,
            tooltip=f"{row['start_station_name']} -> {row['end_station_name']}: {row['flow_count']}",
        ).add_to(m)
    return m
