from __future__ import annotations

import pandas as pd


def rain_intensity(value: float | int | None) -> str:
    if pd.isna(value) or value <= 0:
        return "No rain"
    if value <= 2.5:
        return "Light rain"
    if value <= 10:
        return "Moderate rain"
    return "Heavy rain"


def rain_level(value: float | int | None) -> int:
    if pd.isna(value) or value <= 0:
        return 0
    if value <= 2.5:
        return 1
    if value <= 10:
        return 2
    return 3


def snow_intensity(value: float | int | None) -> str:
    if pd.isna(value) or value <= 0:
        return "No snow"
    if value <= 10:
        return "Light snow"
    if value <= 50:
        return "Moderate snow"
    return "Heavy snow"


def snow_level(value: float | int | None) -> int:
    if pd.isna(value) or value <= 0:
        return 0
    if value <= 10:
        return 1
    if value <= 50:
        return 2
    return 3


def wind_intensity(value: float | int | None) -> str:
    if pd.isna(value):
        return "Unknown wind"
    if value < 3:
        return "Light wind"
    if value < 8:
        return "Moderate wind"
    return "Strong wind"


def wind_level(value: float | int | None) -> int:
    if pd.isna(value):
        return 0
    if value < 3:
        return 1
    if value < 8:
        return 2
    return 3


def temperature_band(value: float | int | None) -> str:
    if pd.isna(value):
        return "Unknown temperature"
    if value < 0:
        return "Freezing"
    if value < 10:
        return "Cold"
    if value < 22:
        return "Mild"
    return "Hot"


def temperature_level(value: float | int | None) -> int:
    if pd.isna(value):
        return 0
    if value < 0:
        return 1
    if value < 10:
        return 2
    if value < 22:
        return 3
    return 4


def add_weather_categories(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "precipitation" in data.columns:
        data["rain_intensity"] = data["precipitation"].map(rain_intensity)
        data["rain_level"] = data["precipitation"].map(rain_level)
    if "snow" in data.columns:
        data["snow_intensity"] = data["snow"].map(snow_intensity)
        data["snow_level"] = data["snow"].map(snow_level)
    if "wind_speed" in data.columns:
        data["wind_intensity"] = data["wind_speed"].map(wind_intensity)
        data["wind_level"] = data["wind_speed"].map(wind_level)
    if "temperature" in data.columns:
        data["temperature_band"] = data["temperature"].map(temperature_band)
        data["temperature_level"] = data["temperature"].map(temperature_level)
    return data


def describe_weather_scenario(temperature: float, precipitation: float, snow: float, wind_speed: float) -> str:
    return (
        f"{temperature_band(temperature)}; "
        f"{rain_intensity(precipitation)}; "
        f"{snow_intensity(snow)}; "
        f"{wind_intensity(wind_speed)}"
    )
