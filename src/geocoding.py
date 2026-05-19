from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from .config import EXTERNAL_DIR, ensure_dirs


ADDRESS_CACHE_PATH = EXTERNAL_DIR / "station_addresses.csv"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"


def load_address_cache(path: Path = ADDRESS_CACHE_PATH) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype={"station_id": str})
    return pd.DataFrame(columns=["station_id", "station_name", "lat", "lng", "display_address"])


def get_cached_address(station_id: str, path: Path = ADDRESS_CACHE_PATH) -> str | None:
    cache = load_address_cache(path)
    match = cache[cache["station_id"].astype(str) == str(station_id)]
    if match.empty:
        return None
    address = match.iloc[-1]["display_address"]
    return None if pd.isna(address) else str(address)


def cache_address(
    station_id: str,
    station_name: str,
    lat: float,
    lng: float,
    display_address: str,
    path: Path = ADDRESS_CACHE_PATH,
) -> None:
    ensure_dirs()
    cache = load_address_cache(path)
    cache = cache[cache["station_id"].astype(str) != str(station_id)]
    new_row = pd.DataFrame(
        [
            {
                "station_id": str(station_id),
                "station_name": station_name,
                "lat": lat,
                "lng": lng,
                "display_address": display_address,
            }
        ]
    )
    pd.concat([cache, new_row], ignore_index=True).to_csv(path, index=False)


def reverse_geocode_address(station_id: str, station_name: str, lat: float, lng: float) -> str:
    cached = get_cached_address(station_id)
    if cached:
        return cached

    response = requests.get(
        NOMINATIM_REVERSE_URL,
        params={"format": "jsonv2", "lat": lat, "lon": lng, "zoom": 18, "addressdetails": 1},
        headers={"User-Agent": "citibike-demand-forecast-local-app/1.0"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    address = payload.get("display_name") or station_name
    cache_address(station_id, station_name, lat, lng, address)
    return address

