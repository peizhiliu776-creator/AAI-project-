from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.weather import DEFAULT_STATION_ID, save_noaa_daily_weather


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NOAA CDO daily weather for model features.")
    parser.add_argument("--start", required=True, help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end", required=True, help="End date, YYYY-MM-DD.")
    parser.add_argument("--token", default=None, help="NOAA CDO token. If omitted, uses NOAA_TOKEN env var.")
    parser.add_argument("--station-id", default=DEFAULT_STATION_ID, help="NOAA station id. Default is Central Park.")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    args = parser.parse_args()
    output = Path(args.output) if args.output else None
    path = save_noaa_daily_weather(args.start, args.end, token=args.token, station_id=args.station_id, output_path=output)
    print(f"Saved NOAA weather to {path}")


if __name__ == "__main__":
    main()

