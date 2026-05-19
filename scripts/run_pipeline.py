from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Citi Bike preprocessing, feature engineering, and model training.")
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional row limit per CSV for quick testing.")
    parser.add_argument("--sample-rows-per-file", type=int, default=None, help="Randomly sample this many rows from each CSV for balanced smaller experiments.")
    parser.add_argument("--total-sample-rows", type=int, default=None, help="Randomly sample about this many total rows, balanced evenly across CSV files.")
    parser.add_argument("--no-train", action="store_true", help="Skip model training.")
    parser.add_argument("--weather", type=str, default=None, help="Optional weather CSV with timestamp and weather columns.")
    parser.add_argument("--holidays", type=str, default=None, help="Optional holiday CSV with a date column.")
    parser.add_argument("--max-train-rows", type=int, default=500000, help="Limit model training rows after full preprocessing to reduce memory use. Use 0 for all rows.")
    parser.add_argument("--fast-models", action="store_true", help="Train only faster models for quicker local experiments.")
    parser.add_argument("--selection-metric", choices=["combined", "MAE"], default="combined", help="Choose final model by combined metric ranking or by lowest MAE.")
    parser.add_argument("--no-rider-models", action="store_true", help="Skip member/casual rider-specific model training.")
    parser.add_argument("--app-recent-hours", type=int, default=168, help="Recent hours per station/rider kept for the Streamlit app context. Use 0 to keep all app context rows.")
    args = parser.parse_args()
    result = run_pipeline(
        sample_rows=args.sample_rows,
        sample_rows_per_file=args.sample_rows_per_file,
        total_sample_rows=args.total_sample_rows,
        weather_path=args.weather,
        holiday_path=args.holidays,
        max_train_rows=None if args.max_train_rows == 0 else args.max_train_rows,
        fast_models=args.fast_models,
        selection_metric=args.selection_metric,
        train_rider_models=not args.no_rider_models,
        train=not args.no_train,
        app_recent_hours=None if args.app_recent_hours == 0 else args.app_recent_hours,
    )
    print(result["summary"].to_string(index=False))
    if result["metrics"] is not None:
        print(result["metrics"]["metrics"].to_string())


if __name__ == "__main__":
    main()
