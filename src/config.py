from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = PROJECT_ROOT / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
EXTERNAL_DIR = OUTPUTS_DIR / "external"

RAW_EXCLUDE_PARTS = {"processed", "models", "outputs", "app", "src", "scripts", ".git", "__pycache__"}

MIN_TRIP_MINUTES = 1
MAX_TRIP_HOURS = 24
GRID_SIZE_DEGREES = 0.01
DEFAULT_TOP_K = 20


def ensure_dirs() -> None:
    for path in [PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR, PREDICTIONS_DIR, EXTERNAL_DIR]:
        path.mkdir(parents=True, exist_ok=True)
