from __future__ import annotations

import pandas as pd


def precision_at_k(actual: pd.DataFrame, predicted: pd.DataFrame, key: str, k: int = 20) -> float:
    actual_top = set(actual.sort_values("demand", ascending=False).head(k)[key])
    pred_top = set(predicted.sort_values("predicted_demand", ascending=False).head(k)[key])
    if not pred_top:
        return 0.0
    return len(actual_top & pred_top) / len(pred_top)


def top_k_overlap(actual: pd.DataFrame, predicted: pd.DataFrame, key: str, k: int = 20) -> int:
    actual_top = set(actual.sort_values("demand", ascending=False).head(k)[key])
    pred_top = set(predicted.sort_values("predicted_demand", ascending=False).head(k)[key])
    return len(actual_top & pred_top)

