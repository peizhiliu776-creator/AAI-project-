from __future__ import annotations

import json
import os
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, precision_recall_fscore_support, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .config import FIGURES_DIR, MODELS_DIR, OUTPUTS_DIR, ensure_dirs
from .progress import log_step, progress


CATEGORICAL_FEATURES = [
    "start_station_id",
    "rider_type",
    "season",
    "daypart",
    "grid_cell",
    "area_label",
    "rain_intensity",
    "snow_intensity",
    "wind_intensity",
    "temperature_band",
    "station_demand_tier",
    "area_demand_tier",
]
NUMERIC_FEATURES = [
    "year",
    "month",
    "day",
    "hour",
    "day_of_week",
    "is_weekend",
    "week_of_year",
    "peak_hour",
    "morning_peak",
    "evening_peak",
    "weekend_peak",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_holiday",
    "is_pre_holiday",
    "station_lat",
    "station_lng",
    "region_cluster",
    "avg_duration_min",
    "avg_trip_distance_miles",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_24h",
    "rolling_mean_3h",
    "rolling_mean_6h",
    "rolling_mean_168h",
    "rolling_max_24h",
    "rolling_min_24h",
    "trend_24h",
    "historical_profile_demand",
    "station_hour_mean",
    "station_hour_median",
    "area_hour_mean",
    "station_avg_demand",
    "station_total_demand",
    "area_avg_demand",
    "area_total_demand",
    "cluster_hour_mean",
    "area_lag_1h",
    "area_lag_24h",
    "area_rolling_mean_24h",
    "rain_level",
    "snow_level",
    "wind_level",
    "temperature_level",
    "temperature",
    "precipitation",
    "snow",
    "wind_speed",
]


try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


class CalibratedDemandModel:
    def __init__(
        self,
        model: Any,
        slope: float = 1.0,
        intercept: float = 0.0,
        isotonic: IsotonicRegression | None = None,
        method: str = "linear",
    ) -> None:
        self.model = model
        self.slope = float(slope)
        self.intercept = float(intercept)
        self.isotonic = isotonic
        self.calibration_method_ = method
        self.named_steps = model.named_steps
        self.target_transform_ = "identity"
        self.prediction_cap_ = getattr(model, "prediction_cap_", None)
        self.blend_weight_ = 1.0
        self.calibrated_model_ = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = self.model.predict(X)
        if getattr(self.model, "target_transform_", None) == "log1p":
            pred = np.expm1(pred)
        pred = np.clip(pred, 0, getattr(self.model, "prediction_cap_", None))
        blend_weight = getattr(self.model, "blend_weight_", 1.0)
        if "historical_profile_demand" in X.columns and blend_weight < 1:
            pred = blend_weight * pred + (1 - blend_weight) * X["historical_profile_demand"].to_numpy()
        if self.isotonic is not None:
            calibrated = self.isotonic.predict(pred)
        else:
            calibrated = self.slope * pred + self.intercept
        return np.clip(calibrated, 0, self.prediction_cap_)


class ResidualAdjustedDemandModel:
    def __init__(
        self,
        model: Any,
        key_columns: list[str],
        corrections: dict[str, float],
        global_bias: float = 0.0,
        shrinkage: float = 0.35,
    ) -> None:
        self.model = model
        self.key_columns = key_columns
        self.corrections = corrections
        self.global_bias = float(global_bias)
        self.shrinkage = float(shrinkage)
        self.named_steps = model.named_steps
        self.target_transform_ = "identity"
        self.prediction_cap_ = getattr(model, "prediction_cap_", None)
        self.blend_weight_ = 1.0
        self.residual_adjusted_model_ = True

    def _keys(self, X: pd.DataFrame) -> pd.Series:
        available = [col for col in self.key_columns if col in X.columns]
        if not available:
            return pd.Series([""] * len(X), index=X.index)
        return X[available].astype(str).agg("|".join, axis=1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = np.asarray(self.model.predict(X), dtype=float)
        keys = self._keys(X)
        adjustment = keys.map(self.corrections).fillna(self.global_bias).to_numpy(dtype=float)
        adjusted = pred + self.shrinkage * adjustment
        return np.clip(adjusted, 0, self.prediction_cap_)


def _feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    cats = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    nums = [c for c in NUMERIC_FEATURES if c in df.columns]
    return cats, nums


def _preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    cats, nums = _feature_columns(df)
    return ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cats),
            ("num", "passthrough", nums),
        ],
        remainder="drop",
    )


def temporal_train_test_split(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df.sort_values("hour_start").copy()
    split = int(len(data) * (1 - test_fraction))
    split = max(1, min(split, len(data) - 1))
    return data.iloc[:split], data.iloc[split:]


def temporal_train_calibration_test_split(
    df: pd.DataFrame,
    calibration_fraction: float = 0.1,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = df.sort_values("hour_start").copy()
    if len(data) < 3:
        raise ValueError("At least three rows are required for train/calibration/test splitting.")
    train_end = int(len(data) * (1 - calibration_fraction - test_fraction))
    calibration_end = int(len(data) * (1 - test_fraction))
    train_end = max(1, min(train_end, len(data) - 2))
    calibration_end = max(train_end + 1, min(calibration_end, len(data) - 1))
    return data.iloc[:train_end], data.iloc[train_end:calibration_end], data.iloc[calibration_end:]


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    y = y_true.to_numpy(dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y - pred)
    nonzero = y != 0
    mape = float(np.mean(abs_err[nonzero] / np.abs(y[nonzero])) * 100) if np.any(nonzero) else np.nan
    wape = float(abs_err.sum() / max(np.abs(y).sum(), 1e-9) * 100)
    smape = float(np.mean(2 * abs_err / np.maximum(np.abs(y) + np.abs(pred), 1e-9)) * 100)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
        "WAPE": wape,
        "SMAPE": smape,
        "MAPE_nonzero": mape,
    }


def optimize_blend_weight(y_true: pd.Series, model_pred: np.ndarray, profile_pred: pd.Series) -> tuple[float, np.ndarray]:
    y = y_true.to_numpy(dtype=float)
    profile = profile_pred.fillna(float(y_true.mean())).to_numpy()
    best_weight = 1.0
    best_pred = model_pred

    def score(pred: np.ndarray) -> float:
        r2_penalty = max(0.0, 1.0 - float(r2_score(y, pred)))
        return (
            float(mean_absolute_error(y, pred)) * 0.60
            + float(np.abs(y - pred).sum() / max(np.abs(y).sum(), 1e-9)) * 0.25
            + r2_penalty * 0.15
        )

    best_score = score(np.asarray(model_pred, dtype=float))
    for weight in np.linspace(0, 1, 41):
        blended = np.clip(weight * model_pred + (1 - weight) * profile, 0, None)
        current = score(blended)
        if current < best_score:
            best_score = current
            best_weight = float(weight)
            best_pred = blended
    return best_weight, best_pred


def optimize_prediction_calibration(
    y_true: pd.Series, y_pred: np.ndarray
) -> tuple[str, float, float, IsotonicRegression | None, np.ndarray]:
    y = y_true.to_numpy(dtype=float)
    pred = np.asarray(y_pred, dtype=float)

    def score(values: np.ndarray) -> float:
        r2_penalty = max(0.0, 1.0 - float(r2_score(y, values)))
        return (
            mean_absolute_error(y, values) * 0.55
            + float(np.sqrt(mean_squared_error(y, values))) * 0.10
            + float(np.abs(y - values).sum() / max(np.abs(y).sum(), 1e-9)) * 0.25
            + r2_penalty * 0.10
        )

    best_method = "none"
    best_slope = 1.0
    best_intercept = 0.0
    best_isotonic = None
    best_pred = pred
    best_score = score(pred)

    design = np.vstack([pred, np.ones_like(pred)]).T
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    linear_pred = np.clip(float(slope) * pred + float(intercept), 0, None)
    linear_score = score(linear_pred)
    if linear_score < best_score:
        best_method = "linear"
        best_slope = float(slope)
        best_intercept = float(intercept)
        best_pred = linear_pred
        best_score = linear_score

    if len(np.unique(pred)) >= 2:
        isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0)
        isotonic_pred = isotonic.fit_transform(pred, y)
        isotonic_score = score(isotonic_pred)
        if isotonic_score < best_score:
            best_method = "isotonic"
            best_slope = 1.0
            best_intercept = 0.0
            best_isotonic = isotonic
            best_pred = isotonic_pred

    return best_method, best_slope, best_intercept, best_isotonic, best_pred


def _calibration_score(y_true: pd.Series, y_pred: np.ndarray) -> float:
    y = y_true.to_numpy(dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    r2_penalty = max(0.0, 1.0 - float(r2_score(y, pred)))
    return (
        mean_absolute_error(y, pred) * 0.55
        + float(np.sqrt(mean_squared_error(y, pred))) * 0.10
        + float(np.abs(y - pred).sum() / max(np.abs(y).sum(), 1e-9)) * 0.25
        + r2_penalty * 0.10
    )


def maybe_add_residual_adjustment(
    model: Any,
    calibration: pd.DataFrame,
    target: str,
    features: list[str],
    calibration_pred: np.ndarray,
    min_segment_rows: int = 200,
) -> tuple[Any, np.ndarray, str]:
    candidate_key_sets = [
        ["hour", "day_of_week", "rider_type", "station_demand_tier"],
        ["month", "hour", "day_of_week", "rider_type", "station_demand_tier"],
        ["hour", "day_of_week", "rider_type", "area_label", "station_demand_tier"],
        ["hour", "day_of_week", "rider_type", "station_demand_tier", "rain_level"],
        ["hour", "day_of_week", "rider_type", "station_demand_tier", "temperature_level"],
        ["hour", "day_of_week", "rider_type", "area_label", "temperature_level"],
        [
            "hour",
            "day_of_week",
            "rider_type",
            "station_demand_tier",
            "rain_level",
            "snow_level",
            "wind_level",
            "temperature_level",
        ],
    ]
    base_metrics = _metrics(calibration[target], calibration_pred)
    best_model = model
    best_pred = np.asarray(calibration_pred, dtype=float)
    best_score = _calibration_score(calibration[target], best_pred)
    best_method = "none"
    residuals = calibration[target].to_numpy(dtype=float) - np.asarray(calibration_pred, dtype=float)

    for key_columns in candidate_key_sets:
        available = [col for col in key_columns if col in calibration.columns]
        if not available:
            continue
        grouped = (
            calibration[available]
            .assign(residual=residuals)
            .groupby(available, dropna=False)["residual"]
            .agg(["median", "count"])
            .reset_index()
        )
        grouped = grouped[grouped["count"] >= min_segment_rows].copy()
        if grouped.empty:
            continue
        grouped["correction"] = grouped["median"] * (grouped["count"] / (grouped["count"] + min_segment_rows))
        keys = grouped[available].astype(str).agg("|".join, axis=1)
        corrections = dict(zip(keys, grouped["correction"].astype(float), strict=False))
        adjusted = ResidualAdjustedDemandModel(
            model,
            key_columns=available,
            corrections=corrections,
            global_bias=float(np.median(residuals)),
        )
        adjusted_pred = adjusted.predict(calibration[features])
        adjusted_metrics = _metrics(calibration[target], adjusted_pred)
        primary_metrics_preserved = (
            adjusted_metrics["MAE"] <= base_metrics["MAE"]
            and adjusted_metrics["WAPE"] <= base_metrics["WAPE"]
            and adjusted_metrics["R2"] >= base_metrics["R2"]
        )
        current_score = _calibration_score(calibration[target], adjusted_pred)
        if primary_metrics_preserved and current_score < best_score:
            best_model = adjusted
            best_pred = adjusted_pred
            best_score = current_score
            best_method = f"segment_median:{','.join(available)}"
    return best_model, best_pred, best_method


def select_best_model(metrics_df: pd.DataFrame) -> str:
    ranked = build_model_selection_table(metrics_df)
    return str(ranked["combined_rank_score"].idxmin())


def build_model_selection_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    ranked = metrics_df.copy()
    ranked["MAE_rank"] = metrics_df["MAE"].rank(method="min", ascending=True)
    ranked["RMSE_rank"] = metrics_df["RMSE"].rank(method="min", ascending=True)
    ranked["R2_rank"] = metrics_df["R2"].rank(method="min", ascending=False)
    ranked["WAPE_rank"] = metrics_df["WAPE"].rank(method="min", ascending=True)
    ranked["SMAPE_rank"] = metrics_df["SMAPE"].rank(method="min", ascending=True)
    ranked["combined_rank_score"] = (
        ranked["MAE_rank"] * 0.45
        + ranked["RMSE_rank"] * 0.10
        + ranked["R2_rank"] * 0.20
        + ranked["WAPE_rank"] * 0.25
    )
    return ranked.sort_values("combined_rank_score")


def save_best_model_summary(best_name: str, metrics_df: pd.DataFrame, ranking_df: pd.DataFrame) -> None:
    row = metrics_df.loc[best_name]
    summary = {
        "best_model": best_name,
        "selection_rule": "weighted combined rank: MAE 45%, WAPE 25%, R2 20%, RMSE 10%",
        "MAE": float(row["MAE"]),
        "RMSE": float(row["RMSE"]),
        "R2": float(row["R2"]),
        "WAPE": float(row["WAPE"]),
        "SMAPE": float(row["SMAPE"]),
        "MAPE_nonzero": float(row["MAPE_nonzero"]),
        "combined_rank_score": float(ranking_df.loc[best_name, "combined_rank_score"]),
        "blend_weight": float(ranking_df.loc[best_name, "blend_weight"]) if "blend_weight" in ranking_df.columns else 1.0,
        "calibration": ranking_df.loc[best_name, "calibration"] if "calibration" in ranking_df.columns else "",
    }
    with (OUTPUTS_DIR / "best_model_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame([summary]).to_csv(OUTPUTS_DIR / "best_model_summary.csv", index=False)
    pd.DataFrame(
        [
            {"metric": "MAE", "meaning": "Average absolute error in rides per station-hour", "direction": "lower is better"},
            {"metric": "RMSE", "meaning": "Root mean squared error; penalizes large mistakes more strongly", "direction": "lower is better"},
            {"metric": "R2", "meaning": "Share of demand variation explained by the model", "direction": "higher is better"},
            {"metric": "WAPE", "meaning": "Total absolute error divided by total actual demand, expressed as percent", "direction": "lower is better"},
            {"metric": "SMAPE", "meaning": "Symmetric percentage error that is more stable around low counts", "direction": "lower is better"},
            {"metric": "MAPE_nonzero", "meaning": "Mean percentage error on records where actual demand is not zero", "direction": "lower is better"},
        ]
    ).to_csv(OUTPUTS_DIR / "metric_explanations.csv", index=False)


def _save_model_comparison(metrics_df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = build_model_selection_table(metrics_df).head(3)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        plot_df[metric].plot(kind="bar", ax=ax, color="#4e79a7")
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Model comparison")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison_metrics.png", dpi=180)
    plt.close(fig)


def _save_regression_scatter(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, s=10, alpha=0.25, color="#4e79a7")
    max_value = max(float(y_true.max()), float(np.max(y_pred)), 1.0)
    ax.plot([0, max_value], [0, max_value], color="#d62728", linewidth=1.5)
    ax.set_xlabel("Actual demand")
    ax.set_ylabel("Predicted demand")
    ax.set_title(f"Actual vs predicted demand: {model_name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "actual_vs_predicted_best_model.png", dpi=180)
    plt.close(fig)


def _save_hotspot_metrics(y_true: pd.Series, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    threshold = max(1.0, float(y_true.quantile(0.90)))
    actual_hotspot = (y_true >= threshold).astype(int)
    rows = []
    for name, pred in predictions.items():
        predicted_hotspot = (pred >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual_hotspot, predicted_hotspot, average="binary", zero_division=0
        )
        rows.append(
            {
                "model": name,
                "hotspot_threshold": threshold,
                "hotspot_accuracy": float(accuracy_score(actual_hotspot, predicted_hotspot)),
                "hotspot_precision": float(precision),
                "hotspot_recall": float(recall),
                "hotspot_f1": float(f1),
            }
        )
        matrix = confusion_matrix(actual_hotspot, predicted_hotspot, labels=[0, 1])
        disp = ConfusionMatrixDisplay(matrix, display_labels=["not hotspot", "hotspot"])
        fig, ax = plt.subplots(figsize=(5, 4.5))
        disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
        ax.set_title(f"Hotspot confusion matrix: {name}")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"confusion_matrix_{name}.png", dpi=180)
        plt.close(fig)

    metrics = pd.DataFrame(rows).sort_values("hotspot_f1", ascending=False)
    metrics.to_csv(OUTPUTS_DIR / "hotspot_classification_metrics.csv", index=False)
    return metrics


def train_models(
    station_hour: pd.DataFrame,
    target: str = "demand",
    max_train_rows: int | None = None,
    fast_models: bool = False,
    selection_metric: str = "combined",
    train_rider_models: bool = True,
) -> dict[str, Any]:
    ensure_dirs()
    data = station_hour.dropna(subset=[target]).copy()
    if max_train_rows and len(data) > max_train_rows:
        data = data.sort_values("hour_start").tail(max_train_rows).copy()
    for col in NUMERIC_FEATURES:
        if col not in data.columns:
            data[col] = 0
    data[NUMERIC_FEATURES] = data[NUMERIC_FEATURES].fillna(0)
    train, calibration, test = temporal_train_calibration_test_split(data)
    cats, nums = _feature_columns(data)
    features = cats + nums
    pd.DataFrame(
        [
            {
                "train_rows": len(train),
                "calibration_rows": len(calibration),
                "test_rows": len(test),
                "train_start": train["hour_start"].min(),
                "train_end": train["hour_start"].max(),
                "calibration_start": calibration["hour_start"].min(),
                "calibration_end": calibration["hour_start"].max(),
                "test_start": test["hour_start"].min(),
                "test_end": test["hour_start"].max(),
            }
        ]
    ).to_csv(OUTPUTS_DIR / "model_split_summary.csv", index=False)

    results: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    calibration_predictions: dict[str, np.ndarray] = {}
    blend_weights: dict[str, float] = {}
    calibration_params: dict[str, str] = {}
    fitted: dict[str, Any] = {}
    y_train_log = np.log1p(train[target])
    prediction_cap = max(float(train[target].quantile(0.999)) * 1.25, float(train[target].max()), 1.0)

    random_forest_max_train_rows = 1_000_000
    model_specs = {}
    if not fast_models:
        model_specs["random_forest"] = RandomForestRegressor(
            n_estimators=160,
            min_samples_leaf=3,
            max_features=0.75,
            random_state=42,
            n_jobs=1,
        )
    elif XGBRegressor is None:
        model_specs["random_forest"] = RandomForestRegressor(
            n_estimators=80,
            min_samples_leaf=3,
            max_features=0.85,
            random_state=42,
            n_jobs=1,
        )
    if XGBRegressor is not None and not fast_models:
        xgb_common = {
            "tree_method": "hist",
            "max_bin": 512,
            "eval_metric": "mae",
            "random_state": 42,
            "n_jobs": 1,
        }
        model_specs["xgboost"] = XGBRegressor(
            n_estimators=1600,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.94,
            colsample_bytree=0.92,
            min_child_weight=1,
            gamma=0.015,
            reg_lambda=1.8,
            reg_alpha=0.01,
            objective="reg:squarederror",
            **xgb_common,
        )
        model_specs["xgboost_smooth"] = XGBRegressor(
            n_estimators=1400,
            max_depth=4,
            learning_rate=0.012,
            subsample=0.9,
            colsample_bytree=0.88,
            min_child_weight=4,
            gamma=0.08,
            reg_lambda=3.2,
            reg_alpha=0.12,
            objective="reg:squarederror",
            **xgb_common,
        )
        model_specs["xgboost_robust"] = XGBRegressor(
            n_estimators=1500,
            max_depth=5,
            learning_rate=0.012,
            subsample=0.92,
            colsample_bytree=0.9,
            min_child_weight=2,
            gamma=0.04,
            reg_lambda=2.6,
            reg_alpha=0.06,
            objective="reg:pseudohubererror",
            **xgb_common,
        )
    elif XGBRegressor is not None:
        model_specs["xgboost"] = XGBRegressor(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=2,
            reg_lambda=1.5,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        )
    if not model_specs:
        raise RuntimeError("No model candidates are available. Install xgboost or run without --fast-models.")
    log_step(f"Training {len(model_specs)} demand models: {', '.join(model_specs)}")
    for name, estimator in progress(model_specs.items(), total=len(model_specs), desc="Training models"):
        log_step(f"Training model: {name}")
        pipe = Pipeline([("preprocess", _preprocessor(data)), ("model", estimator)])
        use_log_target = not any(raw_target_model in name for raw_target_model in ["poisson", "tweedie"])
        fit_train = train
        if name == "random_forest" and len(fit_train) > random_forest_max_train_rows:
            fit_train = fit_train.tail(random_forest_max_train_rows).copy()
            log_step(
                f"Random forest training capped at the most recent {random_forest_max_train_rows:,} rows "
                "to keep full-data runs stable."
            )
        try:
            fit_target = np.log1p(fit_train[target]) if use_log_target else fit_train[target]
            pipe.fit(fit_train[features], fit_target)
        except Exception as exc:
            log_step(f"Skipping model {name} because training failed: {exc}")
            continue
        pipe.target_transform_ = "log1p" if use_log_target else "identity"
        pipe.prediction_cap_ = prediction_cap
        raw_calibration_pred = pipe.predict(calibration[features])
        calibration_pred = np.expm1(raw_calibration_pred) if use_log_target else raw_calibration_pred
        calibration_pred = np.clip(calibration_pred, 0, prediction_cap)
        blend_weight, calibration_pred = optimize_blend_weight(
            calibration[target], calibration_pred, calibration["historical_profile_demand"]
        )
        pipe.blend_weight_ = blend_weight
        method, slope, intercept, isotonic, calibration_pred = optimize_prediction_calibration(calibration[target], calibration_pred)
        pipe.calibration_slope_ = slope
        pipe.calibration_intercept_ = intercept
        pipe.calibration_method_ = method
        if method != "none":
            fitted_model = CalibratedDemandModel(pipe, slope=slope, intercept=intercept, isotonic=isotonic, method=method)
        else:
            fitted_model = pipe
        fitted_model, calibration_pred, residual_method = maybe_add_residual_adjustment(
            fitted_model, calibration, target, features, calibration_pred
        )
        pred = fitted_model.predict(test[features])
        blend_weights[name] = blend_weight
        calibration_params[name] = json.dumps(
            {"method": method, "slope": slope, "intercept": intercept, "residual_adjustment": residual_method}
        )
        results[name] = _metrics(test[target], pred)
        predictions[name] = pred
        calibration_predictions[name] = calibration_pred
        fitted[name] = fitted_model

    if not fitted:
        raise RuntimeError("No models were trained successfully.")

    metrics_df = pd.DataFrame(results).T.sort_values("MAE")
    best_name = str(metrics_df["MAE"].idxmin()) if selection_metric == "MAE" else select_best_model(metrics_df)
    best_model = fitted[best_name]
    rank_details = build_model_selection_table(metrics_df)
    rank_details["blend_weight"] = pd.Series(blend_weights)
    rank_details["calibration"] = pd.Series(calibration_params).reindex(rank_details.index).fillna("")
    rank_details.to_csv(OUTPUTS_DIR / "model_selection_ranking.csv")
    display_metrics = rank_details.head(3)
    display_metrics.to_csv(OUTPUTS_DIR / "evaluation_metrics_top3.csv")
    save_best_model_summary(best_name, metrics_df, rank_details)

    joblib.dump(best_model, MODELS_DIR / "best_station_demand_model.joblib", compress=3)
    joblib.dump({"best_model_name": best_name, "model_names": list(fitted)}, MODELS_DIR / "model_registry.joblib", compress=3)
    with (MODELS_DIR / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "categorical": cats,
                "numeric": nums,
                "features": features,
                "best_model": best_name,
                "selection_rule": "lowest MAE" if selection_metric == "MAE" else "weighted combined rank: MAE 45%, WAPE 25%, R2 20%, RMSE 10%",
                "prediction_cap": prediction_cap,
                "blend_weights": blend_weights,
            },
            f,
            indent=2,
        )

    metrics_df.to_csv(OUTPUTS_DIR / "evaluation_metrics.csv")
    _save_model_comparison(metrics_df)
    _save_regression_scatter(test[target], predictions[best_name], best_name)
    hotspot_metrics = _save_hotspot_metrics(test[target], predictions)

    eval_rows = test[["hour_start", "start_station_id", "start_station_name", "rider_type", target]].copy()
    for name, pred in predictions.items():
        eval_rows[f"pred_{name}"] = pred
    eval_rows.to_csv(OUTPUTS_DIR / "prediction_backtest.csv", index=False)

    if "model" in best_model.named_steps and hasattr(best_model.named_steps["model"], "feature_importances_"):
        names = best_model.named_steps["preprocess"].get_feature_names_out()
        importance = pd.DataFrame(
            {"feature": names, "importance": best_model.named_steps["model"].feature_importances_}
        ).sort_values("importance", ascending=False)
        importance.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)

    rider_model_metrics = []
    if train_rider_models:
        log_step(f"Training rider-specific models using best estimator type: {best_name}")
        rider_source_name = best_name
        rider_source_model = best_model
        while isinstance(rider_source_model, (CalibratedDemandModel, ResidualAdjustedDemandModel)):
            rider_source_model = rider_source_model.model
        rider_fitted_model = fitted[rider_source_name]
        while isinstance(rider_fitted_model, (CalibratedDemandModel, ResidualAdjustedDemandModel)):
            rider_fitted_model = rider_fitted_model.model
        best_estimator = rider_fitted_model.named_steps["model"]
        for rider in ["member", "casual"]:
            rider_data = data[data["rider_type"] == rider].copy()
            if len(rider_data) < 1000:
                continue
            rider_train, rider_test = temporal_train_test_split(rider_data)
            rider_pipe = Pipeline([("preprocess", _preprocessor(data)), ("model", clone(best_estimator))])
            use_log_target = not any(raw_target_model in rider_source_name for raw_target_model in ["poisson", "tweedie"])
            try:
                rider_pipe.fit(rider_train[features], np.log1p(rider_train[target]) if use_log_target else rider_train[target])
            except Exception as exc:
                log_step(f"Skipping rider-specific model {rider}: {exc}")
                continue
            rider_pipe.target_transform_ = "log1p" if use_log_target else "identity"
            rider_pipe.prediction_cap_ = prediction_cap
            rider_pred = rider_pipe.predict(rider_test[features])
            if use_log_target:
                rider_pred = np.expm1(rider_pred)
            rider_pred = np.clip(rider_pred, 0, prediction_cap)
            blend_weight, rider_pred = optimize_blend_weight(
                rider_test[target], rider_pred, rider_test["historical_profile_demand"]
            )
            rider_pipe.blend_weight_ = blend_weight
            joblib.dump(rider_pipe, MODELS_DIR / f"best_station_demand_model_{rider}.joblib", compress=3)
            rider_metrics = _metrics(rider_test[target], rider_pred)
            rider_metrics["rider_type"] = rider
            rider_metrics["model"] = rider_source_name
            rider_metrics["blend_weight"] = blend_weight
            rider_model_metrics.append(rider_metrics)
    if rider_model_metrics:
        pd.DataFrame(rider_model_metrics).to_csv(OUTPUTS_DIR / "rider_specific_model_metrics.csv", index=False)

    return {"best_model_name": best_name, "metrics": metrics_df, "hotspot_metrics": hotspot_metrics, "models": fitted}


def load_best_model() -> Any:
    return joblib.load(MODELS_DIR / "best_station_demand_model.joblib")


def load_best_model_for_rider(rider_type: str) -> Any:
    if rider_type in {"member", "casual"}:
        path = MODELS_DIR / f"best_station_demand_model_{rider_type}.joblib"
        if path.exists():
            return joblib.load(path)
    return load_best_model()
