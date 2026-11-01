"""Shared schema and time checks for collection, processing and forecasting."""

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


WEATHER_COLUMNS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "precipitation", "rain", "snowfall",
    "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low",
    "cloud_cover_mid", "cloud_cover_high", "wind_speed_10m", "wind_speed_100m",
    "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m",
    "et0_fao_evapotranspiration", "vapour_pressure_deficit",
    "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm",
    "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm", "soil_moisture_100_to_255cm",
    "weather_code", "is_day",
]
METADATA_COLUMNS = ["time", "city", "latitude", "longitude"]


def validate_hourly_frame(frame: pd.DataFrame, required_columns=()) -> None:
    """Reject missing/duplicate hours instead of treating the next row as +1h."""
    missing = set(["time", *required_columns]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(frame) < 2:
        raise ValueError("At least two hourly rows are required.")
    times = pd.to_datetime(frame["time"], errors="coerce")
    if times.isna().any() or times.duplicated().any():
        raise ValueError("Invalid or duplicate timestamps.")
    if not times.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Data must be sorted and continuous at hourly intervals.")
    if frame.isna().any().any():
        raise ValueError("Missing values remain; do not impute future labels.")
    numeric = frame.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Non-finite numeric values in dataset.")


def load_feature_selection(path: Path, target: str, horizon: int, dataset_path=None) -> list[str]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report["target"] != target or report["horizon_hours"] != horizon:
        raise ValueError("Target/horizon differs from the feature selection run.")
    features = report["selected_features"]
    if not features or len(features) != len(set(features)) or target not in features:
        raise ValueError("Selected features must be unique and include the current target.")
    if set(features) & set(METADATA_COLUMNS):
        raise ValueError("Metadata must not be used as model features.")
    if dataset_path is not None:
        digest = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
        if report.get("processed_sha256") != digest:
            raise ValueError("Dataset differs from its feature selection report; rerun clean_raw.")
    return features


def build_forecast_dataset(frame, features, target, horizon):
    """Pair each input time with a measured target exactly horizon hours later."""
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Forecast horizon must be a positive integer.")
    validate_hourly_frame(frame, features)
    if len(frame) <= horizon:
        raise ValueError("Not enough rows for the requested forecast horizon.")
    if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in features):
        raise ValueError("All model features must be numeric.")
    label = f"{target}_future_{horizon}h"
    data = frame[["time", *features]].copy()
    data["time"] = pd.to_datetime(data["time"])
    data["label_time"] = data["time"].shift(-horizon)
    data[label] = frame[target].shift(-horizon)
    data = data.iloc[:-horizon].reset_index(drop=True)
    return data[["time", "label_time", *features, label]], label
