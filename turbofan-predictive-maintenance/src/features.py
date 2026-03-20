"""
Feature Engineering for NASA CMAPSS FD001

This module contains functions to create derived features from sensor data
to improve RUL prediction.

Includes:
- Lag features: past sensor values (t-1, t-3, t-5)
- Rolling features: moving statistics (mean, standard deviation)
- Trend features: differences against rolling means

Author: Franklin Ramos
Date: 2026-02-03
"""

import pandas as pd
import numpy as np
from typing import List


def create_lag_features(
    df: pd.DataFrame, sensor_cols: List[str], lags: List[int] = [1, 3, 5]
) -> pd.DataFrame:
    """
    Create lag features for the specified sensors.

    Args:
        df: DataFrame with sensor data
        sensor_cols: List of sensor columns
        lags: List of lags to create

    Returns:
        DataFrame with lag features added
    """
    df_result = df.copy()

    for sensor in sensor_cols:
        if sensor in df.columns:
            for lag in lags:
                df_result[f"{sensor}_lag_{lag}"] = df_result.groupby("unit_id")[
                    sensor
                ].shift(lag)

    return df_result


def create_rolling_features(
    df: pd.DataFrame, sensor_cols: List[str], windows: List[int] = [5, 10, 20]
) -> pd.DataFrame:
    """
    Create rolling statistics (mean, std) for the specified sensors.

    Args:
        df: DataFrame with sensor data
        sensor_cols: List of sensor columns
        windows: List of windows for rolling statistics

    Returns:
        DataFrame with rolling features added
    """
    df_result = df.copy()

    for sensor in sensor_cols:
        if sensor in df.columns:
            for window in windows:
                # Rolling mean
                df_result[f"{sensor}_rolling_mean_{window}"] = df_result.groupby(
                    "unit_id"
                )[sensor].transform(lambda x: x.rolling(window, min_periods=1).mean())

                # Rolling std
                df_result[f"{sensor}_rolling_std_{window}"] = df_result.groupby(
                    "unit_id"
                )[sensor].transform(lambda x: x.rolling(window, min_periods=1).std())

    return df_result


def create_trend_features(
    df: pd.DataFrame, sensor_cols: List[str], windows: List[int] = [5, 10]
) -> pd.DataFrame:
    """
    Create trend features (difference between current value and rolling mean).

    Args:
        df: DataFrame with sensor data
        sensor_cols: List of sensor columns
        windows: List of windows for trend calculation

    Returns:
        DataFrame with trend features added
    """
    df_result = df.copy()

    for sensor in sensor_cols:
        if sensor in df.columns:
            for window in windows:
                rolling_mean = df_result.groupby("unit_id")[sensor].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                df_result[f"{sensor}_trend_{window}"] = df_result[sensor] - rolling_mean

    return df_result


def prepare_features(
    df: pd.DataFrame,
    lag_features: List[int] = [1, 3, 5],
    rolling_windows: List[int] = [5, 10, 20],
    trend_windows: List[int] = [5, 10],
) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.

    Args:
        df: DataFrame with raw data
        lag_features: List of lags to create
        rolling_windows: List of windows for rolling statistics
        trend_windows: List of windows for trends

    Returns:
        DataFrame with all features created
    """
    # Identify sensor columns
    sensor_cols = [col for col in df.columns if col.startswith("s_")]

    print(f"   - Sensores detectados: {len(sensor_cols)}")

    # Apply transformations
    df_features = df.copy()

    # Lag features
    print(f"   - Creando lag features: {lag_features}")
    df_features = create_lag_features(df_features, sensor_cols, lag_features)

    # Rolling features
    print(f"   - Creando rolling features: {rolling_windows}")
    df_features = create_rolling_features(df_features, sensor_cols, rolling_windows)

    # Trend features
    print(f"   - Creando trend features: {trend_windows}")
    df_features = create_trend_features(df_features, sensor_cols, trend_windows)

    return df_features
