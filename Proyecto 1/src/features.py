"""
Feature Engineering para NASA CMAPSS FD001

Este módulo contiene funciones para crear features derivados
a partir de los datos de sensores para mejorar la predicción de RUL.

Incluye:
- Lag features: Valores pasados de sensores (t-1, t-3, t-5)
- Rolling features: Estadísticas móviles (media, desviación estándar)
- Trend features: Diferencias respecto a medias móviles

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
    Crea lag features para los sensores especificados.

    Args:
        df: DataFrame con datos de sensores
        sensor_cols: Lista de columnas de sensores
        lags: Lista de lags a crear

    Returns:
        DataFrame con lag features agregados
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
    Crea rolling statistics (mean, std) para los sensores especificados.

    Args:
        df: DataFrame con datos de sensores
        sensor_cols: Lista de columnas de sensores
        windows: Lista de ventanas para rolling statistics

    Returns:
        DataFrame con rolling features agregados
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
    Crea trend features (diferencia entre valor actual y rolling mean).

    Args:
        df: DataFrame con datos de sensores
        sensor_cols: Lista de columnas de sensores
        windows: Lista de ventanas para calcular trends

    Returns:
        DataFrame con trend features agregados
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
    Aplica todas las transformaciones de feature engineering.

    Args:
        df: DataFrame con datos crudos
        lag_features: Lista de lags a crear
        rolling_windows: Lista de ventanas para rolling statistics
        trend_windows: Lista de ventanas para trends

    Returns:
        DataFrame con todos los features creados
    """
    # Identificar columnas de sensores
    sensor_cols = [col for col in df.columns if col.startswith("s_")]

    print(f"   - Sensores detectados: {len(sensor_cols)}")

    # Aplicar transformaciones
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
