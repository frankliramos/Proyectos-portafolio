"""
Data Loading Utilities for NASA CMAPSS FD001

Este módulo contiene funciones para cargar y preparar los datos
del dataset NASA CMAPSS (conjunto FD001) para tareas de
mantenimiento predictivo (predicción de RUL).

Author: [Tu Nombre]
Date: 2026-01-28
"""

from pathlib import Path
import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_DIR,
    FD001_TRAIN_FILE,
    FD001_TEST_FILE,
    FD001_RUL_FILE,
)


def _load_fd001_raw(path: Path) -> pd.DataFrame:
    """
    Carga un archivo FD001 (train o test) en formato txt.

    Args:
        path (Path): Ruta al archivo .txt de FD001.

    Returns:
        pd.DataFrame: DataFrame con columnas nombradas.
    """
    # Definición de columnas según especificación NASA CMAPSS
    col_names = [
        "unit_id",
        "time_cycles",
        "op_1",
        "op_2",
        "op_3",
    ]
    # 21 sensores
    sensor_cols = [f"s_{i}" for i in range(1, 22)]
    col_names.extend(sensor_cols)

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=col_names,
        engine="python",
    )

    return df


def load_fd001_train() -> pd.DataFrame:
    """
    Carga el conjunto de entrenamiento FD001 desde RAW_DATA_DIR.

    Returns:
        pd.DataFrame: Datos de entrenamiento sin columna de RUL aún.
    """
    return _load_fd001_raw(FD001_TRAIN_FILE)


def load_fd001_test() -> pd.DataFrame:
    """
    Carga el conjunto de test FD001 desde RAW_DATA_DIR.

    Returns:
        pd.DataFrame: Datos de test sin RUL (RUL viene en archivo separado).
    """
    return _load_fd001_raw(FD001_TEST_FILE)


def load_fd001_rul() -> pd.DataFrame:
    """
    Carga el archivo de RUL (Remaining Useful Life) para FD001 (test).

    Returns:
        pd.DataFrame: DataFrame con una sola columna 'RUL'.
    """
    df_rul = pd.read_csv(
        FD001_RUL_FILE,
        sep=r"\s+",
        header=None,
        names=["RUL"],
        engine="python",
    )
    return df_rul


def add_rul_to_train(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula y agrega la columna RUL al DataFrame de entrenamiento.

    Para cada motor (unit_id), el RUL se define como:
        RUL = max(time_cycles) - time_cycles

    Args:
        train_df (pd.DataFrame): DataFrame de entrenamiento FD001.

    Returns:
        pd.DataFrame: DataFrame con columna adicional 'RUL'.
    """
    df = train_df.copy()

    # Tiempo máximo de cada motor
    max_cycles = df.groupby("unit_id")["time_cycles"].max().reset_index()
    max_cycles.columns = ["unit_id", "max_cycle"]

    # Merge
    df = df.merge(max_cycles, on="unit_id", how="left")
    df["RUL"] = df["max_cycle"] - df["time_cycles"]
    df = df.drop(columns=["max_cycle"])

    return df


def add_true_rul_to_test(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega la columna RUL real al conjunto de test usando el archivo RUL_FD001.

    La NASA entrega para test el RUL restante para cada motor
    en el último ciclo registrado. Se propaga este valor para
    cada fila de ese motor hacia atrás en el tiempo.

    Args:
        test_df (pd.DataFrame): DataFrame de test FD001 (sin RUL).

    Returns:
        pd.DataFrame: DataFrame de test con columna 'RUL' agregada.
    """
    df = test_df.copy()
    df_rul = load_fd001_rul().reset_index()
    df_rul.rename(columns={"index": "unit_id"}, inplace=True)
    df_rul["unit_id"] = df_rul["unit_id"] + 1  # unit_id inicia en 1

    # Obtiene el último ciclo de cada motor en test
    last_cycles = df.groupby("unit_id")["time_cycles"].max().reset_index()
    last_cycles.columns = ["unit_id", "last_cycle"]

    # Merge con RUL proporcionado
    last_cycles = last_cycles.merge(df_rul, on="unit_id", how="left")

    # Mapeo de RUL en el último ciclo
    df = df.merge(
        last_cycles[["unit_id", "last_cycle", "RUL"]], on="unit_id", how="left"
    )

    # Ajustar RUL por ciclo (contando hacia atrás desde el último ciclo)
    df["RUL"] = df["RUL"] + (df["time_cycles"] - df["last_cycle"])
    df = df.drop(columns=["last_cycle"])

    return df
