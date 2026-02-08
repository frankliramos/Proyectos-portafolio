# src/data_loading.py
"""
Módulo de carga y preparación de datos para el dataset NASA CMAPSS (FD001).

Responsabilidades principales:
- Cargar los archivos de texto originales (train/test/RUL).
- Construir la etiqueta de Remaining Useful Life (RUL) para el set de entrenamiento.
- Unir la información de RUL para el set de test utilizando el archivo de RUL.
- Opcionalmente, guardar versiones procesadas en formato columna (Parquet).

Este módulo está pensado para ser usado tanto en notebooks de EDA
como en scripts de entrenamiento de modelos.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .config import (
    FD001_TRAIN_FILE,
    FD001_TEST_FILE,
    FD001_RUL_FILE,
    FD001_PROCESSED_FILE,
    PROCESSED_DATA_DIR,
)


# Definimos nombres de columnas de acuerdo a la descripción del dataset
FD_COLUMN_NAMES = (
    ["unit_id", "time_cycles"]  # columnas 1-2
    + [f"op_setting_{i}" for i in range(1, 4)]  # columnas 3-5
    + [f"sensor_{i}" for i in range(1, 27)]  # columnas 6-26
)


def _load_fd001_raw(
    train_path: str | None = None,
    test_path: str | None = None,
    rul_path: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga los archivos de texto crudos para el subset FD001.
    """
    train_file = str(train_path) if train_path is not None else str(FD001_TRAIN_FILE)
    test_file = str(test_path) if test_path is not None else str(FD001_TEST_FILE)
    rul_file = str(rul_path) if rul_path is not None else str(FD001_RUL_FILE)

    train_df = pd.read_csv(
        train_file,
        sep=r"\s+",
        header=None,
        names=FD_COLUMN_NAMES,
    )

    test_df = pd.read_csv(
        test_file,
        sep=r"\s+",
        header=None,
        names=FD_COLUMN_NAMES,
    )

    rul_df = pd.read_csv(
        rul_file,
        sep=r"\s+",
        header=None,
        names=["RUL"],
    )

    return train_df, test_df, rul_df


def add_rul_to_train(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade la columna de Remaining Useful Life (RUL) al DataFrame de entrenamiento.
    """
    max_cycle_per_unit = (
        train_df.groupby("unit_id")["time_cycles"].max().rename("max_cycle")
    )

    train_with_max = train_df.merge(
        max_cycle_per_unit, left_on="unit_id", right_index=True, how="left"
    )

    train_with_max["RUL"] = train_with_max["max_cycle"] - train_with_max["time_cycles"]

    train_with_rul = train_with_max.drop(columns=["max_cycle"])

    return train_with_rul


def add_rul_to_test_using_rul_file(
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Añade información de RUL "global" al set de test utilizando el archivo RUL_FD001.
    """
    max_cycle_per_unit_test = (
        test_df.groupby("unit_id")["time_cycles"].max().rename("max_cycle")
    )

    test_with_max = test_df.merge(
        max_cycle_per_unit_test, left_on="unit_id", right_index=True, how="left"
    )

    test_with_max["is_last_cycle"] = (
        test_with_max["time_cycles"] == test_with_max["max_cycle"]
    )

    n_units = max_cycle_per_unit_test.shape[0]
    if n_units != rul_df.shape[0]:
        raise ValueError(
            f"Número de motores en test ({n_units}) no coincide con número "
            f"de filas en RUL_FD001 ({rul_df.shape[0]})."
        )

    last_cycle_mask = test_with_max["is_last_cycle"]
    last_cycle_indices = test_with_max[last_cycle_mask].index

    if len(last_cycle_indices) != n_units:
        raise RuntimeError(
            "El número de filas marcadas como último ciclo no coincide con el "
            "número de motores en el set de test."
        )

    test_with_max["true_RUL_at_last_cycle"] = np.nan
    test_with_max.loc[last_cycle_indices, "true_RUL_at_last_cycle"] = rul_df[
        "RUL"
    ].values

    test_with_rul = test_with_max.drop(columns=["max_cycle"])

    return test_with_rul


def load_fd001_prepared(
    save_processed: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga y prepara el dataset FD001, devolviendo dataframes listos para EDA/modelado.
    """
    train_raw, test_raw, rul_df = _load_fd001_raw()

    train_prepared = add_rul_to_train(train_raw)
    test_prepared = add_rul_to_test_using_rul_file(test_raw, rul_df)

    if save_processed:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_prepared.to_parquet(FD001_PROCESSED_FILE, index=False)

    return train_prepared, test_prepared
